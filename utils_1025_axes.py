from __future__ import annotations
import os
import time
from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

# -------------------------------
# Quaternion helpers
# -------------------------------
def quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    R = np.array([
        [1 - 2 * (y * y + z * z),      2 * (x * y - z * w),      2 * (x * z + y * w)],
        [2 * (x * y + z * w),          1 - 2 * (x * x + z * z),  2 * (y * z - x * w)],
        [2 * (x * z - y * w),          2 * (y * z + x * w),      1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return R

def quaternion_to_rpy(q: Sequence[float]) -> np.ndarray:
    R = quaternion_to_rotation_matrix(*q)
    return pin.rpy.matrixToRpy(R)

# -------------------------------
# Model / geometry
# -------------------------------
def load_robot_models(
    urdf_path: str,
    package_dirs: Sequence[str] | None = None,
    filter_visual_to_upper_body: bool = True,
    upper_body_joint_names: Sequence[str] | None = None,
):
    if package_dirs is None:
        package_dirs = ["./airexo/urdf_models/robot"]

    model, collision_model, visual_model_complete = pin.buildModelsFromUrdf(
        urdf_path,
        package_dirs=list(package_dirs),
        geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL],
    )
    data = model.createData()
    if not filter_visual_to_upper_body:
        return model, data, collision_model, visual_model_complete

    if upper_body_joint_names is None:
        upper_body_joint_names = [
            "r_joint1","r_joint2","r_joint3","r_joint4","r_joint5","r_joint6","r_joint7",
            "r_hand_joint_cylinder","cylinder_to_r_gripper_joint",
            "r_Link1","r_gripper_Link11","r_gripper_Link2","r_gripper_Link22",
            "l_joint1","l_joint2","l_joint3","l_joint4","l_joint5","l_joint6","l_joint7",
            "l_hand_joint_cylinder","cylinder_to_l_gripper_joint",
            "l_Link1","l_gripper_Link11","l_gripper_Link2","l_gripper_Link22",
        ]

    upper_ids = [model.getJointId(nm) for nm in upper_body_joint_names if model.existJointName(nm)]
    visual_model = pin.GeometryModel()
    for geom in visual_model_complete.geometryObjects:
        if geom.parentJoint in upper_ids:
            visual_model.addGeometryObject(geom.copy())

    print(f"✅ 成功构建完整机器人模型，自由度: {model.nq}")
    print(f"✅ 上半身几何体数量: {len(visual_model.geometryObjects)}")
    return model, data, collision_model, visual_model

def build_active_indices(model: pin.Model, joint_names: Sequence[str]) -> List[int]:
    idxs = []
    for nm in joint_names:
        jid = model.getJointId(nm)
        j = model.joints[jid]
        idxs.extend(list(range(j.idx_q, j.idx_q + j.nq)))
    return idxs

# -------------------------------
# CSV 读取
# -------------------------------
def read_dualarm_csv(right_csv: str, left_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    right_df = pd.read_csv(right_csv)
    left_df  = pd.read_csv(left_csv)

    def _parse(df):
        cols = list(df.columns)
        if any(c.startswith("arm1_") for c in cols):
            prefix = "arm1_"
        elif any(c.startswith("arm2_") for c in cols):
            prefix = "arm2_"
        else:
            raise ValueError(f"❌ CSV {cols} 中无 arm1_ 或 arm2_ 前缀")
        pos  = df[[f"{prefix}px", f"{prefix}py", f"{prefix}pz"]].to_numpy()
        quat = df[[f"{prefix}qw", f"{prefix}qx", f"{prefix}qy", f"{prefix}qz"]].to_numpy()
        eul  = np.array([pin.rpy.matrixToRpy(quaternion_to_rotation_matrix(*q)) for q in quat])
        return pos, eul

    right_positions, right_euls = _parse(right_df)
    left_positions, left_euls = _parse(left_df)
    return right_positions, right_euls, left_positions, left_euls

# -------------------------------
# IK core
# -------------------------------
def compute_ik(model, data, target_pos, target_eul, q_init, active_idxs, end_effector_name,
               max_iter=2000, eps=1e-4, stall_threshold=50, damp_base=1e-4, alpha=0.1):
    frame_id = model.getFrameId(end_effector_name)
    target_R = pin.rpy.rpyToMatrix(target_eul)
    target_pose = pin.SE3(target_R, np.asarray(target_pos))

    q = q_init.copy()
    prev_err = np.inf
    stall = 0
    best_q = q.copy()
    min_err = np.inf
    I6 = np.eye(6)

    for _ in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        cur_pose = data.oMf[frame_id]
        err_vec = pin.log(target_pose.inverse() * cur_pose).vector
        err = np.linalg.norm(err_vec)
        if err < min_err:
            min_err, best_q = err, q.copy()
        if err < eps:
            return True, q, err
        if abs(err - prev_err) < 1e-6:
            stall += 1
            if stall > stall_threshold:
                return False, best_q, min_err
        else:
            stall = 0
        prev_err = err
        J = pin.computeFrameJacobian(model, data, q, frame_id)
        J_act = J[:, active_idxs]
        damp = min(max(damp_base, err * 0.05), 5e-2)
        dq_act = J_act.T @ np.linalg.solve(J_act @ J_act.T + damp * I6, err_vec)
        dq = np.zeros(model.nv)
        dq[list(active_idxs)] = dq_act
        q = pin.integrate(model, q, -alpha * dq)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
    return False, best_q, min_err

def select_best_solution_by_error(sols, errs, q_prev, active_idxs, per_joint_max_delta,
                                  total_max_delta, lambda_weight):
    valid, v_errs, v_deltas = [], [], []
    a = np.array(list(active_idxs))
    for q_sol, err in zip(sols, errs):
        delta = q_sol[a] - q_prev[a]
        max_d = np.max(np.abs(delta))
        if max_d <= per_joint_max_delta and np.linalg.norm(delta) <= total_max_delta:
            valid.append(q_sol)
            v_errs.append(err)
            v_deltas.append(np.linalg.norm(delta))
    if valid:
        scores = [e + lambda_weight * d for e, d in zip(v_errs, v_deltas)]
        k = int(np.argmin(scores))
        return valid[k], v_errs[k]
    else:
        k = int(np.argmin(errs))
        print(f"⚠️ 无平滑解，回退误差最小 (误差:{errs[k]:.4f})")
        return sols[k], errs[k]


# -------------------------------
# Meshcat 可视化
# -------------------------------
def init_meshcat(model, c_model, v_model):
    viz = MeshcatVisualizer(model, c_model, v_model)
    viz.initViewer(); viz.loadViewerModel(); viz.viewer.open()
    return viz

def look_at(viz, camera_pos, target_pos, up=(0,0,1)):
    c = np.asarray(camera_pos, float)
    t = np.asarray(target_pos, float)
    T = np.eye(4); T[:3, 3] = t
    viz.viewer["/Cameras/default"].set_transform(T)
    off = c - t
    viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", off.tolist())
    fwd = (t - c) / np.linalg.norm(t - c)
    up = np.asarray(up, float)
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    true_up = np.cross(fwd, right)
    R = np.eye(4); R[:3,0]=right; R[:3,1]=true_up; R[:3,2]=-fwd
    viz.viewer["/Cameras/default/rotated"].set_transform(R)

def draw_target_sphere(viz, parent:str, name:str, position:np.ndarray, radius:float=0.05, color:int=0xff0000):
    """在指定父节点下绘制球体"""
    viz.viewer[parent][name].set_object(
        g.Sphere(radius),
        g.MeshLambertMaterial(color=color, opacity=0.8)
    )
    viz.viewer[parent][name].set_transform(tf.translation_matrix(position))

def draw_target_axes(viz, parent:str, name_prefix:str, pose:pin.SE3, axis_len:float=0.15):
    px = np.array([[0, axis_len], [0, 0], [0, 0]])
    py = np.array([[0, 0], [0, axis_len], [0, 0]])
    pz = np.array([[0, 0], [0, 0], [0, axis_len]])
    viz.viewer[parent][f"{name_prefix}/x"].set_object(
        g.Line(g.PointsGeometry(px), g.MeshBasicMaterial(color=0xff0000)))
    viz.viewer[parent][f"{name_prefix}/y"].set_object(
        g.Line(g.PointsGeometry(py), g.MeshBasicMaterial(color=0x00ff00)))
    viz.viewer[parent][f"{name_prefix}/z"].set_object(
        g.Line(g.PointsGeometry(pz), g.MeshBasicMaterial(color=0x0000ff)))
    M = pose.homogeneous
    viz.viewer[parent][f"{name_prefix}/x"].set_transform(M)
    viz.viewer[parent][f"{name_prefix}/y"].set_transform(M)
    viz.viewer[parent][f"{name_prefix}/z"].set_transform(M)

# -------------------------------
# 单帧求解
# -------------------------------
def solve_one_frame(model, data, q_curr, active_idxs, pos, eul, end_eff,
                    ik_attempts=20, noise_level=0.0,
                    per_joint_max_delta=0.10, total_max_delta=0.4, lambda_weight=0.2):
    prev = q_curr.copy()
    sols, errs = [], []
    for _ in range(int(ik_attempts)):
        q_init = q_curr.copy()
        noise = (np.random.rand(len(active_idxs))-0.5)*noise_level
        q_init[list(active_idxs)] = np.clip(
            q_init[list(active_idxs)] + noise,
            model.lowerPositionLimit[list(active_idxs)],
            model.upperPositionLimit[list(active_idxs)],
        )
        ok, q_res, err = compute_ik(model, data, pos, eul, q_init, active_idxs, end_eff)
        sols.append(q_res); errs.append(err)
    q_sel, e_sel = select_best_solution_by_error(
        sols, errs, prev, active_idxs,
        per_joint_max_delta, total_max_delta, lambda_weight
    )
    return q_sel, float(e_sel)

# -------------------------------
# ✅ 主函数：分别在两臂坐标系中求解并显示 <<< 修改点 >>>
# -------------------------------
def run_dual_arm_trajectory(
    urdf_path:str,
    right_csv:str,
    left_csv:str,
    save_joint_csv:bool=True,
    right_end_effector:str="r_gripper_base_link",
    left_end_effector:str="l_gripper_base_link",
    right_active_joint_names:Sequence[str]=("r_joint1","r_joint2","r_joint3","r_joint4","r_joint5","r_joint6","r_joint7"),
    left_active_joint_names:Sequence[str]=("l_joint1","l_joint2","l_joint3","l_joint4","l_joint5","l_joint6","l_joint7"),
    # 手眼变换
    right_qw:float=1.0, right_qx:float=0, right_qy:float=0, right_qz:float=0,
    right_tx:float=0, right_ty:float=0, right_tz:float=0,
    left_qw:float=1.0, left_qx:float=0, left_qy:float=0, left_qz:float=0,
    left_tx:float=0, left_ty:float=0, left_tz:float=0,
    ik_attempts:int=20, noise_level:float=0.01, per_joint_max_delta:float=0.15,
    total_max_delta:float=0.4, lambda_weight:float=0.1, sleep_each:float=0.004,
    package_dirs:Sequence[str]|None=None, set_initial_camera:bool=True,
):
    model,data,c_model,v_model = load_robot_models(urdf_path, package_dirs, True)
    r_pos, r_eul, l_pos, l_eul = read_dualarm_csv(right_csv, left_csv)
    N = min(len(r_pos), len(l_pos))
    print(f"读取轨迹成功，共 {N} 帧")

    right_idxs = build_active_indices(model, right_active_joint_names)
    left_idxs  = build_active_indices(model, left_active_joint_names)
    q = pin.neutral(model)

    q[right_idxs] = [0.71,1.07,1.01,1.32,-0.07,0.32,-1.39]
    q[left_idxs]  = [-0.71,-1.07,-1.01,-1.32,0.07,-0.32,1.39]

    viz = init_meshcat(model,c_model,v_model)
    if set_initial_camera:
        look_at(viz, [-0.0032,0.20,1.8026], [-0.0032,0.6003,1.1526])

    # 创建两个局部根节点 <<< 修改点 >>>
    viz.viewer["dualarm_scene/right_arm_base"].set_transform(np.eye(4))
    viz.viewer["dualarm_scene/left_arm_base"].set_transform(np.eye(4))

    # 手眼变换
    R_bc_r = quaternion_to_rotation_matrix(right_qw, right_qx, right_qy, right_qz)
    t_bc_r = np.array([right_tx,right_ty,right_tz])
    R_bc_l = quaternion_to_rotation_matrix(left_qw, left_qx, left_qy, left_qz)
    t_bc_l = np.array([left_tx,left_ty,left_tz])

    all_rows=[]; dt=1.0/60.0

    for i in range(200,N):
        print(f"\n--- Frame {i} ---")

        # 在各臂坐标系中目标姿态（不再转到世界坐标系） <<< 修改点 >>>
        r_b_p = R_bc_r @ r_pos[i] + t_bc_r
        l_b_p = R_bc_l @ l_pos[i] + t_bc_l

        R_fix = pin.rpy.rpyToMatrix(0, np.pi/2, 0)  # x→z
        r_b_R = R_bc_r @ pin.rpy.rpyToMatrix(r_eul[i]) @ R_fix
        l_b_R = R_bc_l @ pin.rpy.rpyToMatrix(l_eul[i]) @ R_fix

        r_b_eul = pin.rpy.matrixToRpy(r_b_R)
        l_b_eul = pin.rpy.matrixToRpy(l_b_R)

        qr, er = solve_one_frame(model, data, q, right_idxs, r_b_p, r_b_eul, right_end_effector,
                                 ik_attempts, noise_level, per_joint_max_delta,
                                 total_max_delta, lambda_weight)
        q[right_idxs] = qr[right_idxs]
        ql, el = solve_one_frame(model, data, q, left_idxs, l_b_p, l_b_eul, left_end_effector,
                                 ik_attempts, noise_level, per_joint_max_delta,
                                 total_max_delta, lambda_weight)
        q[left_idxs] = ql[left_idxs]

        print(f"误差:右 {er:.4f}, 左 {el:.4f}")

        # # 显示：直接在两个局部系中画目标球及坐标轴
        # draw_target_sphere(viz,"dualarm_scene/right_arm_base","right_target_sphere",r_b_p,0.05,0xff0000)
        # draw_target_axes(viz,"dualarm_scene/right_arm_base","right_target_axes",pin.SE3(r_b_R,r_b_p))
        # draw_target_sphere(viz,"dualarm_scene/left_arm_base","left_target_sphere",l_b_p,0.05,0x0000ff)
        # draw_target_axes(viz,"dualarm_scene/left_arm_base","left_target_axes",pin.SE3(l_b_R,l_b_p))

        # 显示整机姿态
        viz.display(q)

        # 获取当前frame下两个base的变换
        frame_id_r_base = model.getFrameId("r_base_link1")
        frame_id_l_base = model.getFrameId("l_base_link1")

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        r_base_pose = data.oMf[frame_id_r_base]
        l_base_pose = data.oMf[frame_id_l_base]

        # # 绘制这两个坐标系（每帧刷新一次）
        # draw_target_axes(viz, "dualarm_scene", "r_base_link1_axes", r_base_pose, axis_len=0.1)
        # draw_target_axes(viz, "dualarm_scene", "l_base_link1_axes", l_base_pose, axis_len=0.1)

        # 获取左右末端当前姿态
        frame_id_r_ee = model.getFrameId(right_end_effector)
        frame_id_l_ee = model.getFrameId(left_end_effector)
        r_ee_pose = data.oMf[frame_id_r_ee]
        l_ee_pose = data.oMf[frame_id_l_ee]

        # 固定旋转，让夹爪朝向 z 方向（原 x→z）
        R_fix = pin.rpy.rpyToMatrix(0, np.pi/2, 0)
        r_corrected_pose = pin.SE3(r_ee_pose.rotation @ R_fix, r_ee_pose.translation)
        l_corrected_pose = pin.SE3(l_ee_pose.rotation @ R_fix, l_ee_pose.translation)

        # 画出修正后的末端坐标系，看朝向是否为 z 轴
        draw_target_axes(viz, "dualarm_scene", "r_gripper_corrected_axes", r_corrected_pose, axis_len=0.15)
        draw_target_axes(viz, "dualarm_scene", "l_gripper_corrected_axes", l_corrected_pose, axis_len=0.15)

        all_rows.append([i * dt] + q.tolist())
        time.sleep(float(sleep_each))

    if save_joint_csv:
        os.makedirs("joint_data", exist_ok=True)
        cols = ["timestamp"] + [f"q{j}" for j in range(model.nq)]
        pd.DataFrame(all_rows, columns=cols).to_csv("joint_data/solved_joint_trajectory_local.csv", index=False)
        print("✅ Joint 数据已保存到 joint_data/solved_joint_trajectory_local.csv")

# -------------------------------
# 入口
# -------------------------------
if __name__ == "__main__":
    run_dual_arm_trajectory(
        urdf_path="./airexo/urdf_models/robot/true_robot.urdf",
        right_csv="./train_video/arm2_poses.csv",
        left_csv="./train_video/arm1_poses.csv",
    )