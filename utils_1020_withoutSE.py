"""
Dual-arm IK utilities (无手眼变换版)。

功能：
- 直接使用 CSV 中的 arm1/arm2 位姿 (px/py/pz + 四元数 qw/qx/qy/qz)
- 不再进行 R_bc @ p + t_bc 之类的变换，直接假设输入为世界坐标
- IK 解算并在 Meshcat 中可视化目标点(球+坐标轴)
- 最终保存 joint 数据到 joint_data/solved_joint_trajectory2.csv
"""
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
    """Convert quaternion (w,x,y,z) to rotation matrix."""
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def quaternion_to_rpy(q: Sequence[float]) -> np.ndarray:
    """Convert quaternion (w,x,y,z) → RPY (pinocchio顺序)."""
    R = quaternion_to_rotation_matrix(*q)
    return pin.rpy.matrixToRpy(R)


# -------------------------------
# Model helpers
# -------------------------------

def load_robot_models(urdf_path: str, package_dirs: Sequence[str] | None = None,
                      filter_visual_to_upper_body: bool = True,
                      upper_body_joint_names: Sequence[str] | None = None):
    """加载机器人URDF并构建上半身可视化模型。"""
    if package_dirs is None:
        package_dirs = ["./airexo/urdf_models/robot"]

    model, collision_model, visual_model_complete = pin.buildModelsFromUrdf(
        urdf_path, package_dirs=list(package_dirs),
        geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL])
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

    ids = [model.getJointId(nm) for nm in upper_body_joint_names if model.existJointName(nm)]
    v_model = pin.GeometryModel()
    for geom in visual_model_complete.geometryObjects:
        if geom.parentJoint in ids:
            v_model.addGeometryObject(geom.copy())

    print(f"✅ 成功加载模型，自由度: {model.nq}, 上半身视觉物体数: {len(v_model.geometryObjects)}")
    return model, data, collision_model, v_model


def build_active_indices(model: pin.Model, names: Sequence[str]) -> List[int]:
    idxs = []
    for nm in names:
        jid = model.getJointId(nm)
        joint = model.joints[jid]
        idxs.extend(range(joint.idx_q, joint.idx_q + joint.nq))
    return idxs


# -------------------------------
# CSV 读取
# -------------------------------

def read_dualarm_csv(right_csv: str, left_csv: str):
    """严格按传入顺序解析：第一个为右臂CSV，第二个为左臂CSV。"""
    right_df = pd.read_csv(right_csv)
    left_df = pd.read_csv(left_csv)

    def parse(df):
        cols = list(df.columns)
        if any(c.startswith("arm1_") for c in cols):
            pre = "arm1_"
        elif any(c.startswith("arm2_") for c in cols):
            pre = "arm2_"
        else:
            raise ValueError(f"❌ CSV 列名无 arm1_/arm2_ 前缀: {cols}")
        pos = df[[f"{pre}px", f"{pre}py", f"{pre}pz"]].to_numpy()
        quat = df[[f"{pre}qw", f"{pre}qx", f"{pre}qy", f"{pre}qz"]].to_numpy()
        eul = np.array([pin.rpy.matrixToRpy(quaternion_to_rotation_matrix(*q)) for q in quat])
        return pos, eul

    r_pos, r_eul = parse(right_df)
    l_pos, l_eul = parse(left_df)
    return r_pos, r_eul, l_pos, l_eul


# -------------------------------
# IK core
# -------------------------------

def compute_ik(model, data, target_pos, target_eul, q_init, active_idxs, ee_name,
               max_iter=2000, eps=1e-3, stall_threshold=50, damp_base=1e-4, alpha=0.1):
    frame_id = model.getFrameId(ee_name)
    target_R = pin.rpy.rpyToMatrix(target_eul)
    target_pose = pin.SE3(target_R, np.asarray(target_pos))
    q = q_init.copy()
    best_q = q.copy()
    min_err = np.inf
    prev_err = np.inf
    stall = 0
    I6 = np.eye(6)

    for _ in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        cur = data.oMf[frame_id]
        err_vec = pin.log(target_pose.inverse() * cur).vector
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
        J_a = J[:, active_idxs]
        damp = min(max(damp_base, err * 0.05), 5e-2)
        dq_a = J_a.T @ np.linalg.solve(J_a @ J_a.T + damp * I6, err_vec)
        dq = np.zeros(model.nv); dq[list(active_idxs)] = dq_a
        q = pin.integrate(model, q, -alpha * dq)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
    return False, best_q, min_err


def select_best_solution_by_error(sols, errs, q_prev, active_idxs, per_joint_max, total_max, w):
    valid = []
    v_err, v_del = [], []
    a = np.array(list(active_idxs))
    for q_sol, err in zip(sols, errs):
        delta = q_sol[a] - q_prev[a]
        max_d = np.max(np.abs(delta))
        if max_d <= per_joint_max and np.linalg.norm(delta) <= total_max:
            valid.append(q_sol); v_err.append(err); v_del.append(np.linalg.norm(delta))
    if valid:
        scores = [e + w*d for e, d in zip(v_err, v_del)]
        k = int(np.argmin(scores))
        return valid[k], v_err[k]
    else:
        k = int(np.argmin(errs))
        print(f"⚠️ 无平滑解，回退误差最小: {errs[k]:.4f}")
        return sols[k], errs[k]


# -------------------------------
# Meshcat 视图 + 可视化目标
# -------------------------------

def init_meshcat(model, c_model, v_model):
    viz = MeshcatVisualizer(model, c_model, v_model)
    viz.initViewer(); viz.loadViewerModel(); viz.viewer.open()
    return viz


def look_at(viz, cam=[-0.0032, 0.01, 1.7826], target=[-0.0032, -0.5903, 1.3026]):
    c = np.asarray(cam); t = np.asarray(target)
    T = np.eye(4); T[:3,3]=t
    viz.viewer["/Cameras/default"].set_transform(T)
    off = c-t
    viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", off.tolist())
    fwd = (t - c)/np.linalg.norm(t-c)
    right = np.cross([0,0,1], fwd); right/=np.linalg.norm(right)
    up = np.cross(fwd, right)
    R = np.eye(4); R[:3,0]=right; R[:3,1]=up; R[:3,2]=-fwd
    viz.viewer["/Cameras/default/rotated"].set_transform(R)


def draw_target_sphere(viz, name, pos, radius=0.05, color=0xff0000):
    viz.viewer[name].set_object(g.Sphere(radius), g.MeshLambertMaterial(color=color, opacity=0.8))
    viz.viewer[name].set_transform(tf.translation_matrix(pos))


def draw_target_axes(viz, prefix, pose, axis_len=0.15):
    px = np.array([[0, axis_len],[0,0],[0,0]])
    py = np.array([[0,0],[0,axis_len],[0,0]])
    pz = np.array([[0,0],[0,0],[0,axis_len]])
    viz.viewer[f"{prefix}/x"].set_object(g.Line(g.PointsGeometry(px), g.MeshBasicMaterial(color=0xff0000)))
    viz.viewer[f"{prefix}/y"].set_object(g.Line(g.PointsGeometry(py), g.MeshBasicMaterial(color=0x00ff00)))
    viz.viewer[f"{prefix}/z"].set_object(g.Line(g.PointsGeometry(pz), g.MeshBasicMaterial(color=0x0000ff)))
    M = pose.homogeneous
    for ax in ["x","y","z"]:
        viz.viewer[f"{prefix}/{ax}"].set_transform(M)


# -------------------------------
# Solve one frame
# -------------------------------

def solve_one_frame(model, data, q_curr, active_idxs, pos, eul, ee_name,
                    ik_attempts=20, noise_level=0.01,
                    per_joint_max_delta=0.15, total_max_delta=0.4, lambda_weight=0.1):
    prev = q_curr.copy()
    sols, errs = [], []
    for _ in range(int(ik_attempts)):
        q_init = q_curr.copy()
        noise = (np.random.rand(len(active_idxs))-0.5)*noise_level
        q_init[list(active_idxs)] = np.clip(
            q_init[list(active_idxs)] + noise,
            model.lowerPositionLimit[list(active_idxs)],
            model.upperPositionLimit[list(active_idxs)])
        ok, q_res, err = compute_ik(model, data, pos, eul, q_init, active_idxs, ee_name)
        sols.append(q_res); errs.append(err)
    q_sel, e_sel = select_best_solution_by_error(sols, errs, prev, active_idxs,
                                                 per_joint_max_delta, total_max_delta, lambda_weight)
    return q_sel, e_sel


# -------------------------------
# Main Runner (无手眼变换)
# -------------------------------

def run_dual_arm_trajectory(
    urdf_path:str,
    right_csv:str,
    left_csv:str,
    right_end_effector:str="r_gripper_base_link",
    left_end_effector:str="l_gripper_base_link",
    right_active_joint_names=("r_joint1","r_joint2","r_joint3","r_joint4","r_joint5","r_joint6","r_joint7"),
    left_active_joint_names=("l_joint1","l_joint2","l_joint3","l_joint4","l_joint5","l_joint6","l_joint7"),
    ik_attempts:int=20, noise_level:float=0.01, per_joint_max_delta:float=0.15,
    total_max_delta:float=0.4, lambda_weight:float=0.1, sleep_each:float=0.004,
    save_joint_csv:bool=True, package_dirs:Sequence[str]|None=None):

    model,data,c_model,v_model = load_robot_models(urdf_path, package_dirs, True)
    r_pos,r_eul,l_pos,l_eul = read_dualarm_csv(right_csv,left_csv)
    N = min(len(r_pos), len(l_pos))
    print(f"加载轨迹，共 {N} 帧")

    right_idxs = build_active_indices(model, right_active_joint_names)
    left_idxs  = build_active_indices(model, left_active_joint_names)
    q = pin.neutral(model)
    q[right_idxs] = [0.71,1.07,1.01,1.32,-0.07,0.32,-1.39]
    q[left_idxs]  = [-0.71,-1.07,-1.01,-1.32,0.07,-0.32,1.39]

    viz = init_meshcat(model, c_model, v_model); viz.display(q)
    look_at(viz)
    all_rows=[]; dt=1.0/60.0

    for i in range(N):
        print(f"\n--- Frame {i} ---")
        # 直接使用CSV坐标，不做R_bc变换
        r_w_p, r_w_R = r_pos[i], pin.rpy.rpyToMatrix(r_eul[i])
        l_w_p, l_w_R = l_pos[i], pin.rpy.rpyToMatrix(l_eul[i])
        r_w_eul = r_eul[i]; l_w_eul = l_eul[i]

        qr, er = solve_one_frame(model,data,q,right_idxs,r_w_p,r_w_eul,right_end_effector,
                                 ik_attempts,noise_level,per_joint_max_delta,total_max_delta,lambda_weight)
        q[right_idxs]=qr[right_idxs]

        ql, el = solve_one_frame(model,data,q,left_idxs,l_w_p,l_w_eul,left_end_effector,
                                 ik_attempts,noise_level,per_joint_max_delta,total_max_delta,lambda_weight)
        q[left_idxs]=ql[left_idxs]

        print(f"误差: 右 {er:.4f}, 左 {el:.4f}")
        viz.display(q)

        # 目标点可视化
        r_pose = pin.SE3(r_w_R, r_w_p)
        l_pose = pin.SE3(l_w_R, l_w_p)
        draw_target_sphere(viz,"right_target",r_w_p,0.05,0xff0000)
        draw_target_sphere(viz,"left_target",l_w_p,0.05,0x0000ff)
        draw_target_axes(viz,"right_target_axes",r_pose)
        draw_target_axes(viz,"left_target_axes",l_pose)

        all_rows.append([i*dt]+q.tolist())
        time.sleep(float(sleep_each))

    if save_joint_csv:
        os.makedirs("joint_data",exist_ok=True)
        cols=["timestamp"]+[f"q{j}" for j in range(model.nq)]
        pd.DataFrame(all_rows,columns=cols).to_csv("joint_data/solved_joint_trajectory2.csv",index=False)
        print("✅ Joint 数据已保存到 joint_data/solved_joint_trajectory2.csv")


if __name__ == "__main__":
    run_dual_arm_trajectory(
        urdf_path="./airexo/urdf_models/robot/true_robot.urdf",
        right_csv="./train_video/arm2_poses.csv",
        left_csv="./train_video/arm1_poses.csv",
    )