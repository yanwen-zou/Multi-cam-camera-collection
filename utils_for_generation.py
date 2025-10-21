"""
Dual‑arm IK utilities extracted from the user's script and refactored into reusable functions.

新增功能：
- 在 IK 解算过程中保存每一帧的 joint 结果到 joint_data/solved_joint_trajectory.csv
- 时间戳从 0 开始，按照 1 秒 60 帧 的固定采样率生成
"""
from __future__ import annotations
import os
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

# -------------------------------
# Model / geometry helpers
# -------------------------------

def load_robot_models(
    urdf_path: str,
    package_dirs: Sequence[str] | None = None,
    filter_visual_to_upper_body: bool = True,
    upper_body_joint_names: Sequence[str] | None = None,
) -> Tuple[pin.Model, pin.Data, pin.GeometryModel, pin.GeometryModel]:
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
            # Right arm
            "r_joint1","r_joint2","r_joint3","r_joint4","r_joint5","r_joint6","r_joint7",
            "r_hand_joint_cylinder","cylinder_to_r_gripper_joint",
            "r_Link1","r_gripper_Link11","r_gripper_Link2","r_gripper_Link22",
            # Left arm
            "l_joint1","l_joint2","l_joint3","l_joint4","l_joint5","l_joint6","l_joint7",
            "l_hand_joint_cylinder","cylinder_to_l_gripper_joint",
            "l_Link1","l_gripper_Link11","l_gripper_Link2","l_gripper_Link22",
        ]

    upper_body_joint_ids = [
        model.getJointId(nm) for nm in upper_body_joint_names if model.existJointName(nm)
    ]
    visual_model = pin.GeometryModel()
    for geom in visual_model_complete.geometryObjects:
        if geom.parentJoint in upper_body_joint_ids:
            visual_model.addGeometryObject(geom.copy())

    print(f"成功构建完整机器人模型！自由度: {model.nq}")
    print(f"上半身几何体数量: {len(visual_model.geometryObjects)}")
    return model, data, collision_model, visual_model


def build_active_indices(model: pin.Model, joint_names: Sequence[str]) -> List[int]:
    idxs: List[int] = []
    for name in joint_names:
        jid = model.getJointId(name)
        if jid >= model.njoints:
            raise ValueError(f"Joint '{name}' not found in model (jid={jid}).")
        j = model.joints[jid]
        start = j.idx_q
        for k in range(j.nq):
            idxs.append(start + k)
    return idxs


# -------------------------------
# Data utilities
# -------------------------------

def read_csv_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path)
    positions = df[["x","y","z"]].to_numpy()
    eul_deg = df[["rx","ry","rz"]].to_numpy()
    eul_rad = np.deg2rad(eul_deg)
    return positions, eul_rad


def make_camera_pose(t_base_cam: Iterable[float], rpy_deg: Iterable[float]) -> pin.SE3:
    t = np.asarray(t_base_cam, dtype=float)
    rpy = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    R = pin.rpy.rpyToMatrix(rpy)
    return pin.SE3(R, t)


# -------------------------------
# IK core
# -------------------------------

def compute_ik(
    model: pin.Model,
    data: pin.Data,
    target_position: np.ndarray,
    target_euler_angles: np.ndarray,
    q_init: np.ndarray,
    active_idxs: Sequence[int],
    end_effector_name: str,
    max_iter: int = 2000,
    eps: float = 1e-3,
    stall_threshold: int = 50,
    damp_base: float = 1e-3,
    alpha: float = 0.1,
) -> Tuple[bool, np.ndarray, float]:
    frame_id = model.getFrameId(end_effector_name)
    target_R = pin.rpy.rpyToMatrix(target_euler_angles)
    target_pose = pin.SE3(target_R, np.asarray(target_position))

    q = q_init.copy()
    prev_error_norm = float("inf")
    stall_count = 0
    best_q = q.copy()
    min_error = np.inf
    I6 = np.eye(6)

    for _ in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_pose = data.oMf[frame_id]

        err_vec = pin.log(target_pose.inverse() * current_pose).vector
        err_norm = np.linalg.norm(err_vec)

        if err_norm < min_error:
            min_error, best_q = err_norm, q.copy()
        if err_norm < eps:
            return True, q, err_norm

        if abs(err_norm - prev_error_norm) < 1e-6:
            stall_count += 1
            if stall_count > stall_threshold:
                return False, best_q, min_error
        else:
            stall_count = 0
        prev_error_norm = err_norm

        J = pin.computeFrameJacobian(model, data, q, frame_id)
        J_active = J[:, active_idxs]
        damp = min(max(damp_base, err_norm * 0.05), 5e-2)
        JJt = J_active @ J_active.T + damp * I6
        dq_active = J_active.T @ np.linalg.solve(JJt, err_vec)

        dq = np.zeros(model.nv)
        dq[list(active_idxs)] = dq_active
        q = pin.integrate(model, q, -alpha * dq)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    return False, best_q, min_error


def select_best_solution_by_error(
    solutions: Sequence[np.ndarray],
    errors: Sequence[float],
    q_previous: np.ndarray,
    active_idxs: Sequence[int],
    per_joint_max_delta: float,
    total_max_delta: float,
    lambda_weight: float,
) -> Tuple[np.ndarray, float]:
    valid, v_errs, v_deltas = [], [], []
    a = np.array(list(active_idxs))
    for q_sol, err in zip(solutions, errors):
        delta = q_sol[a] - q_previous[a]
        delta_norm = float(np.linalg.norm(delta))
        max_delta = float(np.max(np.abs(delta)))
        if max_delta <= per_joint_max_delta and delta_norm <= total_max_delta:
            valid.append(q_sol); v_errs.append(err); v_deltas.append(delta_norm)

    if valid:
        scores = [e + lambda_weight * d for e, d in zip(v_errs, v_deltas)]
        k = int(np.argmin(scores))
        return valid[k], v_errs[k]
    else:
        k = int(np.argmin(errors))
        print(f"⚠️ 没有解满足平滑度约束，回退最小误差解 (误差: {errors[k]:.4f})")
        return solutions[k], errors[k]


# -------------------------------
# Visualization helpers
# -------------------------------

def init_meshcat(model: pin.Model, collision_model: pin.GeometryModel, visual_model: pin.GeometryModel) -> MeshcatVisualizer:
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    viz.viewer.open()
    return viz


def look_at(viz: MeshcatVisualizer, camera_pos: Sequence[float], target_pos: Sequence[float], up: np.ndarray | Sequence[float] = (0, 0, 1)) -> None:
    camera_pos = np.asarray(camera_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    T = np.eye(4); T[:3, 3] = target_pos
    viz.viewer["/Cameras/default"].set_transform(T)
    offset = camera_pos - target_pos
    viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", offset.tolist())
    forward = (target_pos - camera_pos); forward /= np.linalg.norm(forward)
    up = np.asarray(up, dtype=float)
    right = np.cross(up, forward); right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    R = np.eye(4); R[:3,0]=right; R[:3,1]=true_up; R[:3,2]=-forward
    viz.viewer["/Cameras/default/rotated"].set_transform(R)


# -------------------------------
# Per‑frame solving
# -------------------------------

def solve_one_frame(
    model: pin.Model,
    data: pin.Data,
    q_current: np.ndarray,
    active_idxs: Sequence[int],
    target_world_pos: np.ndarray,
    target_world_eul: np.ndarray,
    end_effector_name: str,
    ik_attempts: int = 20,
    noise_level: float = 0.01,
    per_joint_max_delta: float = 0.15,
    total_max_delta: float = 0.4,
    lambda_weight: float = 0.1,
) -> Tuple[np.ndarray, float]:
    prev_q = q_current.copy()
    sols, errs = [], []
    for _ in range(int(ik_attempts)):
        q_init = q_current.copy()
        noise = (np.random.rand(len(active_idxs)) - 0.5) * noise_level
        q_init[list(active_idxs)] = np.clip(
            q_init[list(active_idxs)] + noise,
            model.lowerPositionLimit[list(active_idxs)],
            model.upperPositionLimit[list(active_idxs)],
        )
        ok, q_ik, err = compute_ik(model, data, target_world_pos, target_world_eul, q_init, active_idxs, end_effector_name)
        sols.append(q_ik); errs.append(err)

    q_sel, e_sel = select_best_solution_by_error(
        sols, errs, prev_q, active_idxs, per_joint_max_delta, total_max_delta, lambda_weight
    )
    return q_sel, float(e_sel)


# -------------------------------
# High‑level runner
# -------------------------------

def run_dual_arm_trajectory(
    urdf_path: str,
    right_csv: str,
    left_csv: str,
    save_joint_csv: bool = True,
    right_end_effector: str = "r_gripper_base_link",
    left_end_effector: str = "l_gripper_base_link",
    right_active_joint_names: Sequence[str] = ("r_joint1","r_joint2","r_joint3","r_joint4","r_joint5","r_joint6","r_joint7"),
    left_active_joint_names: Sequence[str] = ("l_joint1","l_joint2","l_joint3","l_joint4","l_joint5","l_joint6","l_joint7"),
    right_q_start: Sequence[float] | None = None,
    left_q_start: Sequence[float] | None = None,
    t_base_cam: Sequence[float] = (-0.03, 0.4, 0.55),
    rpy_base_cam_deg: Sequence[float] = (90.0, 0.0, 0.0),
    ik_attempts: int = 20,
    noise_level: float = 0.01,
    per_joint_max_delta: float = 0.15,
    total_max_delta: float = 0.4,
    lambda_weight: float = 0.1,
    sleep_each: float = 0.004,
    package_dirs: Sequence[str] | None = None,
    set_initial_camera: bool = True,
) -> None:
    model, data, collision_model, visual_model = load_robot_models(
        urdf_path, package_dirs, filter_visual_to_upper_body=True
    )
    right_positions, right_euls = read_csv_data(left_csv)
    left_positions, left_euls  = read_csv_data(right_csv)

    N = min(len(right_positions), len(left_positions))
    right_positions, right_euls, left_positions, left_euls = right_positions[:N], right_euls[:N], left_positions[:N], left_euls[:N]
    print(f"加载轨迹数据，总帧数: {N}")

    right_idxs = build_active_indices(model, right_active_joint_names)
    left_idxs = build_active_indices(model, left_active_joint_names)
    q = pin.neutral(model)
    if right_q_start is None: right_q_start = (0.71,1.07,1.01,1.32,-0.07,0.32,-1.39)
    if left_q_start  is None: left_q_start  = (-0.71,-1.07,-1.01,-1.32,0.07,-0.32,1.39)
    q[right_idxs] = np.asarray(right_q_start); q[left_idxs] = np.asarray(left_q_start)

    viz = init_meshcat(model, collision_model, visual_model); viz.display(q)
    if set_initial_camera:
        look_at(viz, camera_pos=[-0.0032,0.01,1.7826], target_pos=[-0.0032,-0.5903,1.1026])
    cam_T = make_camera_pose(t_base_cam, rpy_base_cam_deg)

    # === 新增：保存 joint 数据结构 ===
    all_rows = []
    dt = 1.0/60.0

    for i in range(N):
        print(f"\n--- Processing frame {i} ---")
        # === 右/左手世界 ===
        r_world_p = cam_T.act(np.asarray(right_positions[i]))
        r_world_R = cam_T.rotation @ pin.rpy.rpyToMatrix(right_euls[i]) @ pin.AngleAxis(-np.pi/2, np.array([1,0,0])).matrix()
        r_world_eul = pin.rpy.matrixToRpy(r_world_R)

        l_world_p = cam_T.act(np.asarray(left_positions[i]))
        l_local_R = pin.rpy.rpyToMatrix(left_euls[i])
        l_world_R = cam_T.rotation @ l_local_R @ pin.AngleAxis(np.pi, np.array([0,0,1])).matrix() @ pin.AngleAxis(np.pi/2, np.array([1,0,0])).matrix()
        l_world_eul = pin.rpy.matrixToRpy(l_world_R)

        q_sel_r, err_r = solve_one_frame(model,data,q,right_idxs,r_world_p,r_world_eul,right_end_effector,ik_attempts,noise_level,per_joint_max_delta,total_max_delta,lambda_weight)
        q[right_idxs] = q_sel_r[right_idxs]
        q_sel_l, err_l = solve_one_frame(model,data,q,left_idxs,l_world_p,l_world_eul,left_end_effector,ik_attempts,noise_level,per_joint_max_delta,total_max_delta,lambda_weight)
        q[left_idxs] = q_sel_l[left_idxs]

        print(f"Frame {i}: 右臂误差:{err_r:.4f}, 左臂误差:{err_l:.4f}")
        viz.display(q)

        # === 追加保存行 ===
        timestamp = i*dt
        all_rows.append([timestamp]+q.tolist())
        time.sleep(float(sleep_each))

    if save_joint_csv:
        os.makedirs("joint_data", exist_ok=True)
        colnames = ["timestamp"]+[f"q{j}" for j in range(model.nq)]
        pd.DataFrame(all_rows, columns=colnames).to_csv("joint_data/solved_joint_trajectory_for_generation.csv", index=False)
        print(f"✅ Joint 数据已保存到 joint_data/solved_joint_trajectory_for_generation.csv")


if __name__ == "__main__":
    run_dual_arm_trajectory(
        urdf_path="./airexo/urdf_models/robot/true_robot.urdf",
        right_csv="./train_video/hand_landmarks_3d_offline_right.csv",
        left_csv="./train_video/hand_landmarks_3d_offline_left.csv",
    )