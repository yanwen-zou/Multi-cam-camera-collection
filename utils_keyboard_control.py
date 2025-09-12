"""
Dual‑arm IK utilities extracted from the user's script and refactored into reusable functions.

Main entry points:
- load_robot_models(urdf_path, package_dirs) -> (model, data, collision_model, visual_model)
- build_active_indices(model, joint_names) -> List[int]
- compute_ik(...) -> (success: bool, q: np.ndarray, error: float)
- select_best_solution_by_error(...)
- look_at(viz, camera_pos, target_pos, up=(0,0,1))
- init_meshcat(model, collision_model, visual_model) -> viz
- make_camera_pose(t_base_cam, rpy_deg) -> pin.SE3
- read_csv_data(file_path) -> (positions[N,3], euler_angles_rad[N,3])
- solve_one_frame(...)
- run_dual_arm_trajectory(...)  # convenience runner for left/right trajectories

All functions are pure utilities so you can import them from other scripts.
"""
from __future__ import annotations
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
from pynput import keyboard

# === 全局键盘状态 ===
key_state = {}

def on_press(key):
    try:
        k = key.char
        key_state[k] = True
    except AttributeError:
        pass

def on_release(key):
    try:
        k = key.char
        key_state[k] = False
    except AttributeError:
        pass
# -------------------------------
# Model / geometry helpers
# -------------------------------

def load_robot_models(
    urdf_path: str,
    package_dirs: Sequence[str] | None = None,
    filter_visual_to_upper_body: bool = True,
    upper_body_joint_names: Sequence[str] | None = None,
) -> Tuple[pin.Model, pin.Data, pin.GeometryModel, pin.GeometryModel]:
    """Load URDF and (optionally) filter visual model to upper body.

    Args:
        urdf_path: Path to URDF.
        package_dirs: Additional package search paths.
        filter_visual_to_upper_body: If True, keep only the geometry attached
            to the joints listed in `upper_body_joint_names`.
        upper_body_joint_names: Joint names considered "upper body". If None,
            uses the default set from the original script.

    Returns:
        model, data, collision_model, visual_model
    """
    if package_dirs is None:
        package_dirs = ["./airexo/urdf_models/robot"]

    # Build full models
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
            # Right gripper
            "r_hand_joint_cylinder","cylinder_to_r_gripper_joint",
            "r_Link1","r_gripper_Link11","r_gripper_Link2","r_gripper_Link22",
            # Left arm
            "l_joint1","l_joint2","l_joint3","l_joint4","l_joint5","l_joint6","l_joint7",
            # Left gripper
            "l_hand_joint_cylinder","cylinder_to_l_gripper_joint",
            "l_Link1","l_gripper_Link11","l_gripper_Link2","l_gripper_Link22",
        ]

    # Get joint IDs present in model
    upper_body_joint_ids = [
        model.getJointId(nm) for nm in upper_body_joint_names if model.existJointName(nm)
    ]

    # Filter visual geometry to those joints
    visual_model = pin.GeometryModel()
    for geom in visual_model_complete.geometryObjects:
        if geom.parentJoint in upper_body_joint_ids:
            visual_model.addGeometryObject(geom.copy())

    print(f"成功构建完整机器人模型！自由度: {model.nq}")
    print(f"上半身几何体数量: {len(visual_model.geometryObjects)}")
    return model, data, collision_model, visual_model


def build_active_indices(model: pin.Model, joint_names: Sequence[str]) -> List[int]:
    """Return configuration indices for the given joint names.

    Works for 1‑DoF revolute joints and also for multi‑DoF joints (uses nq).
    """
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
    """Read trajectory CSV with columns [x,y,z, rx,ry,rz] (degrees for r*).

    Returns:
        positions: (N,3)
        euler_angles_rad: (N,3) radians (Pinocchio RPY order)
    """
    df = pd.read_csv(file_path)
    positions = df[["x","y","z"]].to_numpy()
    eul_deg = df[["rx","ry","rz"]].to_numpy()
    eul_rad = np.deg2rad(eul_deg)
    return positions, eul_rad


def make_camera_pose(t_base_cam: Iterable[float], rpy_deg: Iterable[float]) -> pin.SE3:
    """Create a base->camera SE3 from translation and RPY(deg)."""
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
    eps: float = 1e-2,
    stall_threshold: int = 50,
    damp_base: float = 1e-3,
    alpha: float = 0.1,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
) -> Tuple[bool, np.ndarray, float]:
    """Damped least‑squares IK with step size and early‑stop.

    Returns:
        success, q_best, error_best
    """
    frame_id = model.getFrameId(end_effector_name)
    if frame_id < 0 or frame_id >= len(model.frames):
        raise ValueError(f"End effector frame '{end_effector_name}' not found in the model.")

    target_R = pin.rpy.rpyToMatrix(target_euler_angles)
    target_pose = pin.SE3(target_R, np.asarray(target_position))

    q = q_init.copy()

        # ---- 若提供了自定义上下界，先把初值 q_init 在 active joints 上夹住 ----
    if joint_lower is not None or joint_upper is not None:
        qa = q[list(active_idxs)].copy()
        if joint_lower is not None:
            qa = np.maximum(qa, np.asarray(joint_lower, dtype=float))
        if joint_upper is not None:
            qa = np.minimum(qa, np.asarray(joint_upper, dtype=float))
        q[list(active_idxs)] = qa

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    current_pose = data.oMf[frame_id]

    prev_error_norm = float("inf")
    stall_count = 0
    best_q = q.copy()
    min_error = np.linalg.norm(pin.log(target_pose.inverse() * current_pose).vector)

    I6 = np.eye(6)

    for _ in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_pose = data.oMf[frame_id]

        err_vec = pin.log(target_pose.inverse() * current_pose).vector
        err_norm = float(np.linalg.norm(err_vec))

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

        # Adaptive damping
        damp = min(max(damp_base, err_norm * 0.05), 5e-2)
        JJt = J_active @ J_active.T + damp * I6
        dq_active = J_active.T @ np.linalg.solve(JJt, err_vec)

        dq = np.zeros(model.nv)
        dq[list(active_idxs)] = dq_active
        q = pin.integrate(model, q, -alpha * dq)

        # ---- 自定义关节上下界（仅对 active joints 生效；若未提供则退回model全局界）----
        if joint_lower is not None or joint_upper is not None:
            qa = q[list(active_idxs)].copy()
            if joint_lower is not None:
                qa = np.maximum(qa, np.asarray(joint_lower, dtype=float))
            if joint_upper is not None:
                qa = np.minimum(qa, np.asarray(joint_upper, dtype=float))
            q[list(active_idxs)] = qa

        # 最后仍确保不会越过模型的硬性界（双保险）
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
    """Pick a solution that balances low error and motion smoothness."""
    valid, v_errs, v_deltas = [], [], []
    a = np.array(list(active_idxs))
    for q_sol, err in zip(solutions, errors):
        delta = q_sol[a] - q_previous[a]
        delta_norm = float(np.linalg.norm(delta))
        max_delta = float(np.max(np.abs(delta)))
        if max_delta <= per_joint_max_delta and delta_norm <= total_max_delta:
            valid.append(q_sol)
            v_errs.append(err)
            v_deltas.append(delta_norm)

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

    # Focus point
    T = np.eye(4)
    T[:3, 3] = target_pos
    viz.viewer["/Cameras/default"].set_transform(T)

    # Camera offset relative to focus
    offset = camera_pos - target_pos
    viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", offset.tolist())

    forward = (target_pos - camera_pos)
    forward /= np.linalg.norm(forward)
    up = np.asarray(up, dtype=float)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)

    R = np.eye(4)
    R[:3, 0] = right
    R[:3, 1] = true_up
    R[:3, 2] = -forward
    viz.viewer["/Cameras/default/rotated"].set_transform(R)


def draw_target_axes(viz: MeshcatVisualizer, name_prefix: str, pose: pin.SE3, axis_len: float = 0.15) -> None:
    """Draw XYZ axes at a given SE3 pose."""
    px = np.array([[0, axis_len], [0, 0], [0, 0]])
    py = np.array([[0, 0], [0, axis_len], [0, 0]])
    pz = np.array([[0, 0], [0, 0], [0, axis_len]])

    viz.viewer[f"{name_prefix}/x"].set_object(
        g.Line(g.PointsGeometry(px), g.MeshBasicMaterial(color=0xff0000))
    )
    viz.viewer[f"{name_prefix}/y"].set_object(
        g.Line(g.PointsGeometry(py), g.MeshBasicMaterial(color=0x00ff00))
    )
    viz.viewer[f"{name_prefix}/z"].set_object(
        g.Line(g.PointsGeometry(pz), g.MeshBasicMaterial(color=0x0000ff))
    )
    viz.viewer[f"{name_prefix}/x"].set_transform(pose.homogeneous)
    viz.viewer[f"{name_prefix}/y"].set_transform(pose.homogeneous)
    viz.viewer[f"{name_prefix}/z"].set_transform(pose.homogeneous)


def draw_target_sphere(viz: MeshcatVisualizer, name: str, position: np.ndarray, radius: float = 0.05, color: int = 0xff0000) -> None:
    viz.viewer[name].set_object(g.Sphere(radius), g.MeshLambertMaterial(color=color, opacity=0.8))
    viz.viewer[name].set_transform(tf.translation_matrix(position))


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
    joint_lower_active: np.ndarray | None = None,
    joint_upper_active: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    """Multiple noisy restarts, pick smoothest good solution."""
    prev_q = q_current.copy()
    sols, errs = [], []
    for _ in range(int(ik_attempts)):
        q_init = q_current.copy()
        # Add small noise only on active joints
        noise = (np.random.rand(len(active_idxs)) - 0.5) * noise_level
        q_init[list(active_idxs)] = np.clip(
            q_init[list(active_idxs)] + noise,
            model.lowerPositionLimit[list(active_idxs)],
            model.upperPositionLimit[list(active_idxs)],
        )
        ok, q_ik, err = compute_ik(
            model, data, target_world_pos, target_world_eul, q_init,
            active_idxs, end_effector_name,
            joint_lower=joint_lower_active,
            joint_upper=joint_upper_active,
        )
        sols.append(q_ik)
        errs.append(err)

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
    right_end_effector: str = "r_gripper_base_link",
    left_end_effector: str = "l_gripper_base_link",
    right_active_joint_names: Sequence[str] = ("r_joint1","r_joint2","r_joint3","r_joint4","r_joint5","r_joint6","r_joint7"),
    left_active_joint_names: Sequence[str] = ("l_joint1","l_joint2","l_joint3","l_joint4","l_joint5","l_joint6","l_joint7"),
    right_q_start: Sequence[float] | None = None,
    left_q_start: Sequence[float] | None = None,
    t_base_cam: Sequence[float] = (-0.03, 0.6, 1.15),
    rpy_base_cam_deg: Sequence[float] = (90.0, 0.0, 0.0),
    ik_attempts: int = 20,
    noise_level: float = 0.01,
    per_joint_max_delta: float = 0.15,
    total_max_delta: float = 0.4,
    lambda_weight: float = 0.1,
    sleep_each: float = 0.1,
    package_dirs: Sequence[str] | None = None,
    set_initial_camera: bool = True,
) -> None:
    """Convenience function that replicates the original main loop.

    Designed so you can quickly plug it into a new script; for finer control,
    use the lower‑level utilities above.
    """
    model, data, collision_model, visual_model = load_robot_models(
        urdf_path, package_dirs, filter_visual_to_upper_body=True
    )

    right_positions, right_euls = read_csv_data(left_csv)
    left_positions, left_euls = read_csv_data(right_csv)

    N = min(len(right_positions), len(left_positions))
    right_positions = right_positions[:N]
    right_euls = right_euls[:N]
    left_positions = left_positions[:N]
    left_euls = left_euls[:N]
    print(f"加载轨迹数据，总帧数: {N}")

    right_idxs = build_active_indices(model, right_active_joint_names)
    left_idxs = build_active_indices(model, left_active_joint_names)
    print(f"右臂活跃关节索引: {right_idxs}")
    print(f"左臂活跃关节索引: {left_idxs}")

    q = pin.neutral(model)

    if right_q_start is None:
        right_q_start = (0.71, 1.07, 1.01, 1.32, -0.07, 0.32, -1.39)
    if left_q_start is None:
        left_q_start = (-0.71, -1.07, -1.01, -1.32, 0.07, -0.32, 1.39)
    print(f"右臂初始关节角: {right_q_start, 2}")
    print(f"左臂初始关节角: {left_q_start, 2}")
    q[right_idxs] = np.clip(np.asarray(right_q_start), model.lowerPositionLimit[right_idxs], model.upperPositionLimit[right_idxs])
    q[left_idxs]  = np.clip(np.asarray(left_q_start),  model.lowerPositionLimit[left_idxs],  model.upperPositionLimit[left_idxs])

    viz = init_meshcat(model, collision_model, visual_model)
    viz.display(q)

    if set_initial_camera:
        look_at(
            viz,
            camera_pos=[-0.0032, 0.01, 1.7826],
            target_pos=[-0.0032, -0.5903, 1.3026],
        )

    cam_T = make_camera_pose(t_base_cam, rpy_base_cam_deg)

    # Precompute frame IDs
    r_ff = right_end_effector
    l_ff = left_end_effector

    # 第0帧关节约束数组（按关节顺序）
    # 前三项给真实约束值，后面填大数（如1e3）表示不限制
    first_frame_constrain = np.array([0.3, 0.7, 1e2, 1e3, 1e3, 1e3, 1e3])
    eps_first = 1e0

    for i in range(N):
        print(f"\n--- Processing frame {i} ---")

        # === 世界系目标的计算逻辑（保持不变）===
        r_local_p = np.asarray(right_positions[i])
        r_world_p = cam_T.act(r_local_p)
        r_local_R = pin.rpy.rpyToMatrix(right_euls[i])
        r_world_R = cam_T.rotation @ r_local_R
        r_world_eul = pin.rpy.matrixToRpy(r_world_R)

        l_local_p = np.asarray(left_positions[i])
        l_world_p = cam_T.act(l_local_p)
        l_local_R = pin.rpy.rpyToMatrix(left_euls[i])
        l_world_R = cam_T.rotation @ l_local_R
        l_world_eul = pin.rpy.matrixToRpy(l_world_R)

        # === 键盘控制左臂 ===
        for j, idx in enumerate(left_idxs[:6]):
            pos_key = str(j+1)        # "1"..."6"
            neg_key = "qwerty"[j]     # "q","w","e","r","t","y"

            if key_state.get(pos_key, False):
                q[idx] += step
            if key_state.get(neg_key, False):
                q[idx] -= step

        # === 可视化 ===
        viz.display(q)

        r_pose = pin.SE3(r_world_R, r_world_p)
        l_pose = pin.SE3(l_world_R, l_world_p)
        draw_target_sphere(viz, "right_target_sphere", r_world_p, radius=0.05, color=0xff0000)
        draw_target_sphere(viz, "left_target_sphere",  l_world_p, radius=0.05, color=0x0000ff)
        draw_target_axes(viz, "right_target_axes", r_pose)
        draw_target_axes(viz, "left_target_axes",  l_pose)

        print("左臂角度:", np.round(q[left_idxs[:6]], 3))

        time.sleep(float(sleep_each))



# When used as a script, provide a minimal runnable example using defaults
if __name__ == "__main__":

    
    # 启动监听线程
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 每次按键改变的角度增量
    step = 0.02  # rad

    run_dual_arm_trajectory(
        urdf_path="./airexo/urdf_models/robot/true_robot.urdf",
        right_csv="./train_video/hand_landmarks_3d_offline_right.csv",
        left_csv="./train_video/hand_landmarks_3d_offline_left.csv",
    )
