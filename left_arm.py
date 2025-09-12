"""
Left‑arm only viewer & IK runner.

Usage:
    python run_left_arm_ik.py 

It loads the robot, renders only the LEFT arm in Meshcat, and solves an input
left‑arm trajectory CSV [x,y,z,rx,ry,rz] (deg for r*). Edit the paths below.
"""
from __future__ import annotations
import time
import numpy as np
import pinocchio as pin

import utils as dai

# ===================== User Config =====================
URDF_PATH = "./airexo/urdf_models/robot/true_robot.urdf"
LEFT_CSV  = "./train_video/hand_landmarks_3d_offline_left.csv"

LEFT_END_EFFECTOR = "l_gripper_base_link"
LEFT_ACTIVE_JOINT_NAMES = (
    "l_joint1","l_joint2","l_joint3","l_joint4","l_joint5","l_joint6","l_joint7"
)

# 左臂初始姿态（可按需调整）
LEFT_Q_START = (-0.71, -1.07, -1.01, -1.32, 0.07, -0.32, 1.39)

# 相机位姿：base→camera（用于把相机系的目标点转换到世界系）
T_BASE_CAM = (-0.03, 0.6, 1.15)
RPY_BASE_CAM_DEG = (90.0, 0.0, 0.0)

# IK 超参
IK_ATTEMPTS = 20
NOISE_LEVEL = 0.01
PER_JOINT_MAX_DELTA = 0.15
TOTAL_MAX_DELTA = 0.4
LAMBDA_WEIGHT = 0.1
SLEEP_EACH = 0.05
PACKAGE_DIRS = ("./airexo/urdf_models/robot",)
# =======================================================

# 仅保留左臂相关几何体以减少渲染负担并避免右臂干扰
LEFT_VISUAL_JOINTS = (
    # Left arm
    "l_joint1","l_joint2","l_joint3","l_joint4","l_joint5","l_joint6","l_joint7",
    # Left gripper (按你的URDF命名增减)
    "l_hand_joint_cylinder","cylinder_to_l_gripper_joint",
    "l_Link1","l_gripper_Link11","l_gripper_Link2","l_gripper_Link22",
)


def main() -> None:
    # 1) 加载模型（只渲染左臂几何）
    model, data, collision_model, visual_model = dai.load_robot_models(
        urdf_path=URDF_PATH,
        package_dirs=PACKAGE_DIRS,
        filter_visual_to_upper_body=True,
        upper_body_joint_names=LEFT_VISUAL_JOINTS,
    )

    # 2) 读左臂轨迹（相机系）
    left_positions, left_euls = dai.read_csv_data(LEFT_CSV)
    N = len(left_positions)
    print(f"加载左臂轨迹，共 {N} 帧")

    # 3) 关节索引与初始q
    left_idxs = dai.build_active_indices(model, LEFT_ACTIVE_JOINT_NAMES)
    print(f"左臂活跃关节索引: {left_idxs}")

    q = pin.neutral(model)
    q[left_idxs] = np.clip(
        np.asarray(LEFT_Q_START),
        model.lowerPositionLimit[left_idxs],
        model.upperPositionLimit[left_idxs],
    )

    # 4) Meshcat 初始化并设定相机朝向
    viz = dai.init_meshcat(model, collision_model, visual_model)
    viz.display(q)
    dai.look_at(
        viz,
        camera_pos=[-0.0032, 0.01, 1.7826],
        target_pos=[-0.0032, -0.5903, 1.3026],
    )

    # 5) 准备相机外参（base→camera）
    cam_T = dai.make_camera_pose(T_BASE_CAM, RPY_BASE_CAM_DEG)

    # 6) 主循环：对每一帧目标求IK，仅更新左臂
    for i in range(N):
        print(f"\n--- Frame {i} ---")
        # 目标从相机系→世界系
        l_local_p = np.asarray(left_positions[i])
        l_world_p = cam_T.act(l_local_p)
        l_local_R = pin.rpy.rpyToMatrix(left_euls[i])
        l_world_R = cam_T.rotation @ l_local_R
        l_world_eul = pin.rpy.matrixToRpy(l_world_R)

        # 只对左臂求解，并把解写回 q 的左臂自由度
        q_sel_l, err_l = dai.solve_one_frame(
            model=model,
            data=data,
            q_current=q,
            active_idxs=left_idxs,
            target_world_pos=l_world_p,
            target_world_eul=l_world_eul,
            end_effector_name=LEFT_END_EFFECTOR,
            ik_attempts=IK_ATTEMPTS,
            noise_level=NOISE_LEVEL,
            per_joint_max_delta=PER_JOINT_MAX_DELTA,
            total_max_delta=TOTAL_MAX_DELTA,
            lambda_weight=LAMBDA_WEIGHT,
        )
        q[left_idxs] = q_sel_l[left_idxs]

        print(f"误差: {err_l:.4f}\n左臂关节角: {np.round(q[left_idxs], 3)}")

        viz.display(q)

        # 可视化目标点与坐标轴
        l_pose = pin.SE3(l_world_R, l_world_p)
        dai.draw_target_sphere(viz, "left_target_sphere", l_world_p, radius=0.05, color=0x0000ff)
        dai.draw_target_axes(viz, "left_target_axes", l_pose)

        time.sleep(SLEEP_EACH)


if __name__ == "__main__":
    main()
