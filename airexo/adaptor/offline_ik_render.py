import os
import pinocchio as pin
import numpy as np
import pandas as pd
import hydra
import cv2
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from tqdm import tqdm
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from inv_work import *

# 导入所有必要的模块
from airexo.helpers.logger import setup_loggers
from airexo.helpers.transform import transform_arm

# 你的 Pinocchio IK 函数 (与之前相同，此处省略以保持代码简洁)
# ... [load_robot_model, read_csv_data, compute_ik, select_best_solution_by_error 函数] ...

@hydra.main(
    version_base=None,
    config_path="../configs/adaptor",
    config_name="ik_render"
)
def main(cfg: DictConfig):
    # 初始化日志系统并解析配置
    setup_loggers()
    OmegaConf.resolve(cfg)

    # --- 文件路径 ---
    urdf_path = to_absolute_path(cfg.urdf_path)
    csv_file_path = to_absolute_path(cfg.csv_file_path)
    render_output_dir = to_absolute_path(cfg.render_output_dir)
    
    os.makedirs(render_output_dir, exist_ok=True)
    robot_color_path = os.path.join(render_output_dir, "color")
    robot_depth_path = os.path.join(render_output_dir, "depth")
    robot_mask_path = os.path.join(render_output_dir, "mask")
    os.makedirs(robot_color_path, exist_ok=True) 
    os.makedirs(robot_depth_path, exist_ok=True) 
    os.makedirs(robot_mask_path, exist_ok=True) 

    # --- 从配置文件中直接获取相机参数 ---
    cam_translation = np.array(cfg.camera_params.cam_to_base.translation)
    cam_rpy = np.array(cfg.camera_params.cam_to_base.rpy)
    cam_to_base_matrix = pin.SE3(pin.rpy.rpyToMatrix(cam_rpy), cam_translation)
    
    intrinsic_matrix = np.array([
        [cfg.camera_params.intrinsic.fx, 0, cfg.camera_params.intrinsic.cx],
        [0, cfg.camera_params.intrinsic.fy, cfg.camera_params.intrinsic.cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # --- 初始化渲染器 ---
    robot_renderer = hydra.utils.instantiate(
        cfg.robot_renderer,
        cam_to_base=cam_to_base_matrix,
        intrinsic=intrinsic_matrix
    )
    
    # --- 模型加载 ---
    model, data, _, _ = load_robot_model(urdf_path)
    
    # --- IK 参数 ---
    per_joint_max_delta = cfg.ik.per_joint_max_delta
    total_max_delta = cfg.ik.total_max_delta
    noise_level = cfg.ik.noise_level
    ik_attempts = cfg.ik.ik_attempts
    lambda_weight = cfg.ik.lambda_weight
    
    # --- 关节定义 ---
    active_joint_names = cfg.ik.active_joint_names
    active_joint_ids = [model.getJointId(name) for name in active_joint_names]
    active_idxs = [model.joints[jid].idx_v for jid in active_joint_ids]
    
    # --- 数据加载 ---
    wrist_positions, wrist_euler_angles = read_csv_data(csv_file_path)

    # --- 初始化姿态 ---
    q_current = pin.neutral(model)
    q_start_values = np.array(cfg.ik.q_start_values)
    for joint_name, joint_value in zip(active_joint_names, q_start_values):
        joint_id = model.getJointId(joint_name)
        q_idx = model.joints[joint_id].idx_q
        q_current[q_idx] = joint_value
    
    # 获取左臂关节的中性姿态
    left_joint_neutral = pin.neutral(model)[:7] # 假设左臂是前7个关节
    
    # --- 主循环：遍历轨迹 ---
    for i, (position, euler_angles) in enumerate(tqdm(zip(wrist_positions, wrist_euler_angles), total=len(wrist_positions))):
        local_position = np.array(position)
        world_position = cam_to_base_matrix.act(local_position)
        local_rotation = pin.rpy.rpyToMatrix(euler_angles)
        world_rotation = cam_to_base_matrix.rotation @ local_rotation
        world_euler_angles = pin.rpy.matrixToRpy(world_rotation)
        
        q_previous = q_current.copy()
        solutions = []
        errors = []
        
        for _ in range(ik_attempts):
            q_init_noisy = q_current.copy()
            noise = (np.random.rand(len(active_idxs)) - 0.5) * noise_level
            for idx, n in zip(active_idxs, noise):
                q_init_noisy[idx] += n
            q_init_noisy = np.clip(q_init_noisy, model.lowerPositionLimit, model.upperPositionLimit)

            success, q_ik, err = compute_ik(model, data, world_position, world_euler_angles, q_init_noisy, active_idxs)
            solutions.append(q_ik)
            errors.append(err)
        
        q_current, error = select_best_solution_by_error(solutions, errors, q_previous, active_idxs, per_joint_max_delta, total_max_delta, lambda_weight)
    
        # --- 准备左右臂关节数据以供渲染器使用 ---
        left_joint_data = left_joint_neutral
        right_joint_data = q_current[:7] # 只取前7个关节作为右臂
        
        # --- 使用渲染器生成图像 ---
        robot_renderer.update_joints(left_joint_data, right_joint_data)
        
        color = robot_renderer.render_image()
        depth = robot_renderer.render_depth()
        mask = robot_renderer.render_mask(depth=depth)
        
        depth = np.clip(depth * 1000, 0, 65535).astype(np.uint16)

        cv2.imwrite(os.path.join(robot_color_path, f"{i}.png"), color[:, :, ::-1])
        cv2.imwrite(os.path.join(robot_depth_path, f"{i}.png"), depth)
        cv2.imwrite(os.path.join(robot_mask_path, f"{i}.png"), mask)


if __name__ == '__main__':
    # 请确保 IK 相关的辅助函数（load_robot_model, read_csv_data, compute_ik, select_best_solution_by_error）
    # 在这个文件中被定义或从其他文件正确导入。
    main()