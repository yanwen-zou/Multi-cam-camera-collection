import pinocchio as pin
import numpy as np
import pandas as pd
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
import time

def load_robot_models(urdf_path):
    """
    加载完整的URDF模型，并过滤出仅包含上半身（左右机械臂和夹爪）的几何体。
    """
    robot_dir = "/home/ryan/Documents/GitHub/AirExo-2-test/airexo/urdf_models/robot"
    
    # 加载完整的模型
    model, collision_model, visual_model_complete = pin.buildModelsFromUrdf(
        urdf_path,
        package_dirs=[robot_dir],
        geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL]
    )
    data = model.createData()

        # 定义上半身相关关节（包括左右机械臂和夹爪）
    upper_body_joints = [
        # 右臂关节
        'r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7',
        
        # 右臂夹爪相关关节
        'r_hand_joint_cylinder',           # 连接 r_link7 和 r_Link7_cylinder
        'cylinder_to_r_gripper_joint',     # 连接 r_Link7_cylinder 和 r_gripper_base_link
        'r_Link1',                         # 夹爪关节1
        'r_gripper_Link11',                # 夹爪关节11
        'r_gripper_Link2',                 # 夹爪关节2
        'r_gripper_Link22',                # 夹爪关节22
        
        # 左臂关节
        'l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7',
        
        # 左臂夹爪相关关节
        'l_hand_joint_cylinder',           # 连接 l_link7 和 l_Link7_cylinder
        'cylinder_to_l_gripper_joint',     # 连接 l_Link7_cylinder 和 l_gripper_base_link
        'l_Link1',                         # 夹爪关节1
        'l_gripper_Link11',                # 夹爪关节11
        'l_gripper_Link2',                 # 夹爪关节2
        'l_gripper_Link22',                # 夹爪关节22
    ]
    
    # 获取上半身关节的 ID
    upper_body_joint_ids = [model.getJointId(name) for name in upper_body_joints if model.existJointName(name)]
    
    # 创建新的 visual_model，仅包含上半身几何体
    visual_model = pin.GeometryModel()
    for geom in visual_model_complete.geometryObjects:
        if geom.parentJoint in upper_body_joint_ids:
            visual_model.addGeometryObject(geom.copy())
    
    # 保持完整的 collision_model（可选，如果不需要碰撞检测可以类似过滤）
    print(f"成功构建完整机器人模型！自由度: {model.nq}")
    print(f"上半身几何体数量: {len(visual_model.geometryObjects)}")
    
    return model, data, collision_model, visual_model

def read_csv_data(file_path):
    """
    读取CSV数据并转换为所需格式。
    """
    data = pd.read_csv(file_path)
    positions = data[['x', 'y', 'z']].values
    euler_angles_deg = data[['rx', 'ry', 'rz']].values
    euler_angles_rad = np.deg2rad(euler_angles_deg)
    return positions, euler_angles_rad


def compute_ik(model, data, target_position, target_euler_angles, q_init, active_idxs, end_effector_name,
             max_iter=2000, eps=0.01, stall_threshold=50, damp_base=1e-3, alpha=0.1):
    """
    使用带步长和阻尼的最小二乘法求解逆运动学 (最终优化版)。
    """
    frame_id = model.getFrameId(end_effector_name)
    if frame_id < 0 or frame_id >= len(model.frames):
        raise ValueError(f"End effector frame '{end_effector_name}' not found in the model.")

    target_rotation = pin.rpy.rpyToMatrix(target_euler_angles)
    target_pose = pin.SE3(target_rotation, target_position)

    q = q_init.copy()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    current_pose = data.oMf[frame_id]

    prev_error_norm = float('inf')
    stall_count = 0
    best_q = q.copy()
    min_error = np.linalg.norm(pin.log(target_pose.inverse() * current_pose).vector)

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_pose = data.oMf[frame_id]

        error_vec = pin.log(target_pose.inverse() * current_pose).vector
        error_norm = np.linalg.norm(error_vec)

        if error_norm < min_error:
            min_error = error_norm
            best_q = q.copy()

        if error_norm < eps:
            return True, q, error_norm

        if abs(error_norm - prev_error_norm) < 1e-6:
            stall_count += 1
            if stall_count > stall_threshold:
                return False, best_q, min_error
        else:
            stall_count = 0

        prev_error_norm = error_norm

        J = pin.computeFrameJacobian(model, data, q, frame_id)
        J_active = J[:, active_idxs]

        damp = min(max(damp_base, error_norm * 0.05), 0.05)
        JJt = J_active @ J_active.T + damp * np.eye(6)
        
        dq_active = J_active.T @ np.linalg.solve(JJt, error_vec)
        dq = np.zeros(model.nv)
        dq[active_idxs] = dq_active

        q = pin.integrate(model, q, -alpha * dq)
        
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    return False, best_q, min_error


def select_best_solution_by_error(solutions, errors, q_previous, active_idxs, per_joint_max_delta, total_max_delta, lambda_weight):
    """
    从多个IK解中选择一个既精确又平滑的解。
    """
    valid_solutions = []
    valid_errors = []
    valid_delta_norms = []
    
    for q_ik, err in zip(solutions, errors):
        delta_q = q_ik[active_idxs] - q_previous[active_idxs]
        delta_q_norm = np.linalg.norm(delta_q)
        max_delta_per_joint = np.max(np.abs(delta_q))
        
        if max_delta_per_joint <= per_joint_max_delta and delta_q_norm <= total_max_delta:
            valid_solutions.append(q_ik)
            valid_errors.append(err)
            valid_delta_norms.append(delta_q_norm)
    
    if valid_solutions:
        scores = [err + lambda_weight * delta for err, delta in zip(valid_errors, valid_delta_norms)]
        min_score_idx = np.argmin(scores)
        return valid_solutions[min_score_idx], valid_errors[min_score_idx]
    else:
        min_error_idx = np.argmin(errors)
        print(f"⚠️ Frame 警告: 没有解满足平滑度约束，回退至最小误差解 (误差: {errors[min_error_idx]:.4f})")
        return solutions[min_error_idx], errors[min_error_idx]


def look_at(viz, camera_pos, target_pos, up=np.array([0, 0, 1])):
    camera_pos = np.array(camera_pos)
    target_pos = np.array(target_pos)
    
    # 设置焦点
    T = np.eye(4)
    T[:3, 3] = target_pos
    viz.viewer["/Cameras/default"].set_transform(T)

    # 相机相对焦点的偏移
    offset = camera_pos - target_pos
    viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", offset.tolist())

    # ---- 关键：修正旋转，使相机真的看向目标 ----
    # forward = -Z 方向
    forward = (target_pos - camera_pos)
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)

    R = np.eye(4)
    R[:3, 0] = right
    R[:3, 1] = true_up
    R[:3, 2] = -forward   # 注意这里是 +Z/-Z，可能需要 flip

    viz.viewer["/Cameras/default/rotated"].set_transform(R)

def main():
    # --- 文件路径 ---
    urdf_path = "/home/ryan/Documents/GitHub/AirExo-2-test/airexo/urdf_models/robot/true_robot.urdf"
    right_csv_file_path = "/home/ryan/Documents/GitHub/AirExo-2-test/train_video/hand_landmarks_3d_offline_right.csv"
    left_csv_file_path = "/home/ryan/Documents/GitHub/AirExo-2-test/train_video/hand_landmarks_3d_offline_left.csv"

    # --- 模型加载（使用完整模型） ---
    model, data, collision_model, visual_model = load_robot_models(urdf_path)
    
    # --- 核心参数调整区 ---
    per_joint_max_delta = 0.15      
    total_max_delta = 0.4         
    noise_level = 0.2             
    ik_attempts = 20              
    lambda_weight = 0.1           
    
    # --- 关节定义（完整模型中的左右臂关节索引） ---
    right_active_joints = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']
    left_active_joints = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7']
    
    right_active_idxs = []
    for name in right_active_joints:
        jid = model.getJointId(name)
        if jid < model.njoints:
            idx_q = model.joints[jid].idx_q
            for i in range(model.joints[jid].nq):
                right_active_idxs.append(idx_q + i)
        else:
            raise ValueError(f"Joint '{name}' not found in model.")
    
    left_active_idxs = []
    for name in left_active_joints:
        jid = model.getJointId(name)
        if jid < model.njoints:
            idx_q = model.joints[jid].idx_q
            for i in range(model.joints[jid].nq):
                left_active_idxs.append(idx_q + i)
        else:
            raise ValueError(f"Joint '{name}' not found in model.")
    
    print(f"右臂活跃关节索引: {right_active_idxs}")
    print(f"左臂活跃关节索引: {left_active_idxs}")

    # --- 坐标系变换定义 ---
    t_base_cam = np.array([-0.03, 0.6, 1.15])
    rpy = np.array([np.deg2rad(90), 0.0,0])
    R_base_cam = pin.rpy.rpyToMatrix(rpy)
    camera_pose = pin.SE3(R_base_cam, t_base_cam)

    # --- 数据加载 ---
    right_wrist_positions, right_wrist_euler_angles = read_csv_data(left_csv_file_path)
    left_wrist_positions, left_wrist_euler_angles = read_csv_data(right_csv_file_path)
    
    # 确保两个CSV文件有相同的帧数
    min_frames = min(len(right_wrist_positions), len(left_wrist_positions))
    right_wrist_positions = right_wrist_positions[:min_frames]
    right_wrist_euler_angles = right_wrist_euler_angles[:min_frames]
    left_wrist_positions = left_wrist_positions[:min_frames]
    left_wrist_euler_angles = left_wrist_euler_angles[:min_frames]
    print(f"加载轨迹数据，总帧数: {min_frames}")

    # --- 可视化初始化（使用完整模型）---
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    viz.viewer.open()
    
    # --- 末端执行器名称 ---
    right_end_effector = "r_gripper_base_link"
    left_end_effector = "l_gripper_base_link"
    
    # --- 初始姿态定义 ---
    right_q_start_values = np.array([0.71, 1.07, 1.01, 1.32, -0.07, 0.32, -1.39])
    left_q_start_values = np.array([-0.71, -1.07, -1.01, -1.32, 0.07, -0.32, 1.39]) # 左臂镜像初始姿态
    
    q_current = pin.neutral(model)  # 完整模型的初始 q
    q_current[right_active_idxs] = np.clip(right_q_start_values, model.lowerPositionLimit[right_active_idxs], model.upperPositionLimit[right_active_idxs])
    q_current[left_active_idxs] = np.clip(left_q_start_values, model.lowerPositionLimit[left_active_idxs], model.upperPositionLimit[left_active_idxs])
    
    print(f"右臂初始关节角度: {q_current[right_active_idxs]}")
    print(f"左臂初始关节角度: {q_current[left_active_idxs]}")
    
    # 显示初始姿态
    viz.display(q_current)
    
    # 设置相机视角
    look_at(viz, camera_pos=[-0.0032, 0.01, 1.7826], target_pos=[-0.0032, -0.5903, 1.3026])
    
    time.sleep(2)

    # --- 显示初始末端小球 ---
    # 更新运动学
    pin.forwardKinematics(model, data, q_current)
    pin.updateFramePlacements(model, data)
    
    # # 右臂末端
    # right_ee_frame = model.getFrameId(right_end_effector)
    # right_ee_pos = data.oMf[right_ee_frame].translation
    # viz.viewer["right_ee_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0xff0000, opacity=0.8))
    # viz.viewer["right_ee_sphere"].set_transform(tf.translation_matrix(right_ee_pos))

    # # 左臂末端
    # left_ee_frame = model.getFrameId(left_end_effector)
    # left_ee_pos = data.oMf[left_ee_frame].translation
    # viz.viewer["left_ee_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0x0000ff, opacity=0.8))
    # viz.viewer["left_ee_sphere"].set_transform(tf.translation_matrix(left_ee_pos))

    # # --- 初始显示后暂停3秒 ---
    time.sleep(3)

    # --- 主循环：遍历轨迹 ---
    for i in range(min_frames):
        print(f"\n--- Processing frame {i} ---")

        # 右臂目标位置和姿态变换
        right_local_position = np.array(right_wrist_positions[i])
        right_world_position = camera_pose.act(right_local_position)
        right_local_rotation = pin.rpy.rpyToMatrix(right_wrist_euler_angles[i])
        right_world_rotation = camera_pose.rotation @ right_local_rotation
        right_world_euler_angles = pin.rpy.matrixToRpy(right_world_rotation)
        
        # 左臂目标位置和姿态变换
        left_local_position = np.array(left_wrist_positions[i])
        left_world_position = camera_pose.act(left_local_position)
        left_local_rotation = pin.rpy.rpyToMatrix(left_wrist_euler_angles[i])
        left_world_rotation = camera_pose.rotation @ left_local_rotation
        left_world_euler_angles = pin.rpy.matrixToRpy(left_world_rotation)
        
        # 右臂IK求解（仅更新右臂关节）
        print("右臂求解...")
        right_q_previous = q_current.copy()
        right_solutions = []
        right_errors = []
        
        for _ in range(ik_attempts):
            q_init_noisy = q_current.copy()
            q_init_noisy[right_active_idxs] += (np.random.rand(len(right_active_idxs)) - 0.5) * noise_level
            q_init_noisy[right_active_idxs] = np.clip(q_init_noisy[right_active_idxs], model.lowerPositionLimit[right_active_idxs], model.upperPositionLimit[right_active_idxs])
            success, q_ik, err = compute_ik(model, data, right_world_position, right_world_euler_angles, 
                                          q_init_noisy, right_active_idxs, right_end_effector)
            right_solutions.append(q_ik)
            right_errors.append(err)
        
        right_q_selected, right_error = select_best_solution_by_error(right_solutions, right_errors, right_q_previous, 
                                                                   right_active_idxs, per_joint_max_delta, total_max_delta, lambda_weight)
        q_current[right_active_idxs] = right_q_selected[right_active_idxs]  # 只更新右臂关节
        
        # 左臂IK求解（仅更新左臂关节）
        print("左臂求解...")
        left_q_previous = q_current.copy()
        left_solutions = []
        left_errors = []
        
        for _ in range(ik_attempts):
            q_init_noisy = q_current.copy()
            q_init_noisy[left_active_idxs] += (np.random.rand(len(left_active_idxs)) - 0.5) * noise_level
            q_init_noisy[left_active_idxs] = np.clip(q_init_noisy[left_active_idxs], model.lowerPositionLimit[left_active_idxs], model.upperPositionLimit[left_active_idxs])
            success, q_ik, err = compute_ik(model, data, left_world_position, left_world_euler_angles,
                                          q_init_noisy, left_active_idxs, left_end_effector)
            left_solutions.append(q_ik)
            left_errors.append(err)
        
        left_q_selected, left_error = select_best_solution_by_error(left_solutions, left_errors, left_q_previous,
                                                                 left_active_idxs, per_joint_max_delta, total_max_delta, lambda_weight)
        q_current[left_active_idxs] = left_q_selected[left_active_idxs]  # 只更新左臂关节
    
        print(f"Frame {i}: 右臂误差: {right_error:.4f}, 左臂误差: {left_error:.4f}")
        print(f"右臂关节角: {np.round(q_current[right_active_idxs], 2)}")
        print(f"左臂关节角: {np.round(q_current[left_active_idxs], 2)}")
        
        # 显示完整机器人
        viz.display(q_current)
        
        # 更新相机视角到当前的camera_link位置（如果需要）
        # set_camera_to_camera_link(viz, model, data, q_current)
        
        # 显示右臂目标
        right_target_pose = pin.SE3(right_world_rotation, right_world_position)
        viz.viewer["right_target_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0xff0000, opacity=0.8))
        viz.viewer["right_target_sphere"].set_transform(tf.translation_matrix(right_world_position))
        
        # 显示左臂目标  
        left_target_pose = pin.SE3(left_world_rotation, left_world_position)
        viz.viewer["left_target_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0x0000ff, opacity=0.8))
        viz.viewer["left_target_sphere"].set_transform(tf.translation_matrix(left_world_position))
        
        # 显示坐标轴（右臂）
        viz.viewer["right_target_axes/x"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0.15], [0, 0], [0, 0]])), g.MeshBasicMaterial(color=0xff0000)))
        viz.viewer["right_target_axes/y"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0.15], [0, 0]])), g.MeshBasicMaterial(color=0x00ff00))) 
        viz.viewer["right_target_axes/z"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0], [0, 0.15]])), g.MeshBasicMaterial(color=0x0000ff)))
        viz.viewer["right_target_axes/x"].set_transform(right_target_pose.homogeneous)
        viz.viewer["right_target_axes/y"].set_transform(right_target_pose.homogeneous)
        viz.viewer["right_target_axes/z"].set_transform(right_target_pose.homogeneous)
        
        # 显示坐标轴（左臂）
        viz.viewer["left_target_axes/x"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0.15], [0, 0], [0, 0]])), g.MeshBasicMaterial(color=0xff4444)))
        viz.viewer["left_target_axes/y"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0.15], [0, 0]])), g.MeshBasicMaterial(color=0x44ff44))) 
        viz.viewer["left_target_axes/z"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0], [0, 0.15]])), g.MeshBasicMaterial(color=0x4444ff)))
        viz.viewer["left_target_axes/x"].set_transform(left_target_pose.homogeneous)
        viz.viewer["left_target_axes/y"].set_transform(left_target_pose.homogeneous)
        viz.viewer["left_target_axes/z"].set_transform(left_target_pose.homogeneous)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()