import pinocchio as pin
import numpy as np
import pandas as pd
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
import time

def load_robot_model(urdf_path):
    """
    加载完整的URDF模型。
    """
    robot_dir = "./airexo/urdf_models/robot"

    
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path,
        package_dirs=[robot_dir],
        geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL]
    )
    
    data = model.createData()

    print(f"成功加载完整的运动学和几何模型！运动学自由度 (nq): {model.nq}")
    
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


def compute_ik(model, data, target_position, target_euler_angles, q_init, active_idxs, 
             end_effector_name="r_gripper_base_link",
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
    R[:3, 2] = forward   # 注意这里是 +Z/-Z，可能需要 flip


    viz.viewer["/Cameras/default/rotated"].set_transform(R)

def main():
    # --- 文件路径 ---
    urdf_path = "./airexo/urdf_models/robot/true_robot.urdf"
    csv_file_path_right = "./train_video/hand_landmarks_3d_offline_left.csv"
    csv_file_path_left = "./train_video/hand_landmarks_3d_offline_right.csv"

    # --- 模型加载 ---
    model, data, collision_model, visual_model = load_robot_model(urdf_path)
    
    # --- 核心参数调整区 ---
    per_joint_max_delta = 0.15      
    total_max_delta = 0.4         
    noise_level = 0.2             
    ik_attempts = 20              
    lambda_weight = 0.1           
    
    # --- 关节定义 ---
    active_joints_right = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']
    active_idxs_right = []
    for name in active_joints_right:
        jid = model.getJointId(name)
        if jid < model.njoints:
            idx_q = model.joints[jid].idx_q
            for i in range(model.joints[jid].nq):
                active_idxs_right.append(idx_q + i)
        else:
            raise ValueError(f"Joint '{name}' not found in model.")

    active_joints_left = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7']
    active_idxs_left = []
    for name in active_joints_left:
        jid = model.getJointId(name)
        if jid < model.njoints:
            idx_q = model.joints[jid].idx_q
            for i in range(model.joints[jid].nq):
                active_idxs_left.append(idx_q + i)
        else:
            raise ValueError(f"Joint '{name}' not found in model.")

    print(f"右臂活跃关节索引: {active_idxs_right}")
    print(f"右臂活跃关节数量: {len(active_idxs_right)} (应为7)")
    print(f"左臂活跃关节索引: {active_idxs_left}")
    print(f"左臂活跃关节数量: {len(active_idxs_left)} (应为7)")

    # --- 坐标系变换定义 ---
    t_base_cam = np.array([-0.03, 0.48, 1.15])
    rpy = np.array([np.deg2rad(90), 0.0, 0])
    R_base_cam = pin.rpy.rpyToMatrix(rpy)
    camera_pose = pin.SE3(R_base_cam, t_base_cam)

    # --- 数据加载 ---
    wrist_positions_right, wrist_euler_angles_right = read_csv_data(csv_file_path_right)
    wrist_positions_left, wrist_euler_angles_left = read_csv_data(csv_file_path_left)

    # --- 可视化初始化 ---
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    # print(f"\n可视化窗口已启动，请在浏览器中打开: {viz.url}")
    viz.viewer.open()
    
    # --- 设置相机视角为camera_link ---
    def set_camera_to_camera_link(viz, model, data, q):
        """
        将Meshcat相机设置到camera_link的位置和方向
        """
        try:
            # 更新运动学
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            
            # 查找camera_link的frame ID
            camera_frame_id = None
            for i, frame in enumerate(model.frames):
                if "camera" in frame.name.lower():
                    camera_frame_id = i
                    print(f"找到相机链接: {frame.name}")
                    break
            
            if camera_frame_id is None:
                print("警告：未找到camera_link，使用默认相机视角")
                return
            
            # 获取camera_link的世界坐标系位姿
            camera_pose_world = data.oMf[camera_frame_id]
            
            # 将SE3转换为4x4变换矩阵
            camera_transform = camera_pose_world.homogeneous.copy()
            
            # 调整相机坐标系以匹配Meshcat的相机方向
            camera_adjustment = np.array([
                [1,  0,  0, 0],
                [0, 1,  0, 0], 
                [0,  0, 1, 0],
                [0,  0,  0, 1]
            ])
            
            final_camera_transform = camera_transform @ camera_adjustment
            
            # 设置Meshcat相机变换
            viz.viewer["/Cameras/default"].set_transform(final_camera_transform)
            print("✅ 相机视角已设置为camera_link视角")
            
        except Exception as e:
            print(f"设置相机视角时出错: {e}")
            print("继续使用默认相机视角")
    
    # --- 初始姿态定义 ---
    q_start_values_right = np.array([1.0,0.7, 0.5, 1.5, 0.0, 0.0, 0.0])
    q_start_values_left = np.array([-1.0,-0.7, -0.5, -1.5, 0.0, 0.0, 0.0])  
    q_start = pin.neutral(model)
    q_start[active_idxs_right] = np.clip(q_start_values_right, model.lowerPositionLimit[active_idxs_right], model.upperPositionLimit[active_idxs_right])
    q_start[active_idxs_left] = np.clip(q_start_values_left, model.lowerPositionLimit[active_idxs_left], model.upperPositionLimit[active_idxs_left])
    q_current = q_start.copy()
    print(f"右臂自定义初始关节角度: {q_start[active_idxs_right]}")
    print(f"左臂自定义初始关节角度: {q_start[active_idxs_left]}")

    
    viz.display(q_current)
    
# 设置相机视角到camera_link
    look_at(viz,camera_pos=[-0.0032, 0.01, 1.9526],target_pos=[-0.0032, -0.5903, 1.3026])
    
    time.sleep(2)

    # --- 初始显示目标 ---
    if len(wrist_positions_right) > 0 and len(wrist_positions_left) > 0:
        first_position_right = wrist_positions_right[0]
        first_euler_angles_right = wrist_euler_angles_right[0]
        
        first_local_position_right = np.array(first_position_right)
        first_world_position_right = camera_pose.act(first_local_position_right)
        
        first_local_rotation_right = pin.rpy.rpyToMatrix(first_euler_angles_right)
        first_world_rotation_right = camera_pose.rotation @ first_local_rotation_right
        
        first_position_left = wrist_positions_left[0]
        first_euler_angles_left = wrist_euler_angles_left[0]
        
        first_local_position_left = np.array(first_position_left)
        first_world_position_left = camera_pose.act(first_local_position_left)
        
        first_local_rotation_left = pin.rpy.rpyToMatrix(first_euler_angles_left)
        first_world_rotation_left = camera_pose.rotation @ first_local_rotation_left
        
        # 右臂目标
        target_pose_right = pin.SE3(first_world_rotation_right, first_world_position_right)
        viz.viewer["right_target_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0xff0000, opacity=0.8))
        viz.viewer["right_target_sphere"].set_transform(tf.translation_matrix(first_world_position_right))
        viz.viewer["right_target_axes/x"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0.15], [0, 0], [0, 0]])), g.MeshBasicMaterial(color=0xff0000)))
        viz.viewer["right_target_axes/y"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0.15], [0, 0]])), g.MeshBasicMaterial(color=0x00ff00))) 
        viz.viewer["right_target_axes/z"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0], [0, 0.15]])), g.MeshBasicMaterial(color=0x0000ff)))
        viz.viewer["right_target_axes/x"].set_transform(target_pose_right.homogeneous)
        viz.viewer["right_target_axes/y"].set_transform(target_pose_right.homogeneous)
        viz.viewer["right_target_axes/z"].set_transform(target_pose_right.homogeneous)
        
        # 左臂目标
        target_pose_left = pin.SE3(first_world_rotation_left, first_world_position_left)
        viz.viewer["left_target_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0x00ff00, opacity=0.8))
        viz.viewer["left_target_sphere"].set_transform(tf.translation_matrix(first_world_position_left))
        viz.viewer["left_target_axes/x"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0.15], [0, 0], [0, 0]])), g.MeshBasicMaterial(color=0xff0000)))
        viz.viewer["left_target_axes/y"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0.15], [0, 0]])), g.MeshBasicMaterial(color=0x00ff00))) 
        viz.viewer["left_target_axes/z"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0], [0, 0.15]])), g.MeshBasicMaterial(color=0x0000ff)))
        viz.viewer["left_target_axes/x"].set_transform(target_pose_left.homogeneous)
        viz.viewer["left_target_axes/y"].set_transform(target_pose_left.homogeneous)
        viz.viewer["left_target_axes/z"].set_transform(target_pose_left.homogeneous)

    # 输出左右机械臂末端点位置
    right_ee_frame = model.getFrameId("r_gripper_base_link")
    left_ee_frame = model.getFrameId("l_gripper_base_link")
    pin.forwardKinematics(model, data, q_current)
    pin.updateFramePlacements(model, data)
    right_ee_pos = data.oMf[right_ee_frame].translation
    left_ee_pos = data.oMf[left_ee_frame].translation
    print(f"初始右臂末端点位置: {right_ee_pos}")
    print(f"初始左臂末端点位置: {left_ee_pos}")

    # 输出左右手腕目标小球位置
    print(f"初始右手腕目标小球位置: {first_world_position_right}")
    print(f"初始左手腕目标小球位置: {first_world_position_left}")

    # --- 初始显示后暂停3秒 ---
    time.sleep(3)

    # --- 主循环：遍历轨迹 ---
    for i in range(max(len(wrist_positions_right), len(wrist_positions_left))):
        print(f"\n--- Processing frame {i} ---")

        # 处理右臂
        if i < len(wrist_positions_right):
            position_right = wrist_positions_right[i]
            euler_angles_right = wrist_euler_angles_right[i]

            local_position_right = np.array(position_right)
            world_position_right = camera_pose.act(local_position_right)
            local_rotation_right = pin.rpy.rpyToMatrix(euler_angles_right)
            world_rotation_right = camera_pose.rotation @ local_rotation_right
            world_euler_angles_right = pin.rpy.matrixToRpy(world_rotation_right)
            
            q_previous_right = q_current.copy()
            solutions_right = []
            errors_right = []
            
            for _ in range(ik_attempts):
                q_init_noisy = q_current.copy() + (np.random.rand(model.nq) - 0.5) * noise_level
                q_init_noisy = np.clip(q_init_noisy, model.lowerPositionLimit, model.upperPositionLimit)
                success, q_ik, err = compute_ik(model, data, world_position_right, world_euler_angles_right, q_init_noisy, active_idxs_right, end_effector_name="r_gripper_base_link")
                solutions_right.append(q_ik)
                errors_right.append(err)
            
            q_sol_right, error_right = select_best_solution_by_error(solutions_right, errors_right, q_previous_right, active_idxs_right, per_joint_max_delta, total_max_delta, lambda_weight)
            q_current[active_idxs_right] = q_sol_right[active_idxs_right]
    
            print(f"右臂 Frame {i}: 最终误差: {error_right:.4f}, 关节角: {np.round(q_current[active_idxs_right], 2)}")
            
            # 更新右臂目标
            target_pose_right = pin.SE3(world_rotation_right, world_position_right)
            viz.viewer["right_target_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0xff0000, opacity=0.8))
            viz.viewer["right_target_sphere"].set_transform(tf.translation_matrix(world_position_right))
            viz.viewer["right_target_axes/x"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0.15], [0, 0], [0, 0]])), g.MeshBasicMaterial(color=0xff0000)))
            viz.viewer["right_target_axes/y"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0.15], [0, 0]])), g.MeshBasicMaterial(color=0x00ff00))) 
            viz.viewer["right_target_axes/z"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0], [0, 0.15]])), g.MeshBasicMaterial(color=0x0000ff)))
            viz.viewer["right_target_axes/x"].set_transform(target_pose_right.homogeneous)
            viz.viewer["right_target_axes/y"].set_transform(target_pose_right.homogeneous)
            viz.viewer["right_target_axes/z"].set_transform(target_pose_right.homogeneous)

        # 处理左臂
        if i < len(wrist_positions_left):
            position_left = wrist_positions_left[i]
            euler_angles_left = wrist_euler_angles_left[i]

            local_position_left = np.array(position_left)
            world_position_left = camera_pose.act(local_position_left)
            local_rotation_left = pin.rpy.rpyToMatrix(euler_angles_left)
            world_rotation_left = camera_pose.rotation @ local_rotation_left
            world_euler_angles_left = pin.rpy.matrixToRpy(world_rotation_left)
            
            q_previous_left = q_current.copy()
            solutions_left = []
            errors_left = []
            
            for _ in range(ik_attempts):
                q_init_noisy = q_current.copy() + (np.random.rand(model.nq) - 0.5) * noise_level
                q_init_noisy = np.clip(q_init_noisy, model.lowerPositionLimit, model.upperPositionLimit)
                success, q_ik, err = compute_ik(model, data, world_position_left, world_euler_angles_left, q_init_noisy, active_idxs_left, end_effector_name="l_gripper_base_link")
                solutions_left.append(q_ik)
                errors_left.append(err)
            
            q_sol_left, error_left = select_best_solution_by_error(solutions_left, errors_left, q_previous_left, active_idxs_left, per_joint_max_delta, total_max_delta, lambda_weight)
            q_current[active_idxs_left] = q_sol_left[active_idxs_left]
    
            print(f"左臂 Frame {i}: 最终误差: {error_left:.4f}, 关节角: {np.round(q_current[active_idxs_left], 2)}")
            
            # 更新左臂目标
            target_pose_left = pin.SE3(world_rotation_left, world_position_left)
            viz.viewer["left_target_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0x00ff00, opacity=0.8))
            viz.viewer["left_target_sphere"].set_transform(tf.translation_matrix(world_position_left))
            viz.viewer["left_target_axes/x"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0.15], [0, 0], [0, 0]])), g.MeshBasicMaterial(color=0xff0000)))
            viz.viewer["left_target_axes/y"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0.15], [0, 0]])), g.MeshBasicMaterial(color=0x00ff00))) 
            viz.viewer["left_target_axes/z"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0], [0, 0.15]])), g.MeshBasicMaterial(color=0x0000ff)))
            viz.viewer["left_target_axes/x"].set_transform(target_pose_left.homogeneous)
            viz.viewer["left_target_axes/y"].set_transform(target_pose_left.homogeneous)
            viz.viewer["left_target_axes/z"].set_transform(target_pose_left.homogeneous)
        
        viz.display(q_current)
        
        # 更新相机视角到当前的camera_link位置
        # set_camera_to_camera_link(viz, model, data, q_current)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()