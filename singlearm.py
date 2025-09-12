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


def compute_ik(model, data, target_position, target_euler_angles, q_init, right_active_idxs, 
             max_iter=2000, eps=0.01, stall_threshold=50, damp_base=1e-3, alpha=0.1):
    """
    使用带步长和阻尼的最小二乘法求解逆运动学 (最终优化版)。
    """
    end_effector_name = "r_gripper_base_link"
    frame_id = model.getFrameId(end_effector_name)
    if frame_id < 0 or frame_id >= len(model.frames):
        # 对于简化模型，Pinocchio可能会重命名frame。我们需要找到正确的名称。
        # 这是一个常见的处理方式：假设末端连杆的名称在URDF中是固定的。
        # 'r_gripper_base_link' 是连杆(link)名，它关联的frame可能被重命名。
        # 我们尝试直接用连杆名查找。
        link_id = model.getFrameId("r_gripper_base_link") # 假设连杆名不变
        if link_id < len(model.frames):
             frame_id = link_id
        else:
             raise ValueError(f"End effector frame '{end_effector_name}' not found in the reduced model.")

    target_rotation = pin.rpy.rpyToMatrix(target_euler_angles)
    target_pose = pin.SE3(target_rotation, target_position)

    q = q_init.copy()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    # 在简化模型中，frame ID可能会改变，需要重新获取
    frame_id = model.getFrameId(end_effector_name)
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
        J_active = J[:, right_active_idxs]

        damp = min(max(damp_base, error_norm * 0.05), 0.05)
        JJt = J_active @ J_active.T + damp * np.eye(6)
        
        dq_active = J_active.T @ np.linalg.solve(JJt, error_vec)
        dq = np.zeros(model.nv)
        dq[right_active_idxs] = dq_active

        q = pin.integrate(model, q, -alpha * dq)
        
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    return False, best_q, min_error


def select_best_solution_by_error(solutions, errors, q_previous, right_active_idxs, per_joint_max_delta, total_max_delta, lambda_weight):
    """
    从多个IK解中选择一个既精确又平滑的解。
    """
    valid_solutions = []
    valid_errors = []
    valid_delta_norms = []
    
    for q_ik, err in zip(solutions, errors):
        delta_q = q_ik[right_active_idxs] - q_previous[right_active_idxs]
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
    csv_file_path = "/home/ryan/Documents/GitHub/AirExo-2-test/train_video/hand_landmarks_3d_offline.csv"

    # --- 模型加载 ---
    model, data, collision_model, visual_model = load_robot_models(urdf_path)
    
    # --- 核心参数调整区 ---
    per_joint_max_delta = 0.15      
    total_max_delta = 0.4         
    noise_level = 0.2             
    ik_attempts = 20              
    lambda_weight = 0.1           
    
    # --- 关节定义 ---
    right_active_joints = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']
    # print(f"简化模型活跃关节索引: {right_active_idxs}")

    right_active_idxs = []
    for name in right_active_joints:
        jid = model.getJointId(name)
        if jid < model.njoints:
            idx_q = model.joints[jid].idx_q
            for i in range(model.joints[jid].nq):
                right_active_idxs.append(idx_q + i)
        else:
            raise ValueError(f"Joint '{name}' not found in model.")

    # --- 坐标系变换定义 ---
    t_base_cam = np.array([0.3, -0.7, 1.41])
    rpy = np.array([np.deg2rad(90), 0.0, np.deg2rad(-90)])
    R_base_cam = pin.rpy.rpyToMatrix(rpy)
    camera_pose = pin.SE3(R_base_cam, t_base_cam)

    # --- 数据加载 ---
    wrist_positions, wrist_euler_angles = read_csv_data(csv_file_path)

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
    q_start_values = np.array([1.2, 1.21, 0.82, 0.56, 0.0, 0.0, 0.0])
    q_start = pin.neutral(model)  # 获取中立姿态，长度为 45
    q_start[right_active_idxs] = q_start_values  # 只替换右臂关节
    q_start = np.clip(q_start, model.lowerPositionLimit, model.upperPositionLimit)
    q_current = q_start.copy()
    print(f"自定义初始关节角度: {q_start}")


    
    viz.display(q_current)
    
# 设置相机视角到camera_link
    look_at(viz,camera_pos=[-0.0032, 0.01, 1.7026],target_pos=[-0.0032, -0.5903, 1.3026])
    
    time.sleep(2)

    # --- 主循环：遍历轨迹 ---
    for i, (position, euler_angles) in enumerate(zip(wrist_positions, wrist_euler_angles)):
        print(f"\n--- Processing frame {i} ---")

        local_position = np.array(position)
        world_position = camera_pose.act(local_position)
        local_rotation = pin.rpy.rpyToMatrix(euler_angles)
        world_rotation = camera_pose.rotation @ local_rotation
        world_euler_angles = pin.rpy.matrixToRpy(world_rotation)
        
        q_previous = q_current.copy()
        solutions = []
        errors = []
        
        for _ in range(ik_attempts):
            q_init_noisy = q_current.copy() + (np.random.rand(model.nq) - 0.5) * noise_level
            q_init_noisy = np.clip(q_init_noisy, model.lowerPositionLimit, model.upperPositionLimit)
            success, q_ik, err = compute_ik(model, data, world_position, world_euler_angles, q_init_noisy, right_active_idxs)
            solutions.append(q_ik)
            errors.append(err)
        
        q_current, error = select_best_solution_by_error(solutions, errors, q_previous, right_active_idxs, per_joint_max_delta, total_max_delta, lambda_weight)
    
        print(f"Frame {i}: 最终误差: {error:.4f}, 关节角: {np.round(q_current, 2)}")
        
        viz.display(q_current)
        
        # 更新相机视角到当前的camera_link位置
        # set_camera_to_camera_link(viz, model, data, q_current)
        
        target_pose = pin.SE3(world_rotation, world_position)
        # viz.viewer["target_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0xff0000, opacity=0.8))
        # viz.viewer["target_sphere"].set_transform(tf.translation_matrix(world_position))
        # viz.viewer["target_axes/x"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0.15], [0, 0], [0, 0]])), g.MeshBasicMaterial(color=0xff0000)))
        # viz.viewer["target_axes/y"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0.15], [0, 0]])), g.MeshBasicMaterial(color=0x00ff00))) 
        # viz.viewer["target_axes/z"].set_object(g.Line(g.PointsGeometry(np.array([[0, 0], [0, 0], [0, 0.15]])), g.MeshBasicMaterial(color=0x0000ff)))
        # viz.viewer["target_axes/x"].set_transform(target_pose.homogeneous)
        # viz.viewer["target_axes/y"].set_transform(target_pose.homogeneous)
        # viz.viewer["target_axes/z"].set_transform(target_pose.homogeneous)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()