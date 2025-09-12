import pinocchio as pin
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
import time  # 用于暂停观察

# # --- 添加以下两行代码进行调试 ---
# print(f"DEBUG: Pinocchio version loaded by script: {pin.__version__}")
# print(f"DEBUG: Pinocchio module path loaded by script: {pin.__file__}")
# --- 调试代码结束 ---
# import pinocchio as pin
# print(hasattr(pin, 'normalize'))

def load_robot_model(urdf_path): #/home/tracy/airexo/AirExo-2/airexo/urdf_models/robot
    robot_dir = "/home/ryan/Documents/GitHub/AirExo-2-test/airexo/urdf_models/robot"
    
    # 将robot目录本身作为包目录，这样Pinocchio可以找到相对路径的网格文件
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path,
        package_dirs=[robot_dir],
        geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL]
    )
    data = model.createData()
    return model, data, collision_model, visual_model


def read_csv_data(file_path):
    data = pd.read_csv(file_path)
    positions = data[['x', 'y', 'z']].values
    euler_angles_deg = data[['rx', 'ry', 'rz']].values
    euler_angles_rad = np.deg2rad(euler_angles_deg)  # 统一转弧度
    return positions, euler_angles_rad

import pinocchio as pin
import numpy as np

def compute_ik(model, data, target_position, target_euler_angles, q_init, active_idxs, 
             max_iter=2000, eps=0.08, stall_threshold=50, damp_base=1e-3, alpha=0.1): # 增加alpha参数
    """
    使用阻尼最小二乘法（DLS）求解逆运动学 (修正版)
    """
    end_effector_name = "r_gripper_base_link"
    frame_id = model.getFrameId(end_effector_name)
    if frame_id < 0 or frame_id >= len(model.frames):
        raise ValueError(f"End effector frame '{end_effector_name}' not found")

    target_rotation = pin.rpy.rpyToMatrix(target_euler_angles)
    target_pose = pin.SE3(target_rotation, target_position)

    q = q_init.copy()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    current_pose = data.oMf[frame_id]
    
    print("\n--- IK 求解开始 ---")
    print(f"目标位置: {target_position}, 目标欧拉角: {target_euler_angles}")
    print(f"初始末端位置: {current_pose.translation}, 初始欧拉角: {pin.rpy.matrixToRpy(current_pose.rotation)}")

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
            print(f"✅ IK 收敛，第 {i} 次迭代，误差: {error_norm:.6f}")
            return True, q, error_norm

        if abs(error_norm - prev_error_norm) < 1e-6: # 修改早停判断，基于误差变化量
            stall_count += 1
            if stall_count > stall_threshold:
                print(f"⚠️ IK 早停（误差不再减小），第 {i} 次迭代，当前误差: {error_norm:.6f}, 最佳误差: {min_error:.6f}")
                return False, best_q, min_error
        else:
            stall_count = 0

        prev_error_norm = error_norm

        J = pin.computeFrameJacobian(model, data, q, frame_id)
        J_active = J[:, active_idxs]

        damp = min(max(damp_base, error_norm * 0.05), 0.05)
        JJt = J_active @ J_active.T + damp * np.eye(6)
        
        # error_vec 是从 target 到 current 的变换，我们需要朝反方向移动来减小它
        dq_active = J_active.T @ np.linalg.solve(JJt, error_vec)

        dq = np.zeros(model.nv)
        dq[active_idxs] = dq_active

        # 【核心修改】使用带步长的负方向更新
        q = pin.integrate(model, q, -alpha * dq)
        
        q[active_idxs] = np.clip(q[active_idxs], model.lowerPositionLimit[active_idxs], model.upperPositionLimit[active_idxs])
        if hasattr(pin, 'normalize'):
            q[active_idxs] = pin.normalize(model, q)[active_idxs]

    print(f"⚠️ IK 未收敛，最大迭代次数 {max_iter}，最终误差: {error_norm:.6f}, 最佳误差: {min_error:.6f}")
    return False, best_q, min_error


def select_best_solution_by_error(solutions, errors, q_previous, active_idxs, per_joint_max_delta, total_max_delta, lambda_weight):
    valid_solutions = []
    valid_errors = []
    valid_delta_norms = []
    
    for q_ik, err in zip(solutions, errors):
        delta_q = q_ik[active_idxs] - q_previous[active_idxs]
        delta_q_norm = np.linalg.norm(delta_q)  # 总范数
        max_delta_per_joint = np.max(np.abs(delta_q))  # 最大单个关节变化
        
        if max_delta_per_joint <= per_joint_max_delta and delta_q_norm <= total_max_delta:
            valid_solutions.append(q_ik)
            valid_errors.append(err)
            valid_delta_norms.append(delta_q_norm)
    
    if valid_solutions:
        # 在有效解中，选择最小化 (error + lambda * delta_norm)
        scores = [err + lambda_weight * delta for err, delta in zip(valid_errors, valid_delta_norms)]
        min_score_idx = np.argmin(scores)
        print(f"Selected solution with error: {valid_errors[min_score_idx]}, delta_norm: {valid_delta_norms[min_score_idx]}")
        return valid_solutions[min_score_idx], valid_errors[min_score_idx]
    else:
        # Fallback: 如果无有效解，选min(error)，但打印警告
        min_error_idx = np.argmin(errors)
        print(f"⚠️ No valid solutions within delta limits, falling back to min error: {errors[min_error_idx]}")
        return solutions[min_error_idx], errors[min_error_idx]



# def show_camera_axes(viz, model, data, q_current, camera_link_name="camera_link", axis_length=0.15, set_camera_view=True):
#     """
#     在 Meshcat 中显示指定相机 link 的坐标系，并设置可视化视角为相机视角。
    
#     参数：
#         viz: MeshcatVisualizer 对象
#         model: Pinocchio 机器人模型
#         data: Pinocchio 机器人数据
#         q_current: 当前关节角 numpy 数组
#         camera_link_name: 相机对应的 link 名称
#         axis_length: 坐标轴长度
#         set_camera_view: 是否自动设置为相机视角
#     """
    
#     # 1. 获取相机 link 位姿
#     camera_link_id = model.getFrameId(camera_link_name)
#     pin.forwardKinematics(model, data, q_current)
#     pin.updateFramePlacements(model, data)
#     camera_se3 = data.oMf[camera_link_id]
#     T_world_camera = np.array(camera_se3.homogeneous)  # 4x4 numpy 矩阵
#     print(f"{camera_link_name} 位姿:\n{T_world_camera}")
    
#     # 2. 绘制坐标轴
#     x_points = np.array([[0, axis_length], [0, 0], [0, 0]])
#     y_points = np.array([[0, 0], [0, axis_length], [0, 0]])
#     z_points = np.array([[0, 0], [0, 0], [0, axis_length]])
    
#     viz.viewer["camera_axes/x"].set_object(
#         g.Line(g.PointsGeometry(x_points), g.MeshBasicMaterial(color=0xff0000))
#     )
#     viz.viewer["camera_axes/y"].set_object(
#         g.Line(g.PointsGeometry(y_points), g.MeshBasicMaterial(color=0x00ff00))
#     )
#     viz.viewer["camera_axes/z"].set_object(
#         g.Line(g.PointsGeometry(z_points), g.MeshBasicMaterial(color=0x0000ff))
#     )
    
#     # 3. 设置坐标轴位姿
#     viz.viewer["camera_axes/x"].set_transform(T_world_camera)
#     viz.viewer["camera_axes/y"].set_transform(T_world_camera)
#     viz.viewer["camera_axes/z"].set_transform(T_world_camera)
    
#     if set_camera_view:
#         # 4. 设置 Meshcat 视角为机器人相机视角
#         position = camera_se3.translation
#         rotation = camera_se3.rotation
        
#         # ROS 相机坐标系：z 前向，x 右向，y 下向
#         # 对于 Meshcat 相机视角，我们需要：
#         # - forward：相机看向的方向（相机 z 轴正方向）
#         # - up：相机的上方向（相机 -y 轴方向，因为 ROS 相机 y 轴向下）
        
#         forward_local = np.array([0.0, -1.0, 0.0])  # 相机本地坐标系 z 轴
#         up_local = np.array([0.0, 0.0, 1.0])      # 相机本地坐标系 -y 轴
        
#         # 可选：添加向下倾斜角度来更好地观察机器人
#         tilt_angle = 0 * np.pi / 180  # -30° 向下倾斜，可以调整这个值
#         if tilt_angle != 0:
#             # 绕相机 x 轴旋转（俯仰）
#             cos_t, sin_t = np.cos(tilt_angle), np.sin(tilt_angle)
#             R_tilt = np.array([
#                 [1.0, 0.0, 0.0],
#                 [0.0, cos_t, -sin_t],
#                 [0.0, sin_t, cos_t]
#             ])
#             forward_local = R_tilt @ forward_local
#             up_local = R_tilt @ up_local
        
#         # 转换到世界坐标系
#         forward_world = rotation @ forward_local
#         up_world = rotation @ up_local


#         # try:
#         viz.viewer[camera_link_name].set_property("visible", False)
#         # except Exception as e:
#         #     print(f"Warning: Unable to set property for {camera_link_name}. Error: {e}")

#         # # 关键修改：将相机位置稍微向后移动，避免在头部内部
#         # offset_distance = 0.1  # 向后偏移距离，可以调整这个值
#         # camera_position = position + forward_world * offset_distance  # 相机稍微向后
#         lookat_distance = 1.0  # 看向前方的距离
#         lookat = np.array([0.0, 1.0, 0.0]) # 看向头部前方

#         camera_position = np.array([-0.0032391, -0.590296, 2.502606])  # 随机示例坐标，你可以替换为所需值

#         print(f"设置相机视角:")
#         print(f"  位置: {position}")
#         print(f"  看向: {lookat}")
#         print(f"  上方向: {up_world}")

#         # 计算相机前向向量和上向向量
#         forward_vector = lookat - position
#         forward_vector /= np.linalg.norm(forward_vector)  # 单位化
#         up_vector = up_world / np.linalg.norm(up_world)  # 单位化

#         #     # 设置相机位置和方向
#         # viz.viewer["/Cameras/default"].set_property("position", camera_position.tolist())
#         # viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [0, 0, 0])
#         # viz.viewer["/Cameras/default/rotated/<object>"].set_property("up", up_world.tolist())        
#         # viz.viewer["/Cameras/default/rotated/<object>"].set_property("forward", forward_vector.tolist())
            
#         lookat_distance = 1.0  # 看向前方的距离
#         lookat_target = camera_position + forward_world * lookat_distance
#         # 放大一些长度用于可视化
#         vec_len = 0.3
#         forward_points = np.array([
#             [0, forward_vector[0]*vec_len],
#             [0, forward_vector[1]*vec_len],
#             [0, forward_vector[2]*vec_len]
#         ])
#         up_points = np.array([
#             [0, up_vector[0]*vec_len],
#             [0, up_vector[1]*vec_len],
#             [0, up_vector[2]*vec_len]
#         ])

#         # 在相机位置处绘制朝向箭头
#         viz.viewer["camera_dir/forward"].set_object(
#             g.Line(g.PointsGeometry(forward_points), g.MeshBasicMaterial(color=0xff0000))
#         )
#         viz.viewer["camera_dir/forward"].set_transform(tf.translation_matrix(camera_position))

#         # viz.viewer["camera_dir/up"].set_object(
#         #     g.Line(g.PointsGeometry(up_points), g.MeshBasicMaterial(color=0x00ff00))
#         # )
#         # viz.viewer["camera_dir/up"].set_transform(tf.translation_matrix(new_position))

        
#         # 在头部相机前方放置一个黄色球
#         ball_distance = 0.5  # 球距离相机的距离
#         ball_position = position + forward_world * ball_distance

#         # 设置球体对象
#         viz.viewer["camera_ball"].set_object(
#             g.Sphere(0.05), g.MeshLambertMaterial(color=0xffff00, opacity=0.8)
#         )
#         viz.viewer["camera_ball"].set_transform(tf.translation_matrix(ball_position))

#         print(f"在头部相机绿色坐标轴的反方向放置黄色球，位置: {ball_position}")
#         # 设置相机视角 —— 始终看向黄色小球

#         camera_pos = np.array([-0.0032, -0.2903, 1.5026])
#         target_pos = np.array([-0.0032, -0.5903, 1.5026])

#         # 1. 设置焦点
#         T = np.eye(4)
#         T[:3, 3] = target_pos  # 焦点在 target_pos
#         # viz.viewer["/Cameras/default"].set_transform(T)
#         # 2. 设置相机相对焦点的偏移
#         offset = camera_pos - target_pos
#         # viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", offset.tolist())
#         # viz.viewer["/Cameras/default"].set_property("position", camera_position.tolist())
#         # viz.viewer["/Cameras/default"].set_property("target", camera_target.tolist())
#         # viz.viewer["/Cameras/default/rotated/<object>"].set_property("up", up_world.tolist())
#         look_at(viz,
#         camera_pos=[-0.0032, 0.2, 1.7526],
#         target_pos=[-0.0032, -0.5903, 1.5026])


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
    urdf_path = "/home/ryan/Documents/GitHub/AirExo-2-test/airexo/urdf_models/robot/true_robot.urdf"
    csv_file_path = "/home/ryan/Documents/GitHub/AirExo-2-test/train_video/hand_landmarks_3d_offline.csv"

    # 接收所有四个对象
    model, data, collision_model, visual_model = load_robot_model(urdf_path)
    active_joints = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']
    active_idxs = []
    for name in active_joints:
        jid = model.getJointId(name)
        if jid < model.njoints:
            idx_q = model.joints[jid].idx_q
            for i in range(model.joints[jid].nq):
                active_idxs.append(idx_q + i)
        else:
            raise ValueError(f"Joint '{name}' not found in model.")

    print(f"活跃关节索引: {active_idxs}")
    print(f"活跃关节数量: {len(active_idxs)} (应为7)")
    # 新增：每帧关节变化约束
    per_joint_max_delta = 0.15 #0.0873 # 每个关节最大允许变化（弧度），根据臂速度调整
    total_max_delta = 0.4 #0.23      # 所有活跃关节变化范数的最大允许值
    lambda_weight = 0.1        # 加权delta_q的权重（用于最小化 error + lambda * delta_norm）

    # --- 新增：为IK求解定义随机噪声幅度 ---
    noise_level = 0.2 #0.1 # 随机噪声的幅度（弧度），可以调整这个值

    # 3. 增加尝试次数：进行更广泛的搜索
    ik_attempts = 20              # 在下面的 for 循环中使用这个变量，原为 10 次

    # 手动设置相机到世界坐标系的转换矩阵
    t_base_cam = np.array([0.3,-0.85,1.41])  # 单位：米
    rpy = np.array([np.deg2rad(90), 0.0, np.deg2rad(-90)])  # 弧度
    R_base_cam = pin.rpy.rpyToMatrix(rpy)  # 包含 z 轴翻转
    camera_pose = pin.SE3(R_base_cam, t_base_cam)
    print("T_base_cam (手动写死):\n", camera_pose.homogeneous)

    wrist_positions, wrist_euler_angles = read_csv_data(csv_file_path)

    # 初始化 Meshcat 可视化
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer()
    viz.loadViewerModel()  # 加载 URDF 到浏览器
    
    viz.viewer.open()
    q_current = pin.neutral(model)  # 初始 q

    look_at(viz,camera_pos=[-0.0032, -0.19,1.6526],target_pos=[-0.0032, -0.3903, 1.6026])


    # 调试：打印 CSV 数据
    print(f"CSV positions (first 5): {wrist_positions[:5]}")
    print(f"CSV euler angles (first 5): {wrist_euler_angles[:5]}")
    print(f"Number of joints (nq): {model.nq}")
    
    # 替换原有q_init：自定义自然向前伸展初始姿态
    q_start_values = np.array([1.2, 1.21,0.82,0.56, 0, 0.0, 0.0])
    q_start = np.zeros(model.nq)  # 全零基底
    q_start[active_idxs] = np.clip(q_start_values, model.lowerPositionLimit[active_idxs], model.upperPositionLimit[active_idxs])
    q_current = q_start.copy()  # 用自定义初始替换
    print(f"自定义初始关节角度 (自然向前伸展): {q_start[active_idxs]}")

    print("\n--- IK 功能健全性测试 ---")
    # 在 for 循环前，用于测试
    test_pos = np.array([0.6, 0.0, 1.2]) # 正前方 0.5m，高度 1.2m
    test_rot = pin.rpy.rpyToMatrix(0, 0, 0) # 无旋转
    test_q_init = pin.neutral(model)
    success, q_sol, err = compute_ik(model, data, test_pos, pin.rpy.matrixToRpy(test_rot), test_q_init, active_idxs)
    if success:
        print("✅ 测试目标可达，IK求解器工作正常！")
        # (可选) 在可视化中显示测试结果，以直观确认
        viz.display(q_sol)
        print("在可视化窗口中显示测试结果，5秒后继续处理CSV轨迹...")
        time.sleep(5)
    else:
        print("❌ 测试目标不可达，请检查IK算法或模型！")
    print("--- 测试结束 ---\n")



    # 新增：初始显示 neutral 姿态 + 第一个目标球体/轴，并暂停
    if len(wrist_positions) > 0:
        first_position = wrist_positions[0]
        first_euler_angles = wrist_euler_angles[0]
        
        first_local_position = np.array(first_position)
        first_world_position = camera_pose.act(first_local_position)
        
        first_local_rotation = pin.rpy.rpyToMatrix(first_euler_angles)
        first_world_rotation = camera_pose.rotation @ first_local_rotation
        
        print(f"初始转换后世界位置: {first_world_position}")
        print(f"初始转换后世界旋转矩阵: {first_world_rotation}")
        
        viz.display(q_current)
        
        target_pose = pin.SE3(first_world_rotation, first_world_position)
        
        viz.viewer["target_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0xff0000, opacity=0.8))
        viz.viewer["target_sphere"].set_transform(tf.translation_matrix(first_world_position))
        
        axis_length = 0.15
        x_points = np.array([[0, axis_length], [0, 0], [0, 0]])
        y_points = np.array([[0, 0], [0, axis_length], [0, 0]])
        z_points = np.array([[0, 0], [0, 0], [0, axis_length]])
        
        viz.viewer["target_axes/x"].set_object(g.Line(g.PointsGeometry(x_points), g.MeshBasicMaterial(color=0xff0000)))
        viz.viewer["target_axes/y"].set_object(g.Line(g.PointsGeometry(y_points), g.MeshBasicMaterial(color=0xffff00)))
        viz.viewer["target_axes/z"].set_object(g.Line(g.PointsGeometry(z_points), g.MeshBasicMaterial(color=0x0000ff)))
        
        target_tf = target_pose.homogeneous
        viz.viewer["target_axes/x"].set_transform(target_tf)
        viz.viewer["target_axes/y"].set_transform(target_tf)
        viz.viewer["target_axes/z"].set_transform(target_tf)
        
        print("初始状态观察中：机械臂 (自定义自然姿态) 与第一个目标 (红色球体及坐标轴) 的差距。")
        print(f"初始红色小球位置 (本地): {first_position}")
        print(f"初始红色小球位置 (世界): {first_world_position}")
        time.sleep(3)


    for i, (position, euler_angles) in enumerate(zip(wrist_positions, wrist_euler_angles)):
        print(f"\nProcessing frame {i}: Position={position}, Euler angles={euler_angles}")

        local_position = np.array(position)
        world_position = camera_pose.act(local_position)
        local_rotation = pin.rpy.rpyToMatrix(euler_angles)
        world_rotation = camera_pose.rotation @ local_rotation
        world_euler_angles = pin.rpy.matrixToRpy(world_rotation)
        
        print(f"转换后世界位置: {world_position}")
        print(f"转换后世界旋转矩阵: {world_rotation}")
       
        if i == 0:
            print(f"第一帧使用自定义初始前3关节: {q_current[active_idxs[:3]]} (已粗略指向目标)")
            viz.display(q_current)
            print("显示自定义初始姿态（粗略指向目标），观察与目标差距。")
            time.sleep(3)
            
            q_previous = q_start.copy()
            
            active_ik_idxs = active_idxs[3:]
            solutions = []
            errors = []
            for _ in range(1):
                success, q_ik, err = compute_ik(model, data, world_position, world_euler_angles, q_current.copy(), active_ik_idxs)
                if success:
                    solutions.append(q_ik)
                    errors.append(err)
            
            if solutions:
                q_current, error = select_best_solution_by_error(solutions, errors, q_previous, active_idxs, per_joint_max_delta, total_max_delta, lambda_weight)
                # 🔧 保证非机械臂关节不动
                mask = np.ones(model.nq, dtype=bool)
                mask[active_idxs] = False
                q_current[mask] = q_start[mask]                
                print(f"第一帧后4关节优化后: {q_current[active_idxs[3:]]}")
            else:
                print("第一帧IK无解，使用fallback。")
                q_current = q_previous.copy()

        else:
            # 后续帧：继承上一帧q，优化所有关节，多次尝试选最佳
            q_previous = q_current.copy()
            solutions = []
            errors = []
            
            # --- 方案一修改开始 ---
            for _ in range(ik_attempts): # 原为 range(10)
        # 为初始猜测添加随机噪声，以探索不同的解
                q_init_noisy = q_current.copy() + (np.random.rand(model.nq) - 0.5) * noise_level
                
                # 确保噪声后的q仍在关节限制内
                q_init_noisy = np.clip(q_init_noisy, model.lowerPositionLimit, model.upperPositionLimit)

                # 使用带噪声的初始值进行求解
                success, q_ik, err = compute_ik(model, data, world_position, world_euler_angles, q_init_noisy, active_idxs)
                solutions.append(q_ik)
                errors.append(err)
            # --- 方案一修改结束 ---
            
            q_current, error = select_best_solution_by_error(solutions, errors, q_previous, active_idxs, per_joint_max_delta, total_max_delta, lambda_weight)
            # 🔧 保证非机械臂关节不动
            mask = np.ones(model.nq, dtype=bool)
            mask[active_idxs] = False
            q_current[mask] = q_start[mask]    
        print(f"Frame {i}: Joint Angles: {q_current[active_idxs]}, Error: {error}")
        viz.display(q_current)
        
        target_pose = pin.SE3(world_rotation, world_position)
        
        viz.viewer["target_sphere"].set_object(g.Sphere(0.05), g.MeshLambertMaterial(color=0xff0000, opacity=0.8))
        viz.viewer["target_sphere"].set_transform(tf.translation_matrix(world_position))
        
        axis_length = 0.15
        x_points = np.array([[0, axis_length], [0, 0], [0, 0]])
        y_points = np.array([[0, 0], [0, axis_length], [0, 0]])
        z_points = np.array([[0, 0], [0, 0], [0, axis_length]])
        
        viz.viewer["target_axes/x"].set_object(g.Line(g.PointsGeometry(x_points), g.MeshBasicMaterial(color=0xff0000)))
        viz.viewer["target_axes/y"].set_object(g.Line(g.PointsGeometry(y_points), g.MeshBasicMaterial(color=0xffff00)))
        viz.viewer["target_axes/z"].set_object(g.Line(g.PointsGeometry(z_points), g.MeshBasicMaterial(color=0x0000ff)))
        
        target_tf = target_pose.homogeneous
        viz.viewer["target_axes/x"].set_transform(target_tf)
        viz.viewer["target_axes/y"].set_transform(target_tf)
        viz.viewer["target_axes/z"].set_transform(target_tf)
        
        print(f"Frame {i} 可视化更新：观察浏览器中机械臂 (当前 q) 与红色球体及坐标轴 (目标) 的差距。")
        
        time.sleep(2)

        pin.forwardKinematics(model, data, q_current)
        pin.updateFramePlacements(model, data)
        current_pose = data.oMf[model.getFrameId("r_gripper_base_link")]
        print(f"当前末端位置: {current_pose.translation}, 目标位置 (世界): {world_position}")
        print(f"Frame {i} 红色小球位置 (本地): {position}")
        print(f"Frame {i} 红色小球位置 (世界): {world_position}")

if __name__ == "__main__":
    main()