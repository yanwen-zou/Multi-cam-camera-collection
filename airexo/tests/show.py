import open3d as o3d
import xml.etree.ElementTree as ET
import numpy as np
import os

def get_transform_from_origin(visual):
    """从URDF的<origin>标签提取变换矩阵"""
    origin = visual.find('origin')
    if origin is None:
        return np.eye(4)
    xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
    rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
    
    # 平移矩阵
    T = np.eye(4)
    T[:3, 3] = xyz
    
    # 旋转矩阵（Roll-Pitch-Yaw）
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rpy[0]), -np.sin(rpy[0])],
                   [0, np.sin(rpy[0]), np.cos(rpy[0])]])
    Ry = np.array([[np.cos(rpy[1]), 0, np.sin(rpy[1])],
                   [0, 1, 0],
                   [-np.sin(rpy[1]), 0, np.cos(rpy[1])]])
    Rz = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]), 0],
                   [np.sin(rpy[2]), np.cos(rpy[2]), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T[:3, :3] = R
    
    return T

def load_and_adjust_meshes(urdf_file, package_base_path):
    # 解析URDF文件
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    meshes = []
    
    # 遍历所有<link>
    for link in root.findall('link'):
        link_name = link.get('name')
        visual = link.find('visual')
        if visual is not None:
            mesh_tag = visual.find('geometry/mesh')
            if mesh_tag is not None:
                # 提取并解析网格路径
                mesh_path = mesh_tag.get('filename')
                if mesh_path.startswith("package://"):
                    mesh_path = mesh_path.replace("package://", package_base_path)
                mesh_path = os.path.abspath(mesh_path)
                
                # 加载网格
                if os.path.exists(mesh_path):
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    
                    # 应用URDF中的<origin>变换
                    transform = get_transform_from_origin(visual)
                    mesh.transform(transform)
                    
                    # 可选：为网格着色以区分
                    mesh.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
                    
                    meshes.append((link_name, mesh))
                else:
                    print(f"Mesh file not found: {mesh_path}")
    
    return meshes

def apply_global_correction(meshes, rotation_axis=[1, 0, 0], angle_deg=90):
    """对所有网格应用全局旋转校正"""
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        rotation_axis, np.deg2rad(angle_deg))
    for _, mesh in meshes:
        mesh.rotate(rotation_matrix, center=(0, 0, 0))
    return meshes

# 使用示例
urdf_file = "airexo/urdf_models/robot/our_robot.urdf"  # 替换为你的URDF文件路径
package_base_path = "airexo/urdf_models/robot/meshes"  # 替换为你的STL文件目录
meshes = load_and_adjust_meshes(urdf_file, package_base_path)

# 如果轴向错误，应用全局校正（例如绕x轴旋转90度）
meshes = apply_global_correction(meshes, rotation_axis=[1, 0, 0], angle_deg=90)

# 可视化
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
o3d.visualization.draw_geometries([mesh for _, mesh in meshes] + [coord_frame])

# 可选：保存调整后的网格
for link_name, mesh in meshes:
    o3d.io.write_triangle_mesh(f"adjusted_{link_name}.stl", mesh)