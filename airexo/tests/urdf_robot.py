import os
import hydra
import numpy as np
import open3d as o3d

from omegaconf import OmegaConf

from airexo.helpers.constants import *
from airexo.helpers.urdf_robot import forward_kinematic

@hydra.main(
    version_base = None,
    config_path = os.path.join("..", "configs", "tests", "urdf"),
    config_name = "robot"
)
def main(cfg):
    OmegaConf.resolve(cfg)

    cur_transforms, visuals_map = forward_kinematic(
        left_joint = cfg.left_joint, 
        right_joint = cfg.right_joint,
        left_joint_cfgs = cfg.robot_left_joint_cfgs,
        right_joint_cfgs = cfg.robot_right_joint_cfgs,
        is_rad = False,
        urdf_file = cfg.urdf_file,
        with_visuals_map = True
    )

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width = 1280, height = 720)
    
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2).transform(ROBOT_PREDEFINED_TRANSFORMATION)
    visualizer.add_geometry(frame)

    model_meshes = {}
    for link, transform in cur_transforms.items():
        model_meshes[link] = {}
        for v in visuals_map[link]:
            if v.geom_param is None:
                continue

            tf = ROBOT_PREDEFINED_TRANSFORMATION @ transform.matrix() @ v.offset.matrix()

            if isinstance(v.geom_param, str):
                # mesh文件路径
                mesh_path = os.path.join(os.path.dirname(cfg.urdf_file), v.geom_param)
                mesh = o3d.io.read_triangle_mesh(mesh_path)

            elif isinstance(v.geom_param, tuple):
                # 如果是元组，区分格式：
                if len(v.geom_param) == 3 and v.geom_param[0] == 'cylinder':
                    # ('cylinder', length, radius)
                    length = v.geom_param[1]
                    radius = v.geom_param[2]
                    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)

                elif len(v.geom_param) == 2 and all(isinstance(x, (int, float)) for x in v.geom_param):
                    # (length, radius) 直接当圆柱体处理
                    length, radius = v.geom_param
                    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)

                else:
                    print(f"Unknown geometry param for link {link}: {v.geom_param}")
                    continue
            else:
                print(f"Unknown geometry param type for link {link}: {v.geom_param}")
                continue

            mesh.transform(tf)
            mesh.compute_vertex_normals()
            model_meshes[link][v.geom_param] = mesh
            visualizer.add_geometry(mesh, reset_bounding_box=True)


    visualizer.run()

if __name__ == '__main__':
    main()