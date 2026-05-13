import argparse
import json
import os
import math
import numpy as np
import time
import cv2  # 使用OpenCV处理键盘输入
from rich import print
from PIL import Image


from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import ControlEnv
import robosuite.utils.transform_utils as T  # 用于旋转变换


IMAGE_RESOLUTION = 256
CAMERA_NAME = "agentview"
IMAGE_SAVE_PATH = "LIBERO/xyg_scripts/debug_images2"
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)


def get_libero_env(task, model_family, resolution=256):
    """初始化并返回LIBERO环境及任务描述"""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": resolution, 
        "camera_widths": resolution, 
        "hard_reset": False,
        "has_renderer": True, 
        "has_offscreen_renderer": False, 
        "use_camera_obs": False, 
        "render_camera": CAMERA_NAME
    }
    env = ControlEnv(**env_args)
    env.seed(0)
    return env, task_description


def quat_to_euler(quat):
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]

    alpha = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    beta = math.asin(2 * (qw * qy - qz * qx))
    gamma = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    
    euler = np.array([alpha, beta, gamma])
    
    return euler


def euler_to_quat(euler):
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    
    cr = np.cos(roll * 0.5);
    sr = np.sin(roll * 0.5);
    cp = np.cos(pitch * 0.5);
    sp = np.sin(pitch * 0.5);
    cy = np.cos(yaw * 0.5);
    sy = np.sin(yaw * 0.5);
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    quat = np.array([qx, qy, qz, qw])
    return quat


def change_camera_viewpoint(env, camera_id, camera_name):
    cur_camera_pos = env.sim.model.cam_pos[camera_id].copy()
    cur_camera_quat = env.sim.model.cam_quat[camera_id].copy()
    cur_camera_euler = quat_to_euler(cur_camera_quat)

    delta_camera_pos = np.array([0.5, 0.5, 0.5])
    # delta_camera_pos = np.array([0.5, 0.5, 0.5])
    delta_camera_euler = np.array([0.0, 0.0, 0.0])
    
    tgt_camera_pos = cur_camera_pos + delta_camera_pos
    tgt_camera_quat = euler_to_quat(cur_camera_euler + delta_camera_euler)
    
    env.sim.model.cam_pos[camera_id] = tgt_camera_pos
    env.sim.model.cam_quat[camera_id] = tgt_camera_quat
    env.sim.forward()
    
    camera_img = env.sim.render(
        camera_name=camera_name,  # 指定相机名称
        width=IMAGE_RESOLUTION,                  # 图像宽度
        height=IMAGE_RESOLUTION,                 # 图像高度
        depth=False,                # 是否需要深度图
        mode='offscreen'            # 离屏渲染模式
    )
    
    Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"camera_observation_trans.png"))
    

def change_object_position_and_pose(env, object_name):
    position_delta = np.array([0.1, 0.1, 0.1])
    rotation_delta = np.array([0.0, 0.0, 0.0])
    
    # 查找所有匹配的物体
    total_body_names = [name for name in env.sim.model.body_names if name and name != "world"]
    object_names = [name for name in total_body_names if object_name in name]
    
    if not object_names:
        print(f"找不到名称包含 '{object_name}' 的物体!")
        return False
    
    print(f"找到 {len(object_names)} 个匹配的物体: {object_names}")
    
    position_delta = np.array(position_delta) if position_delta is not None else np.zeros(3)
    rotation_delta = np.array(rotation_delta) if rotation_delta is not None else np.zeros(3)
    
    for obj_name in object_names:
        object_id = env.sim.model.body_name2id(obj_name)
        
        # 方法1: 通过关节修改位置 (适用于有自由关节的物体)
        joint_found = False
        for i in range(env.sim.model.njnt):
            if env.sim.model.jnt_bodyid[i] == object_id:
                joint_type = env.sim.model.jnt_type[i]
                joint_addr = env.sim.model.jnt_qposadr[i]
                
                if joint_type == 0:  # 自由关节 (free joint)
                    # 修改位置 (前3个值)
                    current_pos = env.sim.data.qpos[joint_addr:joint_addr+3].copy()
                    env.sim.data.qpos[joint_addr:joint_addr+3] = current_pos + position_delta
                    
                    # 修改方向 (接下来的4个值，四元数)
                    current_quat = env.sim.data.qpos[joint_addr+3:joint_addr+7].copy()
                    if np.any(rotation_delta != 0):
                        current_euler = quat_to_euler(current_quat)
                        new_euler = current_euler + rotation_delta
                        new_quat = euler_to_quat(new_euler)
                        env.sim.data.qpos[joint_addr+3:joint_addr+7] = new_quat
                    
                    joint_found = True
                    break
        
        # 方法2: 如果没有找到合适的关节，尝试使用mocap (如果物体是mocap控制的)
        if not joint_found:
            for i in range(env.sim.model.nmocap):
                mocap_id = env.sim.model.body_mocapid[object_id]
                if mocap_id != -1:  # -1表示不是mocap控制
                    # 修改mocap位置
                    current_pos = env.sim.data.mocap_pos[mocap_id].copy()
                    env.sim.data.mocap_pos[mocap_id] = current_pos + position_delta
                    
                    # 修改mocap方向
                    if np.any(rotation_delta != 0):
                        current_quat = env.sim.data.mocap_quat[mocap_id].copy()
                        current_euler = quat_to_euler(current_quat)
                        new_euler = current_euler + rotation_delta
                        new_quat = euler_to_quat(new_euler)
                        env.sim.data.mocap_quat[mocap_id] = new_quat
                    
                    joint_found = True
                    break
        
        # 方法3: 最后尝试-如果是静态物体，尝试修改body_pos/body_quat
        # 注意: 这通常不会影响仿真中的物体，除非重新加载环境或是特定环境
        if not joint_found:
            print(f"警告: 无法找到 '{obj_name}' 的可控制关节，尝试直接修改模型属性")
            # 尝试修改模型参数（可能需要重新加载才能生效）
            env.sim.model.body_pos[object_id] = env.sim.model.body_pos[object_id] + position_delta
            
            if np.any(rotation_delta != 0):
                current_quat = env.sim.model.body_quat[object_id].copy()
                current_euler = quat_to_euler(current_quat)
                new_euler = current_euler + rotation_delta
                new_quat = euler_to_quat(new_euler)
                env.sim.model.body_quat[object_id] = new_quat
    
    # 更新物理状态
    env.sim.forward()
    
    # 渲染确认
    camera_img = env.sim.render(
        width=IMAGE_RESOLUTION, 
        height=IMAGE_RESOLUTION, 
        camera_name=CAMERA_NAME  # 使用你环境中实际存在的相机名称
    )
    
    Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"object_observation_position.png"))


def change_object_color_and_texture(env, object_name):
    color = np.array([1.0, 0.0, 0.0])
    alpha = 0.05
    texture_name = None
    texture_rgb = None

    object_ids = []
    for i, name in enumerate(env.sim.model.body_names):
        if name and object_name in name:
            object_ids.append(i)
    
    if not object_ids:
        raise ValueError(f"找不到名称包含 '{object_name}' 的物体!")
    
    print(f"找到 {len(object_ids)} 个匹配的物体")

    if color is not None or alpha is not None:
        for object_id in object_ids:
            # 获取该物体的所有几何体
            for i in range(env.sim.model.ngeom):
                if env.sim.model.geom_bodyid[i] == object_id:
                    geom_id = i
                    current_rgba = env.sim.model.geom_rgba[geom_id].copy()
                
                    # 修改RGB部分
                    if color is not None:
                        current_rgba[:3] = color
                
                    # 修改透明度
                    if alpha is not None:
                        current_rgba[3] = alpha
                    
                    env.sim.model.geom_rgba[geom_id] = current_rgba

    if texture_name is not None or texture_rgb is not None:
        # 处理纹理
        texture_id = -1
        
        # 如果提供了纹理名称
        if texture_name:
            try:
                texture_id = env.sim.model.texture_name2id(texture_name)
            except:
                print(f"找不到名为 '{texture_name}' 的纹理")
        
        # 如果提供了RGB数据
        elif texture_rgb is not None and env.sim.model.ntex > 0:
            texture_id = 0  # 使用第一个纹理
            # 更新纹理数据
            height, width, _ = texture_rgb.shape
            env.sim.model.tex_height[texture_id] = height
            env.sim.model.tex_width[texture_id] = width
            env.sim.model.tex_rgb[texture_id] = texture_rgb
        
        # 应用纹理到物体
        if texture_id >= 0:
            for object_id in object_ids:
                # 获取该物体的所有几何体
                for i in range(env.sim.model.ngeom):
                    if env.sim.model.geom_bodyid[i] == object_id:
                        geom_id = i
                        # 直接设置几何体的纹理ID
                        env.sim.model.geom_texid[geom_id] = texture_id
    
    # 更新物理状态
    env.sim.forward()
    
    # 渲染结果
    img = env.sim.render(
        width=IMAGE_RESOLUTION, 
        height=IMAGE_RESOLUTION, 
        camera_name=CAMERA_NAME
    )
    
    Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"object_observation_alpha.png"))


def change_light(env):
    """
    修改 MuJoCo 环境中的光源属性
    
    参数:
        env: MuJoCo 环境
        light_name: 光源名称 (如果已命名)
        light_id: 光源ID (如果提供名称则忽略)
        position: 新的光源位置 [x, y, z]
        direction: 新的光源方向 [dx, dy, dz]
        color: 新的光源颜色 [r, g, b]
        specular: 镜面反射强度 (0-1)
        ambient: 环境光强度 (0-1)
        diffuse: 漫反射强度 (0-1)
        active: 是否激活光源 (True/False)
        
        如果想要增强光源的高光反射效果，调整 specular。
        如果想要改变场景的整体亮度，调整 ambient。
        如果想要控制光源的扩散强度，调整 diffuse
        
    """
    light_name = "light1"        # env.sim.model.light_names: 'light1', 'light2'
    light_id = None
    position = None
    direction = None
    color = None
    
    specular = 1.5      # default: [0.3, 0.3, 0.3]
    ambient = 1.5       # default: [0.0, 0.0, 0.0]
    diffuse = 1.5       # default: [0.8, 0.8, 0.8]
    
    active = None
    
    # 获取光源ID
    if light_id is None and light_name is not None:
        try:
            light_id = env.sim.model.light_name2id(light_name)
        except:
            print(f"找不到名为 '{light_name}' 的光源")
            
            # 尝试列出所有可用光源
            if env.sim.model.nlight > 0:
                print("可用光源:")
                for i in range(env.sim.model.nlight):
                    name = env.sim.model.light_id2name(i) if hasattr(env.sim.model, "light_id2name") else f"light_{i}"
                    print(f"  ID {i}: {name}")
            return False
    
    # 如果没有提供ID或名称，默认使用第一个光源
    if light_id is None:
        if env.sim.model.nlight > 0:
            light_id = 0
            print(f"使用默认光源 (ID: {light_id})")
        else:
            print("环境中没有光源")
            return False
    
    # 修改光源位置
    if position is not None:
        env.sim.model.light_pos[light_id] = position
        print(f"光源位置设置为 {position}")
    
    # 修改光源方向
    if direction is not None:
        env.sim.model.light_dir[light_id] = direction
        print(f"光源方向设置为 {direction}")
    
    # 修改光源颜色
    if color is not None:
        env.sim.model.light_specular[light_id] = color
        env.sim.model.light_diffuse[light_id] = color
        env.sim.model.light_ambient[light_id] = [c * 0.1 for c in color]  # 环境光通常较弱
        print(f"光源颜色设置为 {color}")
    
    # 单独修改各光照分量
    if specular is not None:
        if isinstance(specular, float):
            env.sim.model.light_specular[light_id] *= specular
        else:
            env.sim.model.light_specular[light_id] = specular
        print(f"镜面反射设置为 {specular}")
    
    if ambient is not None:
        if isinstance(ambient, float):
            env.sim.model.light_ambient[light_id] *= ambient
        else:
            env.sim.model.light_ambient[light_id] = ambient
        print(f"环境光设置为 {ambient}")
    
    if diffuse is not None:
        if isinstance(diffuse, float):
            env.sim.model.light_diffuse[light_id] *= diffuse
        else:
            env.sim.model.light_diffuse[light_id] = diffuse
        print(f"漫反射设置为 {diffuse}")
    
    # 开启/关闭光源
    if active is not None:
        env.sim.model.light_active[light_id] = 1 if active else 0
        print(f"光源已{'激活' if active else '关闭'}")
    
    # 更新物理状态
    env.sim.forward()
    
    # 渲染一张图像验证效果
    img = env.sim.render(
        width=IMAGE_RESOLUTION, 
        height=IMAGE_RESOLUTION, 
        camera_name=CAMERA_NAME
    )
    
    Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"light_observation_lighter.png"))


def main(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    
    task = task_suite.get_task(0)
    need_camera_change = False
    need_object_position_change = False
    need_object_color_change = False
    need_light_change = True
    
    camera_name = CAMERA_NAME
    if need_camera_change:
        camera_id = env.sim.model.camera_name2id(camera_name)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        env.reset()
        change_camera_viewpoint(env, camera_id, camera_name)

    if need_object_position_change:
        object_name = "plate_1"
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        env.reset()
        change_object_position_and_pose(env, object_name)

    if need_object_color_change:
        object_name = "plate_1"
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        env.reset()
        change_object_color_and_texture(env, object_name)

    if need_light_change:
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        env.reset()
        change_light(env)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO任务套件。例如: libero_spatial",
        required=False,
        default="libero_spatial",
    )
    args = parser.parse_args()

    # 启动程序
    main(args)