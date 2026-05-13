"""_summary_
API differences need to be noted when converting rotations between the simulator and the controller. For example,

MuJoCo API uses an [x, y, z, w] quaternion representation as shown here
Whereas, Drake uses the [w, x, y, z] ordering as described here

Returns:
    _type_: _description_
"""
import argparse
import json
import os
import math
import numpy as np
import time
import cv2  # 使用OpenCV处理键盘输入
from rich import print
from PIL import Image

import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
import transforms3d.axangles as axangles

from scipy.spatial.transform import Rotation as R

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import ControlEnv
import robosuite.utils.transform_utils as T  # 用于旋转变换


IMAGE_RESOLUTION = 512
CAMERA_NAME = "agentview"
IMAGE_SAVE_PATH = "LIBERO/xyg_scripts/image_rotation_frame3"
MUJOCO_WXYZ = True
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
    return euler.quat2euler(quat, "sxyz")


def euler_to_quat(euler_angles):
    return euler.euler2quat(euler_angles[0], euler_angles[1], euler_angles[2], "sxyz")


def recolor_robot_body(env, camera_id, camera_name, robot_base_name="robot0_base"):
    """_summary_
        recolor the robot body;
        translate the camera the check the world frame;
        translate the camera to robot base position and fade into it;
    """
    cur_camera_pos = env.sim.model.cam_pos[camera_id].copy()
    cur_camera_quat = env.sim.model.cam_quat[camera_id].copy()
    if not MUJOCO_WXYZ: # xyzw -> wxyz
        cur_camera_quat = np.array([cur_camera_quat[3], cur_camera_quat[0], cur_camera_quat[1], cur_camera_quat[2]])
    cur_camera_euler = quat_to_euler(cur_camera_quat)
    
    # robot_base_name = "robot0_link0"    # robot0_base -> gripper0_eef -> robot0_link0 -> robot0_link1 -> 
    # robot0_link2 -> robot0_link3 -> robot0_link4 -> robot0_link5 -> robot0_link6 -> robot0_link7
    total_body_names = [name for name in env.sim.model.body_names if name and name != "world"]
    robot_base_id = env.sim.model.body_name2id(robot_base_name)
    
    object_ids = [robot_base_id]
    color = np.array([1.0, 0.0, 0.0])
    for object_id in object_ids:
        # 获取该物体的所有几何体
        for i in range(env.sim.model.ngeom):
            if env.sim.model.geom_bodyid[i] == object_id:
                geom_id = i
                current_rgba = env.sim.model.geom_rgba[geom_id].copy()
            
                if color is not None:
                    current_rgba[:3] = color
                
                env.sim.model.geom_rgba[geom_id] = current_rgba

    total_body_names = [name for name in env.sim.model.body_names if name and name != "world"]
    # alpha_list = ['robot0_link0', 'robot0_link1', 'robot0_link2', 'robot0_link3', 'robot0_link4', 'robot0_link5', 'robot0_link6', 'robot0_link7']
    alpha_list = [x for x in total_body_names if 'robot0_' in x or 'gripper0_' in x]
    alpha_list = [x for x in alpha_list if x != robot_base_name]
    object_ids = [env.sim.model.body_name2id(x) for x in alpha_list]
    for object_id in object_ids:
        # 获取该物体的所有几何体
        for i in range(env.sim.model.ngeom):
            if env.sim.model.geom_bodyid[i] == object_id:
                geom_id = i
                current_rgba = env.sim.model.geom_rgba[geom_id].copy()
                current_rgba[3] = 0.05
                env.sim.model.geom_rgba[geom_id] = current_rgba
                
    robot_base_pos = env.sim.model.body_pos[robot_base_id].copy()
    robot_base_quat = env.sim.model.body_quat[robot_base_id].copy()
    print(f"cur_camera_pos: {cur_camera_pos}")
    print(f"cur_camera_quat: {cur_camera_quat}")
    print(f"robot_base_pos: {robot_base_pos}")
    print(f"robot_base_quat: {robot_base_quat}")
    
    for delta_x, delta_y, delta_z in zip([0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]):
        delta_camera_pos = np.array([delta_x, delta_y, delta_z])
        delta_camera_euler = np.array([0.0, 0.0, 0.0])
        
        for positive in [True, False]:
            if positive:    
                tgt_camera_pos = cur_camera_pos + delta_camera_pos
                tgt_camera_quat = euler_to_quat(cur_camera_euler + delta_camera_euler)
            else:
                tgt_camera_pos = cur_camera_pos - delta_camera_pos
                tgt_camera_quat = euler_to_quat(cur_camera_euler - delta_camera_euler)
            
            if not MUJOCO_WXYZ: # wxyz -> xyzw
                tgt_camera_quat = np.array([tgt_camera_quat[1], tgt_camera_quat[2], tgt_camera_quat[3], tgt_camera_quat[0]])
            
            # ToDo
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
            
            Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"{delta_x:.2f}_{delta_y:.2f}_{delta_z:.2f}_{positive}.png"))
    
    
    # camera -> robot_base, 插值, 0%, 20%, 40%, 60%, 80%, 100%
    for alpha in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        tgt_camera_pos = cur_camera_pos + alpha * (robot_base_pos - cur_camera_pos)
        tgt_camera_quat = cur_camera_quat.copy()
        
        if not MUJOCO_WXYZ: # wxyz -> xyzw
            tgt_camera_quat = np.array([tgt_camera_quat[1], tgt_camera_quat[2], tgt_camera_quat[3], tgt_camera_quat[0]])
            
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
        
        Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"camera_to_robot_base_{alpha:.2f}.png"))
    

class Pose:
    # must be wxyz quaterion format
    def __init__(self, position=None, orientation=None):
        if position is None:
            position = np.zeros(3)  # 默认位置为原点
        if orientation is None:
            orientation = np.array([1.0, 0, 0, 0])

        self.position = np.array(position)
        self.orientation = np.array(orientation)

    def get_position(self):
        return self.position

    def get_orientation(self):
        return self.orientation

    def set_position(self, position):
        self.position = np.array(position)

    def set_orientation(self, orientation):
        self.orientation = np.array(orientation)

    def transform(self, pose):
        """将当前Pose与另一个Pose相乘，返回组合后的Pose"""
        new_position = self.position + quaternions.quat2mat(self.orientation) @ pose.position
        new_orientation = quaternions.qmult(self.orientation, pose.orientation)
        return Pose(new_position, new_orientation)

    def orientation_inv(self):
        orientation = self.orientation
        orientation[1:] *= -1   # wxyz, 虚部求负数
        return orientation

    def inverse(self):
        """返回当前Pose的逆"""
        return Pose(-1 * quaternions.quat2mat(self.orientation_inv()) @ self.position, self.orientation_inv())


def rotate_camera_based_on_robot_base(cur_camera_pos, cur_camera_quat, robot_base_pos, robot_base_quat, theta, debug=False):
    """
        given wxyz quaterion format; camera pose rotate around robot base;
    """
    if debug:
        # import ipdb; ipdb.set_trace()
        # return cur_camera_pos + np.array([0.5, 0.0, 0.0]), cur_camera_quat
        return np.array([0.0, 0.0, 0.0]), cur_camera_quat

    # camera pose and robot base pose in world frame
    camera_world_pose = Pose(cur_camera_pos, cur_camera_quat)
    robot_world_pose = Pose(robot_base_pos, robot_base_quat)
    
    robot_robot_pose = Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]))
    
    world_to_robot_transform = robot_robot_pose.transform(robot_world_pose.inverse())
    
    # camera pose in robot frame
    camera_robot_pose = world_to_robot_transform.transform(camera_world_pose)
    
    # 沿着 z 轴旋转 theta 度
    new_camera_robot_pose = Pose(np.array([0, 0, 0]), euler.euler2quat(0, 0, theta / 180 * np.pi)).transform(camera_robot_pose)
    
    # camera pose in world frame
    new_camera_world_pose = world_to_robot_transform.inverse().transform(new_camera_robot_pose)
    
    return new_camera_world_pose.get_position(), new_camera_world_pose.get_orientation()
    

def check_robot_body_as_rotation_base(env, camera_id, camera_name, robot_base_name="robot0_base", theta=0.0, debug=False):
    cur_camera_pos = env.sim.model.cam_pos[camera_id].copy()
    cur_camera_quat = env.sim.model.cam_quat[camera_id].copy()
    
    if False:
        object_name = robot_base_name
        total_body_names = [name for name in env.sim.model.body_names if name and name != "world"]
        object_names = [name for name in total_body_names if object_name in name]
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
                        current_pos[1] = 0.0
                        env.sim.data.qpos[joint_addr:joint_addr+3] = current_pos
                        
                        joint_found = True
                        break
            
            # 方法2: 如果没有找到合适的关节，尝试使用mocap (如果物体是mocap控制的)
            if not joint_found:
                for i in range(env.sim.model.nmocap):
                    mocap_id = env.sim.model.body_mocapid[object_id]
                    if mocap_id != -1:  # -1表示不是mocap控制
                        # 修改mocap位置
                        current_pos = env.sim.data.mocap_pos[mocap_id].copy()
                        current_pos[1] = 0.0
                        env.sim.data.mocap_pos[mocap_id] = current_pos
                        
                        joint_found = True
                        break
            
            # 方法3: 最后尝试-如果是静态物体，尝试修改body_pos/body_quat
            # 注意: 这通常不会影响仿真中的物体，除非重新加载环境或是特定环境
            if not joint_found:
                print(f"警告: 无法找到 '{obj_name}' 的可控制关节，尝试直接修改模型属性")
                # 尝试修改模型参数（可能需要重新加载才能生效）
                current_pos = env.sim.model.body_pos[object_id].copy()
                current_pos[1] = 0.0
                env.sim.model.body_pos[object_id] = current_pos
    
    robot_base_id = env.sim.model.body_name2id(robot_base_name)
    robot_base_pos = env.sim.model.body_pos[robot_base_id].copy()
    robot_base_quat = env.sim.model.body_quat[robot_base_id].copy()
    
    import ipdb; ipdb.set_trace()
    if not MUJOCO_WXYZ: # xyzw -> wxyz
        cur_camera_quat = np.array([cur_camera_quat[3], cur_camera_quat[0], cur_camera_quat[1], cur_camera_quat[2]])
        robot_base_quat = np.array([robot_base_quat[3], robot_base_quat[0], robot_base_quat[1], robot_base_quat[2]])
    
    tgt_camera_pos, tgt_camera_quat = rotate_camera_based_on_robot_base(cur_camera_pos, cur_camera_quat, 
                                                                        robot_base_pos, robot_base_quat, theta, debug=debug)
    
    if not MUJOCO_WXYZ: # wxyz -> xyzw
        tgt_camera_quat = np.array([tgt_camera_quat[1], tgt_camera_quat[2], tgt_camera_quat[3], tgt_camera_quat[0]])
    
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
    
    if not debug:
        Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"rotate_{theta:.2f}.png"))
    else:
        Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"debug_{theta:.2f}.png"))


def main(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    task = task_suite.get_task(0)
    
    need_robot_body_as_rotation_base = False
    
    if need_robot_body_as_rotation_base:
        # robot_base_list = ['robot0_link0', 'robot0_link1', 'robot0_link2', 'robot0_link3', 'robot0_link4', 'robot0_link5', 'robot0_link6', 'robot0_link7']
        robot_base_list = ['robot0_link0']
        for robot_base_name in robot_base_list:
            env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
            env.reset()
            
            camera_name = CAMERA_NAME
            camera_id = env.sim.model.camera_name2id(camera_name)
            recolor_robot_body(env, camera_id, camera_name, robot_base_name)
      
    robot_base_name = 'robot0_link0'
    # robot_base_name = 'plate_1_main'
    camera_name = CAMERA_NAME
    theta_list = [-180, -150, -120, -90, -60, -30, -10, 0, 10, 30, 60, 90, 120, 150, 180]
    debug = False
    
    for theta in theta_list:
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        env.reset()
        camera_id = env.sim.model.camera_name2id(camera_name)
        check_robot_body_as_rotation_base(env, camera_id, camera_name, robot_base_name, theta=theta, debug=debug)
        
        if debug:
            break
    

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
    