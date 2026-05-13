"""_summary_
API differences need to be noted when converting rotations between the simulator and the controller. For example,

MuJoCo API uses an [x, y, z, w] quaternion representation as shown here
Whereas, Drake uses the [w, x, y, z] ordering as described here

Returns:
    _type_: _description_

现在这个代码不能动了，因为我后面会调用它修改环境
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
from itertools import product

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
# IMAGE_SAVE_PATH = "LIBERO/xyg_scripts/image_transparency_example"
IMAGE_SAVE_PATH = "tmp_dir3"
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


def rotate_camera_based_on_robot_base(cur_camera_pos, cur_camera_quat, robot_base_pos, robot_base_quat, theta):
    """
        given wxyz quaterion format; camera pose rotate around robot base;
    """
    # camera pose and robot base pose in world frame
    camera_world_pose = Pose(cur_camera_pos, cur_camera_quat)
    robot_world_pose = Pose(robot_base_pos, robot_base_quat)
    
    robot_robot_pose = Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]))
    
    world_to_robot_transform = robot_robot_pose.transform(robot_world_pose.inverse())
    
    # camera pose in robot frame
    camera_robot_pose = world_to_robot_transform.transform(camera_world_pose)
    
    # 沿着 z 轴旋转 theta 度
    if isinstance(theta, int) or isinstance(theta, float):
        new_camera_robot_pose = Pose(np.array([0, 0, 0]), euler.euler2quat(0, 0, theta / 180 * np.pi)).transform(camera_robot_pose)
    else:
        assert (isinstance(theta, list) or isinstance(theta, tuple)) and len(theta) == 3
        new_camera_robot_pose = Pose(
            np.array([0, 0, 0]), euler.euler2quat(theta[0] / 180 * np.pi, theta[1] / 180 * np.pi, theta[2] / 180 * np.pi)
        ).transform(camera_robot_pose)
    
    # camera pose in world frame
    new_camera_world_pose = world_to_robot_transform.inverse().transform(new_camera_robot_pose)
    
    return new_camera_world_pose.get_position(), new_camera_world_pose.get_orientation()
    

def rotate_camera(env, camera_id, camera_name, robot_base_name="robot0_base", theta=0.0, debug=False):
    cur_camera_pos = env.sim.model.cam_pos[camera_id].copy()
    cur_camera_quat = env.sim.model.cam_quat[camera_id].copy()
    
    robot_base_id = env.sim.model.body_name2id(robot_base_name)
    robot_base_pos = env.sim.model.body_pos[robot_base_id].copy()
    robot_base_quat = env.sim.model.body_quat[robot_base_id].copy()
    
    if not MUJOCO_WXYZ: # xyzw -> wxyz
        cur_camera_quat = np.array([cur_camera_quat[3], cur_camera_quat[0], cur_camera_quat[1], cur_camera_quat[2]])
        robot_base_quat = np.array([robot_base_quat[3], robot_base_quat[0], robot_base_quat[1], robot_base_quat[2]])
    
    tgt_camera_pos, tgt_camera_quat = rotate_camera_based_on_robot_base(cur_camera_pos, cur_camera_quat, 
                                                                        robot_base_pos, robot_base_quat, theta)
    
    if not MUJOCO_WXYZ: # wxyz -> xyzw
        tgt_camera_quat = np.array([tgt_camera_quat[1], tgt_camera_quat[2], tgt_camera_quat[3], tgt_camera_quat[0]])
    
    env.sim.model.cam_pos[camera_id] = tgt_camera_pos
    env.sim.model.cam_quat[camera_id] = tgt_camera_quat
    env.sim.forward()
    
    if debug:
        camera_img = env.sim.render(
            camera_name=camera_name,  # 指定相机名称
            width=IMAGE_RESOLUTION,                  # 图像宽度
            height=IMAGE_RESOLUTION,                 # 图像高度
            depth=False,                # 是否需要深度图
            mode='offscreen'            # 离屏渲染模式
        )
        
        if isinstance(theta, list) or isinstance(theta, tuple):
            Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"rotate_{theta[0]:.2f}_{theta[1]:.2f}_{theta[2]:.2f}.png"))
        else:
            Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"rotate_{0:.2f}_{0:.2f}_{theta:.2f}.png"))
    return env

    
def main(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    task = task_suite.get_task(0)
    
    robot_base_name = 'robot0_link0'
    camera_name = CAMERA_NAME
    # theta_list = [-180, -150, -120, -90, -60, -30, -10, 0, 10, 30, 60, 90, 120, 150, 180]
    
    # theta_list = [30, 50, (8, 0, 0), (-10, 0, 0), (0, -10, 0), (0, 10, 0)]
    # theta_list = [30, 50, ]
    theta_list = [-180, -135, -90, -45, 0, 45, 90, 135, 180, ]
    for x in [-10, 0, 8]:
        for y in [-10, 0, 10]:
            for z in [-10, 0, 10, 30, 50]:
                theta_list.append((x, y, z))
    
    # rotate the camera around the robot base;
    need_rotate_camera = True
    if need_rotate_camera:
        for theta in theta_list:
            env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
            env.reset()
            camera_id = env.sim.model.camera_name2id(camera_name)
            env = rotate_camera(env, camera_id, camera_name, robot_base_name, theta=theta, debug=True)
            

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