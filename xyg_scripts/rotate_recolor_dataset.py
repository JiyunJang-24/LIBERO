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
import torch
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
import transforms3d.axangles as axangles

from scipy.spatial.transform import Rotation as R

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import ControlEnv
import robosuite.utils.transform_utils as T  # 用于旋转变换
from robosuite.utils import camera_utils as CU
from src.lerobot.datasets.visual_cue_utils import (
    remove_extrinsic_camera_axis_correction,
    _rescale_make_motion_basis_axis_rgb_tensor_cam_to_world,
    _get_motion_dynamics_basis,
    _make_motion_basis_axis_rgb_tensor_cam_to_world,
    _make_motion_basis_wrist_axis_rgb_tensor_cam_to_world,
    save_rgb_image,
)

IMAGE_RESOLUTION = 512
CAMERA_NAME = "agentview"
# IMAGE_SAVE_PATH = "LIBERO/xyg_scripts/image_transparency_example"
IMAGE_SAVE_PATH = "tmp_dir2"
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
    new_camera_robot_pose = Pose(np.array([0, 0, 0]), euler.euler2quat(0, 0, theta / 180 * np.pi)).transform(camera_robot_pose)

    # camera pose in world frame
    new_camera_world_pose = world_to_robot_transform.inverse().transform(new_camera_robot_pose)

    return new_camera_world_pose.get_position(), new_camera_world_pose.get_orientation()


def rotate_camera(env, camera_id, camera_name, robot_base_name="robot0_base", theta=0.0, reposition_camera_scale=1.0, debug=False):
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
    env = reposition_camera(env, camera_name, camera_id, robot_base_name, scale=reposition_camera_scale, debug=debug)
    if debug:
        camera_img = env.sim.render(
            camera_name=camera_name,  # 指定相机名称
            width=IMAGE_RESOLUTION,                  # 图像宽度
            height=IMAGE_RESOLUTION,                 # 图像高度
            depth=False,                # 是否需要深度图
            mode='offscreen'            # 离屏渲染模式
        )

        Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"rotate_{theta:.2f}.png"))
    return env

def reposition_camera(
        env,
        camera_name: str,
        camera_id: int,
        robot_base_name: str = "robot0_base",
        scale: float = 0.85,          # 0.85면 베이스 쪽으로 15% 가까워짐, 1.15면 15% 멀어짐
        debug: bool = False,
    ):
    # --- target(base) world pos ---
    base_id = env.sim.model.body_name2id(robot_base_name)
    base_pos_w = env.sim.data.xpos[base_id].copy()   # world

    # --- camera current world pos ---
    cam_bodyid = int(env.sim.model.cam_bodyid[camera_id])
    if cam_bodyid == -1:
        # camera is defined in world frame
        cam_pos_w = env.sim.model.cam_pos[camera_id].copy()
    else:
        # camera is defined in cam_body local frame -> convert local -> world
        body_pos_w = env.sim.data.xpos[cam_bodyid].copy()
        body_R_w = env.sim.data.xmat[cam_bodyid].reshape(3, 3).copy()
        cam_pos_local = env.sim.model.cam_pos[camera_id].copy()
        cam_pos_w = body_pos_w + body_R_w @ cam_pos_local

    # --- move along line: base + scale*(cam-base) ---
    v = cam_pos_w - base_pos_w
    new_cam_pos_w = base_pos_w + scale * v

    # --- write back to model.cam_pos (world or local depending on cam_bodyid) ---
    if cam_bodyid == -1:
        env.sim.model.cam_pos[camera_id] = new_cam_pos_w
    else:
        body_pos_w = env.sim.data.xpos[cam_bodyid].copy()
        body_R_w = env.sim.data.xmat[cam_bodyid].reshape(3, 3).copy()
        new_cam_pos_local = body_R_w.T @ (new_cam_pos_w - body_pos_w)
        env.sim.model.cam_pos[camera_id] = new_cam_pos_local

    env.sim.forward()
    if debug:
        camera_img = env.sim.render(
            camera_name=camera_name,  # 指定相机名称
            width=IMAGE_RESOLUTION,                  # 图像宽度
            height=IMAGE_RESOLUTION,                 # 图像高度
            depth=False,                # 是否需要深度图
            mode='offscreen'            # 离屏渲染模式
        )

        Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"scale_{scale:.2f}.png"))
    return env

def rotate_camera_ur5e(env, ur5e_env, camera_id, camera_name, robot_base_name="robot0_base", theta=0.0, reposition_camera_scale=1.0, debug=False):
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

    ur5e_env.sim.model.cam_pos[camera_id] = tgt_camera_pos
    ur5e_env.sim.model.cam_quat[camera_id] = tgt_camera_quat
    ur5e_env.sim.forward()

    env.sim.model.cam_pos[camera_id] = tgt_camera_pos
    env.sim.model.cam_quat[camera_id] = tgt_camera_quat
    env.sim.forward()

    env = reposition_camera(env, camera_name, camera_id, robot_base_name, scale=reposition_camera_scale, debug=debug)
    ur5e_env.sim_model.cam_pos[camera_id] = env.sim.model.cam_pos[camera_id].copy()
    ur5e_env.sim.model.cam_quat[camera_id] = env.sim.model.cam_quat[camera_id].copy()
    ur5e_env.sim.forward()

    if debug:
        camera_img = ur5e_env.sim.render(
            camera_name=camera_name,  # 指定相机名称
            width=IMAGE_RESOLUTION,                  # 图像宽度
            height=IMAGE_RESOLUTION,                 # 图像高度
            depth=False,                # 是否需要深度图
            mode='offscreen'            # 离屏渲染模式
        )

        Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"rotate_{theta:.2f}.png"))
    return ur5e_env


def color_interpolation(color_a, color_b, alpha):
    return color_a * alpha + color_b * (1 - alpha)


def change_env_light(env, light_id, color, specular, ambient, diffuse, active):
    if light_id is None:
        if env.sim.model.nlight > 0:
            light_id = 0
            print(f"使用默认光源 (ID: {light_id})")
        else:
            print("环境中没有光源")
            return False

    # 修改光源颜色
    if color is not None:
        env.sim.model.light_specular[light_id] = color
        env.sim.model.light_diffuse[light_id] = color
        env.sim.model.light_ambient[light_id] = [c * 0.1 for c in color]  # 环境光通常较弱
        # print(f"光源颜色设置为 {color}")

    # 单独修改各光照分量
    if specular is not None:
        if isinstance(specular, float):
            env.sim.model.light_specular[light_id] *= specular
        else:
            env.sim.model.light_specular[light_id] = specular
        # print(f"镜面反射设置为 {specular}")

    if ambient is not None:
        if isinstance(ambient, float):
            env.sim.model.light_ambient[light_id] *= ambient
        else:
            env.sim.model.light_ambient[light_id] = ambient
        # print(f"环境光设置为 {ambient}")

    if diffuse is not None:
        if isinstance(diffuse, float):
            env.sim.model.light_diffuse[light_id] *= diffuse
        else:
            env.sim.model.light_diffuse[light_id] = diffuse
        # print(f"漫反射设置为 {diffuse}")

    # 开启/关闭光源
    if active is not None:
        env.sim.model.light_active[light_id] = 1 if active else 0
        # print(f"光源已{'激活' if active else '关闭'}")


def recolor_scene(env, alpha, color_light_a, color_light_b, need_print_all_light=False, debug=False, need_change_light=False, base_num=0.1):
    light_name_list = ["light1", "light2"]
    light_id_list = [env.sim.model.light_name2id(light_name) for light_name in light_name_list]

    light_id = None
    color = color_interpolation(color_light_a, color_light_b, alpha)


    if need_change_light:
        factor = 1.0
        specular, ambient, diffuse = base_num + alpha * factor, base_num + alpha * factor, base_num + alpha * factor
        active = None
    else:
        specular, ambient, diffuse, active = None, None, None, None

    # 获取光源ID
    if need_print_all_light and env.sim.model.nlight > 0:
        print("可用光源:")
        for i in range(env.sim.model.nlight):
            name = env.sim.model.light_id2name(i) if hasattr(env.sim.model, "light_id2name") else f"light_{i}"
            print(f"  ID {i}: {name}")

    # 如果没有提供ID或名称，默认使用第一个光源
    for light_id in light_id_list:
        change_env_light(env, light_id, color, specular, ambient, diffuse, active)

    # 更新物理状态
    env.sim.forward()

    # 渲染一张图像验证效果
    if debug:
        img = env.sim.render(
            width=IMAGE_RESOLUTION,
            height=IMAGE_RESOLUTION,
            camera_name=CAMERA_NAME
        )

        Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"basenum_{base_num:.2f}_color_{alpha:.2f}_light_{need_change_light}.png"))

    return env


def recolor_and_rotate_scene(env, alpha, color_light_a, color_light_b, camera_id,
                             camera_name, robot_base_name, theta, debug=True, need_change_light=False, base_num=0.5):
    env = recolor_scene(env, alpha, color_light_a, color_light_b, need_change_light=need_change_light, base_num=base_num)
    env = rotate_camera(env, camera_id, camera_name, robot_base_name, theta)

    if debug:
        env.sim.forward()
        img = env.sim.render(
            width=IMAGE_RESOLUTION,
            height=IMAGE_RESOLUTION,
            camera_name=CAMERA_NAME
        )
        Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"light_{alpha:.2f}_rotate_{theta:.2f}.png"))
    return env


def change_object_transparency(env, object_name, alpha=1.0, debug=False):
    object_ids = []
    for i, name in enumerate(env.envs[0]._env.sim.model.body_names):
        if name and object_name in name:
            object_ids.append(i)

    if not object_ids:
        raise ValueError(f"找不到名称包含 '{object_name}' 的物体!")

    if alpha is not None:
        for object_id in object_ids:
            # 获取该物体的所有几何体
            for i in range(env.envs[0]._env.sim.model.ngeom):
                if env.envs[0]._env.sim.model.geom_bodyid[i] == object_id:
                    geom_id = i
                    current_rgba = env.envs[0]._env.sim.model.geom_rgba[geom_id].copy()
                    # 修改透明度
                    if alpha is not None:
                        current_rgba[3] = alpha

                    env.envs[0]._env.sim.model.geom_rgba[geom_id] = current_rgba

    env.envs[0]._env.sim.forward()

    img = env.envs[0]._env.sim.render(
        width=IMAGE_RESOLUTION,
        height=IMAGE_RESOLUTION,
        camera_name=CAMERA_NAME
    )

    if debug:
        Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"object_observation_transparency_{alpha:.2f}.png"))
        return env, img

    return env


def rotate_object(env, object_name, theta_deg, debug):
    # theta_deg: degrees to rotate around the vertical (Z) axis
    rad = np.deg2rad(theta_deg)

    # 1. Create a rotation quaternion for Z-axis (wxyz format)
    # [cos(a/2), 0, 0, sin(a/2)]
    rot_quat = np.array([np.cos(rad/2), 0, 0, np.sin(rad/2)])

    # 2. Find the body IDs (using your existing logic)
    object_ids = [i for i, name in enumerate(env.sim.model.body_names) if name and object_name in name]
    for body_id in object_ids:
        # --- A. Update the Model Pose (Static/Initialization) ---
        # This affects the "rest" pose in the model
        current_model_quat = env.sim.model.body_quat[body_id].copy()
        new_model_quat = quaternions.qmult(rot_quat, current_model_quat)
        env.sim.model.body_quat[body_id] = new_model_quat

        # --- B. Update the Simulation Data (Dynamic State) ---
        # If the object has a "free joint" (movable), you must update the qpos,
        # otherwise the physics engine will snap it back to its old pose.
        if env.sim.model.body_jntnum[body_id] > 0:
            jnt_adr = env.sim.model.body_jntadr[body_id]
            # Check if it's a free joint (mjtJoint.mjJNT_FREE = 0)
            if env.sim.model.jnt_type[jnt_adr] == 0:
                qpos_adr = env.sim.model.jnt_qposadr[jnt_adr]
                # qpos for free joint is 7 elements: [x, y, z, qw, qx, qy, qz]
                current_data_quat = env.sim.data.qpos[qpos_adr+3 : qpos_adr+7].copy()
                new_data_quat = quaternions.qmult(rot_quat, current_data_quat)
                env.sim.data.qpos[qpos_adr+3 : qpos_adr+7] = new_data_quat
    # 3. Synchronize the physics state
    env.sim.forward()
    img = env.sim.render(
        width=IMAGE_RESOLUTION,
        height=IMAGE_RESOLUTION,
        camera_name=CAMERA_NAME
    )

    if debug:
        Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"rotate_{object_name}_{theta_deg:.2f}.png"))
        return env, img
    return env

import numpy as np

def _get_mujoco_sim_holder(env):
    """
    Returns an object that has `.sim` (mujoco sim).
    Handles cases like:
      - env.sim exists
      - vector env: env.envs[0] is LiberoEnv and may have ._env.sim
      - vector env: env.envs[0] itself has .sim
    """
    if hasattr(env, "sim"):
        return env

    if hasattr(env, "envs") and len(env.envs) > 0:
        e0 = env.envs[0]
        if hasattr(e0, "sim"):
            return e0
        if hasattr(e0, "_env") and hasattr(e0._env, "sim"):
            return e0._env

    raise AttributeError("Cannot find mujoco sim holder. Expected env.sim or env.envs[0].sim / env.envs[0]._env.sim")


def _body_depth(model, body_id):
    """Compute depth from worldbody (0). Smaller => closer to root."""
    depth = 0
    cur = body_id
    while cur != 0:
        cur = int(model.body_parentid[cur])
        depth += 1
        if depth > 10_000:
            break
    return depth


def reposition_object(env, object_name, delta_xyz, move_all=False, debug=False):
    """
    Move an object by a relative translation (dx, dy, dz) in WORLD frame.
    Returns (env, delta_xyz, abs_xyz) where abs_xyz is the resulting world position
    of the representative body (or dict of positions if move_all=True).
    """
    sim_env = _get_mujoco_sim_holder(env)
    sim = sim_env.sim
    model, data = sim.model, sim.data

    prev_img = sim.render(
        width=IMAGE_RESOLUTION,
        height=IMAGE_RESOLUTION,
        camera_name=CAMERA_NAME
    )

    delta_w = np.asarray(delta_xyz, dtype=np.float64).reshape(3)

    # Make sure xmat/xpos are up-to-date for parent-frame conversion
    sim.forward()

    # Find candidate bodies (substring match)
    body_ids = [i for i, name in enumerate(model.body_names) if name and (object_name in name)]
    if not body_ids:
        raise ValueError(f"Cannot find any body whose name contains '{object_name}'")

    # Choose representative body (safer default)
    if not move_all:
        exact = [i for i in body_ids if model.body_names[i] == object_name]
        if exact:
            body_ids = [exact[0]]
        else:
            body_ids = [min(body_ids, key=lambda bid: _body_depth(model, bid))]

    # --- apply translation ---
    for body_id in body_ids:
        parent_id = int(model.body_parentid[body_id])

        if parent_id == 0:
            delta_parent = delta_w
        else:
            parent_R_w = data.xmat[parent_id].reshape(3, 3).copy()   # parent->world
            delta_parent = parent_R_w.T @ delta_w                    # world->parent

        if model.body_jntnum[body_id] > 0:
            jnt_adr = int(model.body_jntadr[body_id])
            if int(model.jnt_type[jnt_adr]) == 0:  # mjJNT_FREE
                qpos_adr = int(model.jnt_qposadr[jnt_adr])
                data.qpos[qpos_adr:qpos_adr+3] += delta_w
                model.body_pos[body_id] += delta_parent
            else:
                model.body_pos[body_id] += delta_parent
        else:
            model.body_pos[body_id] += delta_parent

    sim.forward()

    # --- absolute position(s) in WORLD frame ---
    if move_all:
        abs_xyz = {model.body_names[bid]: data.xpos[bid].copy() for bid in body_ids}
    else:
        rep_id = body_ids[0]
        abs_xyz = data.xpos[rep_id].copy()  # (3,) world xyz

    img = sim.render(
        width=IMAGE_RESOLUTION,
        height=IMAGE_RESOLUTION,
        camera_name=CAMERA_NAME
    )

    if debug:
        Image.fromarray(prev_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, "prev_object_reposition.png"))
        Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, "after_object_reposition.png"))
    return env, delta_xyz, tuple(abs_xyz)


def _find_body_id_by_substring(model, substring: str) -> int:
    body_ids = [i for i, name in enumerate(model.body_names) if name and substring in name]
    if not body_ids:
        raise ValueError(f"Cannot find any body name containing '{substring}'")
    # 보통 object root body가 더 상위에 있으니 depth가 가장 작은 걸 대표로 선택
    return min(body_ids, key=lambda bid: _body_depth(model, bid))

def _find_eef_site_id(model, preferred_site_name: str | None = None) -> int:
    # 1) 사용자가 site 이름을 줬으면 그걸 우선
    if preferred_site_name is not None:
        try:
            return model.site_name2id(preferred_site_name)
        except Exception:
            pass

    # 2) 흔한 eef/gripper site name들
    common = [
        "robot0_grip_site",
        "robot0_eef_site",
        "eef_site",
        "grip_site",
        "gripper_site",
    ]
    for name in common:
        try:
            return model.site_name2id(name)
        except Exception:
            continue

    # 3) 휴리스틱: site 이름에 grip/eef/end 등이 들어간 첫 번째
    for i, name in enumerate(model.site_names):
        if not name:
            continue
        low = name.lower()
        if ("grip" in low) or ("eef" in low) or ("end" in low):
            return i

    raise ValueError(
        "Cannot find an end-effector site. "
        "Please pass eef_site_name explicitly (e.g., eef_site_name='robot0_grip_site')."
    )

def robot_object_distance_xyz(
    env,
    object_name: str,
    eef_site_name: str | None = None,
):
    """
    Returns:
      eef_pos (3,), obj_pos (3,), diff (3,) where diff = obj - eef,
      dist_xyz (3,) = |diff|, dist (float) = L2 norm
    """
    sim_env = _get_mujoco_sim_holder(env)
    sim = sim_env.sim
    model, data = sim.model, sim.data
    sim.forward()
    # EEF position (world)
    eef_site_id = _find_eef_site_id(model, eef_site_name)
    eef_pos = data.site_xpos[eef_site_id].copy()

    # Object position (world) - 대표 body의 xpos
    obj_body_id = _find_body_id_by_substring(model, object_name)
    obj_pos = data.xpos[obj_body_id].copy()

    diff = obj_pos - eef_pos              # (dx, dy, dz)
    dist_xyz = np.abs(diff)               # 축별 거리
    dist = float(np.linalg.norm(diff))    # 3D 거리

    return {
        "eef_pos": eef_pos,
        "obj_pos": obj_pos,
        "diff_xyz": diff,     # obj - eef
        "dist_xyz": dist_xyz, # |dx|,|dy|,|dz|
        "dist": dist,         # sqrt(dx^2+dy^2+dz^2)
        "eef_site_name": model.site_names[eef_site_id],
        "obj_body_name": model.body_names[obj_body_id],
    }

def finger_object_in_contact(env, object_name: str, collision_only: bool = True) -> bool:
    """
    Returns True if any contact exists between gripper fingers and the object.

    - Fingers: geoms whose names start with 'gripper0_finger1' or 'gripper0_finger2'
    - Object: geoms attached to bodies whose body name contains `object_name`,
              fallback to geom name substring match if no body match.
    - collision_only: if True, only consider finger geoms containing 'collision'
                      (recommended, since contacts happen on collision geoms).

    Args:
        env: mujoco env (may be wrapped/vectorized)
        object_name: substring for object body/geom names (e.g., "mug", "cube")
        collision_only: whether to restrict finger geoms to collision geoms

    Returns:
        bool
    """
    sim_env = _get_mujoco_sim_holder(env)
    sim = sim_env.sim
    model, data = sim.model, sim.data

    sim.forward()

    # ---- collect object geom ids ----
    body_ids = [i for i, name in enumerate(model.body_names) if name and (object_name in name)]
    obj_geom_ids = set()
    if body_ids:
        body_ids = set(body_ids)
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) in body_ids:
                obj_geom_ids.add(gid)

    # fallback: match geom names
    if not obj_geom_ids:
        for gid, name in enumerate(model.geom_names):
            if name and (object_name in name):
                obj_geom_ids.add(gid)

    if not obj_geom_ids:
        raise ValueError(f"Cannot find object geoms for '{object_name}' (no matching body/geom names).")

    # ---- collect finger geom ids ----
    finger_prefixes = ("gripper0_finger1", "gripper0_finger2")
    finger_geom_ids = set()
    for gid, name in enumerate(model.geom_names):
        if not name:
            continue
        if name.startswith(finger_prefixes):
            if collision_only and ("collision" not in name):
                continue
            finger_geom_ids.add(gid)

    if not finger_geom_ids:
        raise ValueError("No finger geoms found (check naming / collision_only flag).")

    # ---- scan contacts ----
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in finger_geom_ids and g2 in obj_geom_ids) or (g2 in finger_geom_ids and g1 in obj_geom_ids):
            return True

    return False

def get_visual_cues(env, height, width, overlay=False):
    sim_env = _get_mujoco_sim_holder(env)
    sim = sim_env.sim
    model, data = sim.model, sim.data
    sim.forward()
    item = {}
    item['intrinsic_matrix'] = _to_torch(CU.get_camera_intrinsic_matrix(sim, 'agentview', height, width))
    item['extrinsic_matrix'] = _to_torch(CU.get_camera_extrinsic_matrix(sim, 'agentview'))
    item['observation.image'] = _to_torch(np.flipud(sim.render(width=width, height=height, camera_name='agentview')).copy().transpose(2, 0, 1)).unsqueeze(0)
    item['observation.wrist_image'] = _to_torch(np.flipud(sim.render(width=width, height=height, camera_name='robot0_eye_in_hand')).copy().transpose(2, 0, 1)).unsqueeze(0)
    item["wrist_intrinsic_matrix"] = _to_torch(CU.get_camera_intrinsic_matrix(sim, 'robot0_eye_in_hand', height, width))
    item["wrist_extrinsic_matrix"] = _to_torch(CU.get_camera_extrinsic_matrix(sim, 'robot0_eye_in_hand'))
    eef_site_id = _find_eef_site_id(model, "gripper0_eef")
    eef_pos = _to_torch(np.concatenate((data.site_xpos[eef_site_id].copy().astype(np.float32), np.zeros(4, dtype=np.float32)))).unsqueeze(0)
    item['observation.state'] = eef_pos
    #get extrinsic_matrix, intrinsic_matrix, state, image, wrist_image, 
    extrinsic_matrix = item['extrinsic_matrix']
    extrinsic_matrix = remove_extrinsic_camera_axis_correction(extrinsic_matrix)
    import pdb; pdb.set_trace()
    intrinsic_matrix = item['intrinsic_matrix']
    robot_state = item['observation.state'] #gripper qpos (2), eef pos (3), eef quat (4)
    img = item['observation.image'] # S * C * H * W

    motion_dynamics_basis = _get_motion_dynamics_basis(intrinsic_matrix, cam_to_world=extrinsic_matrix).reshape(-1)
    agentview_axis, _ = _make_motion_basis_axis_rgb_tensor_cam_to_world(
        rgb_tensor=img,                  # (B, 3,H,W)
        motion_dynamics_basis=motion_dynamics_basis,
        cam_to_world=extrinsic_matrix,                  # cam_pose = cam_to_world (고정)
        intrinsic_matrix=intrinsic_matrix,
        robot_eef_abs_poses=robot_state[:, -7:],  # eef pose (B, 7)
        origin_robot=True,
        origin_fallback="pp",
        arrow_len=60,
        return_overlay=overlay,
    ) # (B, 3, H, W)
    save_rgb_image(agentview_axis, "tmp_dir/axis_tensor.png")
    wrist_img = item['observation.wrist_image']
    wrist_intrinsic_matrix = item['wrist_intrinsic_matrix']
    wrist_extrinsic_matrix = item['wrist_extrinsic_matrix']
    wrist_plucker_extrinsic_matrix = remove_extrinsic_camera_axis_correction(wrist_extrinsic_matrix)
    wrist_view_axis, _ = _make_motion_basis_wrist_axis_rgb_tensor_cam_to_world(
        rgb_tensor=wrist_img,
        cam_to_world=wrist_plucker_extrinsic_matrix,
        intrinsic_matrix=wrist_intrinsic_matrix,
        robot_eef_abs_poses=robot_state[:, -7:],
        origin_robot=True,
        origin_fallback="pp",
        arrow_len=60,
        return_overlay=overlay,
    )
    save_rgb_image(wrist_view_axis[0], "tmp_dir/wrist_non_scaled_axis_tensor.png")
    # save_rgb_image(agentview_axis[0], "tmp_dir/axis_tensor.png")
        # save_rgb_image(item['observation.image'][0], "tmp_dir/robot_image.png")

    return agentview_axis, wrist_view_axis


def get_visual_cues_image(env, camera_name, image, state, overlay=False):
    sim_env = _get_mujoco_sim_holder(env)
    sim = sim_env.sim
    model, data = sim.model, sim.data
    sim.forward()
    item = {}
    channel, height, width = image.shape
    intrinsic_matrix = _to_torch(CU.get_camera_intrinsic_matrix(sim, camera_name, height, width))
    extrinsic_matrix = _to_torch(CU.get_camera_extrinsic_matrix(sim, camera_name))
    #get extrinsic_matrix, intrinsic_matrix, state, image, wrist_image, 
    extrinsic_matrix = remove_extrinsic_camera_axis_correction(extrinsic_matrix)
    import pdb; pdb.set_trace()

    if camera_name == "agentview":
        motion_dynamics_basis = _get_motion_dynamics_basis(intrinsic_matrix, cam_to_world=extrinsic_matrix).reshape(-1)
        axis_tensor, _ = _make_motion_basis_axis_rgb_tensor_cam_to_world(
            rgb_tensor=image,                  # (B, 3,H,W)
            motion_dynamics_basis=motion_dynamics_basis,
            cam_to_world=extrinsic_matrix,                  # cam_pose = cam_to_world (고정)
            intrinsic_matrix=intrinsic_matrix,
            robot_eef_abs_poses=state[:, -7:],  # eef pose (B, 7)
            origin_robot=True,
            origin_fallback="pp",
            arrow_len=60,
            return_overlay=overlay,
        ) # (B, 3, H, W)
        save_rgb_image(axis_tensor[0], "tmp_dir/axis_tensor.png")
    elif camera_name == "robot0_eye_in_hand":
        axis_tensor, _ = _make_motion_basis_wrist_axis_rgb_tensor_cam_to_world(
            rgb_tensor=image,
            cam_to_world=extrinsic_matrix,
            intrinsic_matrix=intrinsic_matrix,
            robot_eef_abs_poses=state[:, -7:],
            origin_robot=True,
            origin_fallback="pp",
            arrow_len=60,
            return_overlay=overlay,
        )
        save_rgb_image(axis_tensor[0], "tmp_dir/wrist_non_scaled_axis_tensor.png")
    # save_rgb_image(agentview_axis[0], "tmp_dir/axis_tensor.png")
        # save_rgb_image(item['observation.image'][0], "tmp_dir/robot_image.png")

    return axis_tensor

def _to_torch(x, dtype=torch.float32, device="cpu"):
    # numpy -> torch
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    # torch -> torch
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    # python scalar/list -> torch
    return torch.tensor(x, device=device, dtype=dtype)


def main(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    task = task_suite.get_task(0)

    robot_base_name = 'robot0_link0'
    camera_name = CAMERA_NAME
    theta_list = [-180, -150, -120, -90, -60, -30, -10, 0, 10, 30, 60, 90, 120, 150, 180]
    color_light_a = np.array([1.0, 0.0, 0.0])
    color_light_b = np.array([1.0, 1.0, 0.0])
    alpha_list = np.linspace(0, 1, 5)

    theta_list = [-10, 0, 10]
    alpha_list = np.linspace(0, 1, 3)

    object_name = "akita_black_bowl_2"
    transparency_list = np.linspace(0, 1, 5)

    # rotate the camera around the robot base;
    need_rotate_camera = False
    if need_rotate_camera:
        for theta in theta_list:
            env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
            env.reset()
            camera_id = env.sim.model.camera_name2id(camera_name)
            env = rotate_camera(env, camera_id, camera_name, robot_base_name, theta=theta)

    # recolor the scene;
    need_recolor_scene = True
    need_change_light = True
    if need_recolor_scene:
        for base_num in np.linspace(0.05, 0.95, 8):
            for transparency in alpha_list:
                env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
                env.reset()
                env = recolor_scene(env, alpha=transparency, color_light_a=color_light_a, color_light_b=color_light_b, debug=True,
                                    need_change_light=need_change_light, base_num=base_num)

    need_recolor_and_rotate = False
    if need_recolor_and_rotate:
        for transparency, theta in product(alpha_list, theta_list):
            env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
            env.reset()
            camera_id = env.sim.model.camera_name2id(camera_name)
            env = recolor_and_rotate_scene(env, alpha=transparency, color_light_a=color_light_a, color_light_b=color_light_b,
                                     camera_id=camera_id, camera_name=camera_name, robot_base_name=robot_base_name,
                                     theta=theta, debug=True)

    # 将场景中的部分无色设置为透明，不可见
    need_change_object_transparency = False
    if need_change_object_transparency:
        for transparency in transparency_list:
            env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
            env.reset()
            env = change_object_transparency(env, object_name=object_name, alpha=transparency, debug=True)


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
