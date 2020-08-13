import gibson2
from gibson2.core.physics.interactive_objects import VisualMarker, InteractiveObj, BoxShape
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.utils.utils import parse_config, rotate_vector_3d, l2_distance, quatToXYZW, cartesian_to_polar
from gibson2.envs.base_env import BaseEnv
from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
from transforms3d.quaternions import quat2mat, qmult
import gym
import numpy as np
import os
import pybullet as p
from IPython import embed
import cv2
import time
import collections
import logging

'''
CoRL: added classes NavigateRandomInitEnv and NavigateRandomInitEnvSim2Real modelled after iGibson's NavigateRandomEnv
and NavigateInitEnvSim2Real
Original Authors:
F. Xia, W. B. Shen, C. Li, P. Kasimbeg, M. E. Tchapmi, A. Toshev, R. Martı́n-Martı́n, and
S. Savarese. Interactive Gibson benchmark: A benchmark for interactive navigation in clut-
tered environments. IEEE Robotics and Automation Letters, 5(2):713–720, 2020.
'''

class NavigateEnv(BaseEnv):
    """
    We define navigation environments following Anderson, Peter, et al. 'On evaluation of embodied navigation agents.'
    arXiv preprint arXiv:1807.06757 (2018). (https://arxiv.org/pdf/1807.06757.pdf)
    """
    def __init__(
            self,
            config_file,
            model_id=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            device_idx=0,
            render_to_tensor=False
    ):
        """
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        super(NavigateEnv, self).__init__(config_file=config_file,
                                          model_id=model_id,
                                          mode=mode,
                                          action_timestep=action_timestep,
                                          physics_timestep=physics_timestep,
                                          device_idx=device_idx,
                                          render_to_tensor=render_to_tensor
                                          )
        self.automatic_reset = automatic_reset

    def load_task_setup(self):
        """
        Load task setup, including initialization, termination conditino, reward, collision checking, discount factor
        """
        # initial and target pose
        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_pos_cp = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))
        self.initial_orn_cp = np.array(self.config.get('initial_orn', [0, 0, 0]))
        self.random_init_x_range = np.array(self.config.get('random_init_x_range', [-0.2, 0.2]))
        self.random_init_y_range = np.array(self.config.get('random_init_y_range', [-0.3, 0.3]))
        self.random_init_rot_range = np.array(self.config.get('random_init_rot_range', [-0.4, 0.4]))

        self.target_pos = np.array(self.config.get('target_pos', [5, 5, 0]))
        self.target_orn = np.array(self.config.get('target_orn', [0, 0, 0]))
        self.target_angle = self.target_orn[2]

        self.initial_pos_z_offset = self.config.get('initial_pos_z_offset', 0.1)
        check_collision_distance = self.initial_pos_z_offset * 0.5
        # s = 0.5 * G * (t ** 2)
        check_collision_distance_time = np.sqrt(check_collision_distance / (0.5 * 9.8))
        self.check_collision_loop = int(check_collision_distance_time / self.physics_timestep)

        self.additional_states_dim = self.config.get('additional_states_dim', 0)
        self.goal_format = self.config.get('goal_format', 'polar')

        # termination condition
        self.dist_tol = self.config.get('dist_tol', 0.5)
        self.max_step = self.config.get('max_step', 500)
        self.max_collisions_allowed = self.config.get('max_collisions_allowed', 500)

        # reward
        self.reward_type = self.config.get('reward_type', 'l2')
        assert self.reward_type in ['geodesic', 'l2', 'sparse']

        self.success_reward = self.config.get('success_reward', 10.0)
        self.slack_reward = self.config.get('slack_reward', -0.01)

        # reward weight
        self.potential_reward_weight = self.config.get('potential_reward_weight', 1.0)
        self.collision_reward_weight = self.config.get('collision_reward_weight', -0.1)
        self.angle_reward_weight = self.config.get('angle_reward_weight', 0.0)
        self.angle_to_target = 0.0

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 0.99)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)
        observation_space = OrderedDict()
        if 'sensor' in self.output:
            self.sensor_dim = self.additional_states_dim
            self.sensor_space = gym.spaces.Box(low=-np.inf,
                                               high=np.inf,
                                               shape=(self.sensor_dim,),
                                               dtype=np.float32)
            observation_space['sensor'] = self.sensor_space
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.image_height, self.image_width, 3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_noise_rate = self.config.get('depth_noise_rate', 0.0)
            self.depth_low = self.config.get('depth_low', 0.5)
            self.depth_high = self.config.get('depth_high', 5.0)
            self.depth_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(self.image_height, self.image_width, 1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'rgbd' in self.output:
            self.rgbd_space = gym.spaces.Box(low=0.0,
                                             high=1.0,
                                             shape=(self.image_height, self.image_width, 4),
                                             dtype=np.float32)
            observation_space['rgbd'] = self.rgbd_space
        if 'seg' in self.output:
            self.seg_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.image_height, self.image_width, 1),
                                            dtype=np.float32)
            observation_space['seg'] = self.seg_space
        if 'scan' in self.output:
            self.scan_noise_rate = self.config.get('scan_noise_rate', 0.0)
            self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
            self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
            assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
            self.laser_linear_range = self.config.get('laser_linear_range', 10.0)
            self.laser_angular_range = self.config.get('laser_angular_range', 180.0)
            self.min_laser_dist = self.config.get('min_laser_dist', 0.05)
            self.laser_link_name = self.config.get('laser_link_name', 'scan_link')
            self.scan_space = gym.spaces.Box(low=0.0,
                                             high=1.0,
                                             shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
                                             dtype=np.float32)
            observation_space['scan'] = self.scan_space
        if 'rgb_filled' in self.output:  # use filler
            try:
                import torch.nn as nn
                import torch
                from torchvision import datasets, transforms
                from gibson2.learn.completion import CompletionNet
            except:
                raise Exception('Trying to use rgb_filled ("the goggle"), but torch is not installed. Try "pip install torch torchvision".')

            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()

        self.observation_space = gym.spaces.Dict(observation_space)

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space

    def load_visualization(self):
        """
        Load visualization, such as initial and target position, shortest path, etc
        """
        if self.mode != 'gui':
            return

        cyl_length = 0.2
        self.initial_pos_vis_obj = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                                rgba_color=[1, 0, 0, 0.3],
                                                radius=self.dist_tol,
                                                length=cyl_length,
                                                initial_offset=[0, 0, cyl_length / 2.0])
        self.target_pos_vis_obj = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                               rgba_color=[0, 0, 1, 0.3],
                                               radius=self.dist_tol,
                                               length=cyl_length,
                                               initial_offset=[0, 0, cyl_length / 2.0])
        self.initial_pos_vis_obj.load()
        self.target_pos_vis_obj.load()

        if self.scene.build_graph:
            self.num_waypoints_vis = 250
            self.waypoints_vis = [VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                               rgba_color=[0, 1, 0, 0.3],
                                               radius=0.1,
                                               length=cyl_length,
                                               initial_offset=[0, 0, cyl_length / 2.0])
                                  for _ in range(self.num_waypoints_vis)]
            for waypoint in self.waypoints_vis:
                waypoint.load()

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.floor_num = 0

    def load(self):
        """
        Load navigation environment
        """
        super(NavigateEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_visualization()
        self.load_miscellaneous_variables()

    def global_to_local(self, pos):
        """
        Convert a 3D point in global frame to agent's local frame
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(pos - self.robots[0].get_position(), *self.robots[0].get_rpy())

    def get_additional_states(self):
        """
        :return: non-perception observation, such as goal location
        """
        additional_states = self.global_to_local(self.target_pos)[:2]
        if self.goal_format == 'polar':
            additional_states = np.array(cartesian_to_polar(additional_states[0], additional_states[1]))

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(self.robots[0].get_linear_velocity(),
                                           *self.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(self.robots[0].get_angular_velocity(),
                                            *self.robots[0].get_rpy())[2]
        additional_states = np.append(additional_states, [linear_velocity, angular_velocity])

        if self.config['task'] == 'reaching':
            end_effector_pos_local = self.global_to_local(self.robots[0].get_end_effector_position())
            additional_states = np.append(additional_states, end_effector_pos_local)

        assert additional_states.shape[0] == self.additional_states_dim, \
            'additional states dimension mismatch {} v.s. {}'.format(additional_states.shape[0], self.additional_states_dim)
        return additional_states

    def add_naive_noise_to_sensor(self, sensor_reading, noise_rate, noise_value=1.0):
        """
        Add naive sensor dropout to perceptual sensor, such as RGBD and LiDAR scan
        :param sensor_reading: raw sensor reading, range must be between [0.0, 1.0]
        :param noise_rate: how much noise to inject, 0.05 means 5% of the data will be replaced with noise_value
        :param noise_value: noise_value to overwrite raw sensor reading
        :return: sensor reading corrupted with noise
        """
        if noise_rate <= 0.0:
            return sensor_reading

        assert len(sensor_reading[(sensor_reading < 0.0) | (sensor_reading > 1.0)]) == 0,\
            'sensor reading has to be between [0.0, 1.0]'

        valid_mask = np.random.choice(2, sensor_reading.shape, p=[noise_rate, 1.0 - noise_rate])
        sensor_reading[valid_mask == 0] = noise_value
        return sensor_reading

    def get_depth(self):
        """
        :return: depth sensor reading, normalized to [0.0, 1.0]
        """
        depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
        # 0.0 is a special value for invalid entries
        depth[depth < self.depth_low] = 0.0
        depth[depth > self.depth_high] = 0.0

        # re-scale depth to [0.0, 1.0]
        depth /= self.depth_high
        depth = self.add_naive_noise_to_sensor(depth, self.depth_noise_rate, noise_value=0.0)

        return depth

    def get_rgb(self):
        """
        :return: RGB sensor reading, normalized to [0.0, 1.0]
        """
        return self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]

    def get_pc(self):
        """
        :return: pointcloud sensor reading
        """
        return self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]

    def get_normal(self):
        """
        :return: surface normal reading
        """
        return self.simulator.renderer.render_robot_cameras(modes='normal')

    def get_seg(self):
        """
        :return: semantic segmentation mask, normalized to [0.0, 1.0]
        """
        seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
        if self.num_object_classes is not None:
            seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
        return seg

    def get_scan(self):
        """
        :return: LiDAR sensor reading, normalized to [0.0, 1.0]
        """
        laser_angular_half_range = self.laser_angular_range / 2.0
        if self.laser_link_name not in self.robots[0].parts:
            raise Exception('Trying to simulate LiDAR sensor, but laser_link_name cannot be found in the robot URDF file. Please add a link named laser_link_name at the intended laser pose. Feel free to check out assets/models/turtlebot/turtlebot.urdf and examples/configs/turtlebot_p2p_nav.yaml for examples.')
        laser_pose = self.robots[0].parts[self.laser_link_name].get_pose()
        angle = np.arange(-laser_angular_half_range / 180 * np.pi,
                          laser_angular_half_range / 180 * np.pi,
                          self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
        unit_vector_local = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
        transform_matrix = quat2mat([laser_pose[6], laser_pose[3], laser_pose[4], laser_pose[5]])  # [x, y, z, w]
        unit_vector_world = transform_matrix.dot(unit_vector_local.T).T

        start_pose = np.tile(laser_pose[:3], (self.n_horizontal_rays, 1))
        start_pose += unit_vector_world * self.min_laser_dist
        end_pose = laser_pose[:3] + unit_vector_world * self.laser_linear_range
        results = p.rayTestBatch(start_pose, end_pose, 6)  # numThreads = 6

        hit_fraction = np.array([item[2] for item in results])  # hit fraction = [0.0, 1.0] of self.laser_linear_range
        hit_fraction = self.add_naive_noise_to_sensor(hit_fraction, self.scan_noise_rate)
        scan = np.expand_dims(hit_fraction, 1)
        return scan

    def get_state(self, collision_links=[]):
        """
        :param collision_links: collisions from last time step
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if 'sensor' in self.output:
            state['sensor'] = self.get_additional_states()
        if 'rgb' in self.output:
            state['rgb'] = self.get_rgb()
        if 'depth' in self.output:
            state['depth'] = self.get_depth()
        if 'pc' in self.output:
            state['pc'] = self.get_pc()
        if 'rgbd' in self.output:
            rgb = self.get_rgb()
            depth = self.get_depth()
            state['rgbd'] = np.concatenate((rgb, depth), axis=2)
        if 'normal' in self.output:
            state['normal'] = self.get_normal()
        if 'seg' in self.output:
            state['seg'] = self.get_seg()
        if 'rgb_filled' in self.output:
            with torch.no_grad():
                tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
                rgb_filled = self.comp(tensor[None, :, :, :])[0].permute(1, 2, 0).cpu().numpy()
                state['rgb_filled'] = rgb_filled
        if 'scan' in self.output:
            state['scan'] = self.get_scan()
        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (simulator_loop physics timestep)
        :return: collisions from this simulation
        """
        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))
        self.simulator.sync()

        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored
        :param collision_links: original collisions, a list of lists of collisions
        :return: filtered collisions
        """
        new_collision_links = []
        for collision_per_sim_step in collision_links:
            new_collision_per_sim_step = []
            for item in collision_per_sim_step:
                # ignore collision with body b
                if item[2] in self.collision_ignore_body_b_ids:
                    continue

                # ignore collision with robot link a
                if item[3] in self.collision_ignore_link_a_ids:
                    continue

                # ignore self collision with robot link a (body b is also robot itself)
                if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                    continue

                new_collision_per_sim_step.append(item)
            new_collision_links.append(new_collision_per_sim_step)
        return new_collision_links

    def get_position_of_interest(self):
        """
        Get position of interest.
        :return: If pointgoal task, return base position. If reaching task, return end effector position.
        """
        if self.config['task'] == 'pointgoal':
            return self.robots[0].get_position()
        elif self.config['task'] == 'reaching':
            return self.robots[0].get_end_effector_position()

    def get_shortest_path(self, from_initial_pos=False, entire_path=False):
        """
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = self.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        return self.scene.get_shortest_path(self.floor_num, source, target, entire_path=entire_path)

    def get_geodesic_potential(self):
        """
        :return: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path()
        return geodesic_dist

    def get_l2_potential(self):
        """
        :return: L2 distance to the target position
        """
        return l2_distance(self.target_pos, self.get_position_of_interest())

    def get_angle_reward(self):
        """

        Returns: angle difference to target

        """
        parts_xyz = np.array([p.get_pose() for p in self.robots[0].parts.values()]).flatten()
        self.body_xy = (
            parts_xyz[0::3].mean(), parts_xyz[1::3].mean())

        self.walk_target_angle = np.arctan2(self.target_pos[1] - self.body_xy[1],
                                            self.target_pos[0] - self.body_xy[0])

        r, p, yaw = self.robots[0].get_rpy()
        self.angle_to_target = self.walk_target_angle - yaw
        if self.angle_to_target > np.pi:
            self.angle_to_target -= 2 * np.pi
        elif self.angle_to_target < -np.pi:
            self.angle_to_target += 2 * np.pi

        is_forward = np.abs(self.angle_to_target) < 1.57
        diff_angle = np.abs(self.angle_to_target)

        return diff_angle

    def is_goal_reached(self):
        return l2_distance(self.get_position_of_interest(), self.target_pos) < self.dist_tol

    def get_reward(self, collision_links=[], action=None, info={}):
        """
        :param collision_links: collisions from last time step
        :param action: last action
        :param info: a dictionary to store additional info
        :return: reward, info
        """
        collision_links_flatten = [item for sublist in collision_links for item in sublist]
        reward = self.slack_reward  # |slack_reward| = 0.01 per step

        if self.reward_type == 'l2':
            new_potential = self.get_l2_potential()
        elif self.reward_type == 'geodesic':
            new_potential = self.get_geodesic_potential()
            # print("new_potential: ", new_potential )
        potential_reward = self.potential - new_potential
        reward += potential_reward * self.potential_reward_weight  # |potential_reward| ~= 0.1 per step
        self.potential = new_potential

        collision_reward = float(len(collision_links_flatten) > 0)
        self.collision_step += int(collision_reward)
        reward += collision_reward * self.collision_reward_weight  # |collision_reward| ~= 1.0 per step if collision

        # angle_cost
        angle_reward = self.get_angle_reward()
        reward += angle_reward * self.angle_reward_weight

        if self.is_goal_reached():
            reward += self.success_reward  # |success_reward| = 10.0 per step
        return reward, info

    def get_termination(self, collision_links=[], action=None, info={}):
        """
        :param collision_links: collisions from last time step
        :param info: a dictionary to store additional info
        :return: done, info
        """
        done = False

        # goal reached
        if self.is_goal_reached():
            done = True
            info['success'] = True

        # max collisions reached
        if self.collision_step > self.max_collisions_allowed:
            done = True
            info['success'] = False

        # time out
        elif self.current_step >= self.max_step:
            done = True
            info['success'] = False

        if done:
            info['episode_length'] = self.current_step
            info['collision_step'] = self.collision_step
            info['path_length'] = self.path_length
            info['spl'] = float(info['success']) * min(1.0, self.geodesic_dist / self.path_length)

        return done, info

    def before_simulation(self):
        """
        Cache bookkeeping data before simulation
        :return: cache
        """
        return {'robot_position': self.robots[0].get_position()}

    def after_simulation(self, cache, collision_links):
        """
        Accumulate evaluation stats
        :param cache: cache returned from before_simulation
        :param collision_links: collisions from last time step
        """
        old_robot_position = cache['robot_position'][:2]
        new_robot_position = self.robots[0].get_position()[:2]
        self.path_length += l2_distance(old_robot_position, new_robot_position)

    def step_visualization(self):
        if self.mode != 'gui':
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)
        self.target_pos_vis_obj.set_position(self.target_pos)

        if self.scene.build_graph:
            shortest_path, _ = self.get_shortest_path(entire_path=True)
            floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
            num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
            for i in range(num_nodes):
                self.waypoints_vis[i].set_position(pos=np.array([shortest_path[i][0],
                                                                 shortest_path[i][1],
                                                                 floor_height]))
            for i in range(num_nodes, self.num_waypoints_vis):
                self.waypoints_vis[i].set_position(pos=np.array([0.0, 0.0, 100.0]))

    def step(self, action):
        """
        apply robot's action and get state, reward, done and info, following OpenAI gym's convention
        :param action: a list of control signals
        :return: state, reward, done, info
        """
        self.current_step += 1
        if action is not None:
            self.robots[0].apply_action(action)
        cache = self.before_simulation()
        collision_links = self.run_simulation()
        self.after_simulation(cache, collision_links)

        state = self.get_state(collision_links)

        info = {}
        reward, info = self.get_reward(collision_links, action, info)
        done, info = self.get_termination(collision_links, action, info)
        self.step_visualization()

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()
        return state, reward, done, info

    def reset_agent(self):
        """
        Reset the robot's joint configuration and base pose until no collision
        """
        reset_success = False
        max_trials = 500
        for _ in range(max_trials):
            self.reset_initial_and_target_pos()
            if self.test_valid_position('robot', self.robots[0], self.initial_pos, self.initial_orn) and \
                    self.test_valid_position('robot', self.robots[0], self.target_pos):
                reset_success = True
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        self.land('robot', self.robots[0], self.initial_pos, self.initial_orn)

    def reset_initial_and_target_pos(self):
        """
        Reset initial_pos, initial_orn and target_pos
        """
        return

    def check_collision(self, body_id):
        """
        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        for _ in range(self.check_collision_loop):
            self.simulator_step()
            collisions = list(p.getContactPoints(bodyA=body_id))

            if logging.root.level <= logging.DEBUG: #Only going into this if it is for logging --> efficiency
                for item in collisions:
                    logging.debug('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))

            if len(collisions) > 0:
                return False
        return True

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        obj.set_position_orientation([pos[0], pos[1], pos[2] + offset],
                                     quatToXYZW(euler2quat(*orn), 'wxyz'))

    def test_valid_position(self, obj_type, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision
        :param obj_type: string "robot" or "obj"
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        assert obj_type in ['robot', 'obj']

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if obj_type == 'robot':
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if obj_type == 'robot' else obj.body_id
        return self.check_collision(body_id)

    def land(self, obj_type, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation
        :param obj_type: string "robot" or "obj"
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        assert obj_type in ['robot', 'obj']

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if obj_type == 'robot':
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if obj_type == 'robot' else obj.body_id

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.physics_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if obj_type == 'robot':
            obj.robot_specific_reset()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.path_length = 0.0
        self.geodesic_dist = self.get_geodesic_potential()

    def reset(self):
        """
        Reset episode
        """
        self.reset_agent()
        self.simulator.sync()
        state = self.get_state()
        if self.reward_type == 'l2':
            self.potential = self.get_l2_potential()
        elif self.reward_type == 'geodesic':
            self.potential = self.get_geodesic_potential()
        self.reset_variables()
        self.step_visualization()

        return state


class NavigateRandomEnv(NavigateEnv):
    def __init__(
            self,
            config_file,
            model_id=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            random_height=False,
            device_idx=0,
            render_to_tensor=False
    ):
        """
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param random_height: whether to randomize height for target position (for reaching task)
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        super(NavigateRandomEnv, self).__init__(config_file,
                                                model_id=model_id,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                device_idx=device_idx,
                                                render_to_tensor=render_to_tensor)
        self.random_height = random_height

        self.target_dist_min = self.config.get('target_dist_min', 1.0)
        self.target_dist_max = self.config.get('target_dist_max', 10.0)

    def reset_initial_and_target_pos(self):
        """
        Reset initial_pos, initial_orn and target_pos through randomization
        The geodesic distance (or L2 distance if traversable map graph is not built)
        between initial_pos and target_pos has to be between [self.target_dist_min, self.target_dist_max]
        """
        _, self.initial_pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
        max_trials = 100
        dist = 0.0
        for _ in range(max_trials):
            _, self.target_pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
            if self.scene.build_graph:
                _, dist = self.get_shortest_path(from_initial_pos=True)
            else:
                dist = l2_distance(self.initial_pos, self.target_pos)
            if self.target_dist_min < dist < self.target_dist_max:
                break
        if not (self.target_dist_min < dist < self.target_dist_max):
            print("WARNING: Failed to sample initial and target positions")
        self.initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

    def reset(self):
        """
        Reset episode
        """
        self.floor_num = self.scene.get_random_floor()

        if self.scene.is_interactive:
            # reset scene objects
            self.scene.reset_scene_objects()
        else:
            # reset "virtual floor" to the correct height
            self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.02)

        state = super(NavigateRandomEnv, self).reset()
        return state

class NavigateRandomInitEnv(NavigateEnv):
    def __init__(
            self,
            config_file,
            model_id=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            random_height=False,
            device_idx=0,
            render_to_tensor=False,
            random_init_m=False,
            idx=0
    ):
        """ (e.g., numpy),
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param random_height: whether to randomize height for target position (for reaching task)
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        super(NavigateRandomInitEnv, self).__init__(config_file,
                                                model_id=model_id,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                device_idx=device_idx,
                                                render_to_tensor=render_to_tensor)
        self.random_height = random_height

        self.target_dist_min = self.config.get('target_dist_min', 1.0)
        self.target_dist_max = self.config.get('target_dist_max', 7.0)
        self.random_init_m = random_init_m
        self.idx = 0


    def reset_initial_and_target_pos(self):
        """
        Reset initial_pos, initial_orn and target_pos through randomization
        The geodesic distance (or L2 distance if traversable map graph is not built)
        between initial_pos and target_pos has to be between [self.target_dist_min, self.target_dist_max]
        """
        if self.random_init_m is 1:
            _, self.initial_pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)

            max_trials = 500
            dist = 0.0
            for _ in range(max_trials):
                if self.scene.build_graph:
                    _, dist = self.get_shortest_path(from_initial_pos=True)
                else:
                    dist = l2_distance(self.initial_pos, self.target_pos)
                if self.target_dist_min < dist < self.target_dist_max:
                    break
            if not (self.target_dist_min < dist < self.target_dist_max):
                print("WARNING: Failed to sample initial and target positions")
            self.initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        if self.random_init_m is 2:
            rand_x = np.random.uniform(self.random_init_x_range[0], self.random_init_x_range[1])
            rand_y = np.random.uniform(self.random_init_y_range[0], self.random_init_y_range[1])
            rand_rot = np.random.uniform(self.random_init_rot_range[0], self.random_init_rot_range[1])
            self.initial_pos = np.array([self.initial_pos_cp[0] + rand_x, self.initial_pos_cp[1] + rand_y, self.initial_pos_cp[2]])
            self.initial_orn = np.array([0., 0., self.initial_orn_cp[2] + rand_rot])

        if self.random_init_m is 3:
            x_test1 = np.load("../test1.npy")
            self.initial_pos = np.array([self.initial_pos_cp[0] + x_test1[self.idx-1], self.initial_pos_cp[1], self.initial_pos_cp[2]])
        if self.random_init_m is 4:
            x_test2 = np.load("../test2_x.npy")
            y_test2 = np.load("../test2_y.npy")
            rot_test2 = np.load("../test2_rot.npy")
            self.initial_pos = np.array(
                [self.initial_pos_cp[0] + x_test2[self.idx-1], self.initial_pos_cp[1] + y_test2[self.idx-1], self.initial_pos_cp[2]])
            self.initial_orn = np.array([0., 0., self.initial_orn_cp[2] + rot_test2[self.idx-1]])



    def reset(self, pos_idx=0):
        """
        Reset episode
        """
        self.idx = self.idx + 1
        # self.floor_num = self.scene.get_random_floor()
        #
        # if self.scene.is_interactive:
        #     # reset scene objects
        #     self.scene.reset_scene_objects()
        # else:
        #     # reset "virtual floor" to the correct height
        #     self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.02)
        #
        # state = super(NavigateRandomInitEnv, self).reset()
        # return state

class NavigateRandomInitEnvSim2Real(NavigateRandomInitEnv):
    def __init__(self,
                 config_file,
                 model_id=None,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0,
                 render_to_tensor=False,
                 automatic_reset=False,
                 collision_reward_weight=0.0,
                 track='static',
                 random_init_m=0,
                 idx=0,
                 seed=0
                 ):
        super(NavigateRandomInitEnvSim2Real, self).__init__(config_file,
                                                        model_id=model_id,
                                                        mode=mode,
                                                        action_timestep=action_timestep,
                                                        physics_timestep=physics_timestep,
                                                        automatic_reset=automatic_reset,
                                                        random_height=False,
                                                        device_idx=device_idx,
                                                        render_to_tensor=render_to_tensor,
                                                        random_init_m=random_init_m,
                                                        idx=idx)
        self.collision_reward_weight = collision_reward_weight
        np.random.seed(seed)

        assert track in ['static', 'interactive', 'dynamic'], 'unknown track'
        self.track = track

        if self.track == 'interactive':
            self.interactive_objects_num_dups = 1
            self.interactive_objects = self.load_interactive_objects()
            # does not penalize collision with these interactive objects
            self.collision_ignore_body_b_ids |= set([obj.body_id for obj in self.interactive_objects])
        elif self.track == 'dynamic':
            self.num_dynamic_objects = 1
            self.dynamic_objects = []
            self.dynamic_objects_last_actions = []
            for _ in range(self.num_dynamic_objects):
                robot = Turtlebot(self.config)
                self.simulator.import_robot(robot)
                self.dynamic_objects.append(robot)
                self.dynamic_objects_last_actions.append(robot.action_space.sample())

            # dynamic objects will repeat their actions for 10 action timesteps
            self.dynamic_objects_action_repeat = 10

    def load_interactive_objects(self):
        """
        Load interactive objects
        :return: a list of interactive objects
        """
        interactive_objects = []
        interactive_objects_path = [
            'object_2eZY2JqYPQE.urdf',
            'object_lGzQi2Pk5uC.urdf',
            'object_ZU6u5fvE8Z1.urdf',
            'object_H3ygj6efM8V.urdf',
            'object_RcqC01G24pR.urdf'
        ]

        for _ in range(self.interactive_objects_num_dups):
            for urdf_model in interactive_objects_path:
                obj = InteractiveObj(os.path.join(gibson2.assets_path, 'models/sample_urdfs', urdf_model))
                self.simulator.import_object(obj)
                interactive_objects.append(obj)
        return interactive_objects

    def reset_interactive_objects(self):
        """
        Reset the poses of interactive objects to have no collisions with the scene mesh
        """
        sample_x = np.linspace(-2, 4, 10000)
        sample_y = np.linspace(2.6, 3.5, 100)
        max_trials = 100
        for obj in self.interactive_objects:
            reset_success = False
            for _ in range(max_trials):
                # _, pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)

                pos = [np.random.choice(sample_x), np.random.choice(sample_y), 0.]
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                if self.test_valid_position('obj', obj, pos, orn):
                    reset_success = True
                    break

            if not reset_success:
                print("WARNING: Failed to reset interactive obj without collision")

            self.land('obj', obj, pos, orn)

    def reset_dynamic_objects(self):
        """
        Reset the poses of dynamic objects to have no collisions with the scene mesh
        """
        max_trials = 100
        shortest_path, _ = self.get_shortest_path(entire_path=True)
        floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
        for robot in self.dynamic_objects:
            reset_success = False
            for _ in range(max_trials):
                pos = shortest_path[np.random.choice(shortest_path.shape[0])]
                pos = np.array([pos[0], pos[1], floor_height])
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                if self.test_valid_position('robot', robot, pos, orn):
                    reset_success = True
                    break

            if not reset_success:
                print("WARNING: Failed to reset dynamic obj without collision")

            self.land('robot', robot, pos, orn)

    def step_dynamic_objects(self):
        """
        Apply actions to dynamic objects (default: temporally extended random walk)
        """
        if self.current_step % self.dynamic_objects_action_repeat == 0:
            self.dynamic_objects_last_actions = [robot.action_space.sample() for robot in self.dynamic_objects]
        for robot, action in zip(self.dynamic_objects, self.dynamic_objects_last_actions):
            robot.apply_action(action)

    def step(self, action):
        """
        Step dynamic objects as well
        """
        if self.track == 'dynamic':
            self.step_dynamic_objects()

        return super(NavigateRandomInitEnvSim2Real, self).step(action)

    def reset(self, pos_idx=0):
        """
        Reset episode
        """
        self.floor_num = self.scene.get_random_floor()

        if self.scene.is_interactive:
            # reset scene objects
            self.scene.reset_scene_objects()
        else:
            # reset "virtual floor" to the correct height
            self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.02)

        if self.track == 'interactive':
            self.reset_interactive_objects()

        NavigateRandomInitEnv.reset(self, pos_idx)
        state = NavigateEnv.reset(self)

        if self.track == 'dynamic':
            self.reset_dynamic_objects()
            state = self.get_state()

        return state

class NavigateRandomEnvSim2Real(NavigateRandomEnv):
    def __init__(self,
                 config_file,
                 model_id=None,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0,
                 render_to_tensor=False,
                 automatic_reset=False,
                 collision_reward_weight=0.0,
                 track='static'
                 ):
        super(NavigateRandomEnvSim2Real, self).__init__(config_file,
                                                        model_id=model_id,
                                                        mode=mode,
                                                        action_timestep=action_timestep,
                                                        physics_timestep=physics_timestep,
                                                        automatic_reset=automatic_reset,
                                                        random_height=False,
                                                        device_idx=device_idx,
                                                        render_to_tensor=render_to_tensor)
        self.collision_reward_weight = collision_reward_weight

        assert track in ['static', 'interactive', 'dynamic'], 'unknown track'
        self.track = track

        if self.track == 'interactive':
            self.interactive_objects_num_dups = 2
            self.interactive_objects = self.load_interactive_objects()
            # does not penalize collision with these interactive objects
            self.collision_ignore_body_b_ids |= set([obj.body_id for obj in self.interactive_objects])
        elif self.track == 'dynamic':
            self.num_dynamic_objects = 1
            self.dynamic_objects = []
            self.dynamic_objects_last_actions = []
            for _ in range(self.num_dynamic_objects):
                robot = Turtlebot(self.config)
                self.simulator.import_robot(robot)
                self.dynamic_objects.append(robot)
                self.dynamic_objects_last_actions.append(robot.action_space.sample())

            # dynamic objects will repeat their actions for 10 action timesteps
            self.dynamic_objects_action_repeat = 10

    def load_interactive_objects(self):
        """
        Load interactive objects
        :return: a list of interactive objects
        """
        interactive_objects = []
        interactive_objects_path = [
            'object_2eZY2JqYPQE.urdf',
            'object_lGzQi2Pk5uC.urdf',
            'object_ZU6u5fvE8Z1.urdf',
            'object_H3ygj6efM8V.urdf',
            'object_RcqC01G24pR.urdf'
        ]

        for _ in range(self.interactive_objects_num_dups):
            for urdf_model in interactive_objects_path:
                obj = InteractiveObj(os.path.join(gibson2.assets_path, 'models/sample_urdfs', urdf_model))
                self.simulator.import_object(obj)
                interactive_objects.append(obj)
        return interactive_objects

    def reset_interactive_objects(self):
        """
        Reset the poses of interactive objects to have no collisions with the scene mesh
        """
        max_trials = 100
        for obj in self.interactive_objects:
            reset_success = False
            for _ in range(max_trials):
                _, pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                if self.test_valid_position('obj', obj, pos, orn):
                    reset_success = True
                    break

            if not reset_success:
                print("WARNING: Failed to reset interactive obj without collision")

            self.land('obj', obj, pos, orn)

    def reset_dynamic_objects(self):
        """
        Reset the poses of dynamic objects to have no collisions with the scene mesh
        """
        max_trials = 100
        shortest_path, _ = self.get_shortest_path(entire_path=True)
        floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
        for robot in self.dynamic_objects:
            reset_success = False
            for _ in range(max_trials):
                pos = shortest_path[np.random.choice(shortest_path.shape[0])]
                pos = np.array([pos[0], pos[1], floor_height])
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                if self.test_valid_position('robot', robot, pos, orn):
                    reset_success = True
                    break

            if not reset_success:
                print("WARNING: Failed to reset dynamic obj without collision")

            self.land('robot', robot, pos, orn)

    def step_dynamic_objects(self):
        """
        Apply actions to dynamic objects (default: temporally extended random walk)
        """
        if self.current_step % self.dynamic_objects_action_repeat == 0:
            self.dynamic_objects_last_actions = [robot.action_space.sample() for robot in self.dynamic_objects]
        for robot, action in zip(self.dynamic_objects, self.dynamic_objects_last_actions):
            robot.apply_action(action)

    def step(self, action):
        """
        Step dynamic objects as well
        """
        if self.track == 'dynamic':
            self.step_dynamic_objects()

        return super(NavigateRandomEnvSim2Real, self).step(action)

    def reset(self):
        """
        Reset episode
        """
        self.floor_num = self.scene.get_random_floor()

        if self.scene.is_interactive:
            # reset scene objects
            self.scene.reset_scene_objects()
        else:
            # reset "virtual floor" to the correct height
            self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.02)

        if self.track == 'interactive':
            self.reset_interactive_objects()

        state = NavigateEnv.reset(self)

        if self.track == 'dynamic':
            self.reset_dynamic_objects()
            state = self.get_state()

        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--env_type',
                        choices=['deterministic', 'random', 'sim2real'],
                        default='deterministic',
                        help='which environment type (deterministic | random | sim2real)')
    parser.add_argument('--sim2real_track',
                        choices=['static', 'interactive', 'dynamic'],
                        default='static',
                        help='which sim2real track (static | interactive | dynamic)')
    args = parser.parse_args()

    if args.env_type == 'deterministic':
        nav_env = NavigateEnv(config_file=args.config,
                              mode=args.mode,
                              action_timestep=1.0 / 10.0,
                              physics_timestep=1.0 / 40.0)
    elif args.env_type == 'random':
        nav_env = NavigateRandomEnv(config_file=args.config,
                                    mode=args.mode,
                                    action_timestep=1.0 / 10.0,
                                    physics_timestep=1.0 / 40.0)
    elif args.env_type == 'sim2real':
        nav_env = NavigateRandomEnvSim2Real(config_file=args.config,
                                            mode=args.mode,
                                            action_timestep=1.0 / 10.0,
                                            physics_timestep=1.0 / 40.0,
                                            track=args.sim2real_track)

    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        nav_env.reset()
        for _ in range(100):  # 10 seconds
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)
            print('reward', reward)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(nav_env.current_step, time.time() - start))
    nav_env.clean()
