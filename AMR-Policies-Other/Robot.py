import numpy as np
import pybullet as pb
import torch as pt
from Networks import * 

from abc import ABC, abstractmethod

class Robot(ABC):
    @abstractmethod
    def get_robot(self):
        pass
    
    @abstractmethod
    def sensor(self):
        pass
    
    @abstractmethod
    def update_state(self):
        pass

class Automaton:
    def get_robot(self):
        pass

    def sensor(self, map, state, goal):
        # open neighboring cells (N, E, S, W)
        obs = pt.empty(5)
        compass = pt.empty(4, 2)
        size = map.shape[0]

        compass[0] = state.squeeze() + pt.tensor([-1, 0])
        compass[1] = state.squeeze() + pt.tensor([0, 1])
        compass[2] = state.squeeze() + pt.tensor([1, 0])
        compass[3] = state.squeeze() + pt.tensor([0, -1])

        ind1 = (compass == -1).nonzero()
        ind2 = (compass == size).nonzero()

        compass[ind1[:, 0], :] = 0
        compass[ind2[:, 0], :] = 0

        for i in range(goal.shape[0]):
            if pt.equal((state.squeeze() - goal[i]), pt.tensor([0.,0.])):
                goal_flag = 1
                break
            else:
                goal_flag = 0

        obs[0] = map[int(compass[0, 0]), int(compass[0, 1])]
        obs[1] = map[int(compass[1, 0]), int(compass[1, 1])]
        obs[2] = map[int(compass[2, 0]), int(compass[2, 1])]
        obs[3] = map[int(compass[3, 0]), int(compass[3, 1])]
        obs[4] = goal_flag

        return obs

    def update_state(self, state, inputs):
        new_state = state + inputs
        return new_state

    def ins_map(self, ins):
        if pt.equal(ins, pt.tensor(0)):
            input = pt.tensor([-1., 0.])
        elif pt.equal(ins, pt.tensor(1)):
            input = pt.tensor([0., 1.])
        elif pt.equal(ins, pt.tensor(2)):
            input = pt.tensor([1., 0.])
        elif pt.equal(ins, pt.tensor(3)):
            input = pt.tensor([0., -1.])
        else:
            input = pt.tensor([0., 0.])
        return input


class Husky:

    def __init__(self, forward_speed=2.5, gain_u_diff=1, wheel_radius=0.05, wheel_base=0.5,
                 robot_radius=0.3, height=0.15/2, num_x_intercepts=5, num_y_intercepts=10, dt = 0.05):
        self.forward_speed = forward_speed  # forward speed
        self.gain_u_diff = gain_u_diff  # gain on u_diff, will change how much you turn with an input
        self.wheel_radius = wheel_radius  # radius of robot wheel
        self.wheel_base = wheel_base  # length between wheels (i.e., width of base)
        self.robot_radius = robot_radius  # radius of robot
        self.height = height  # rough height of COM of robot
        self.u_diff_max = 0.5 * (self.forward_speed / self.wheel_radius)
        self.u_diff_min = -self.u_diff_max

        self.num_x_intercepts = num_x_intercepts
        self.num_y_intercepts = num_y_intercepts

        self.dt = dt  # time step

        # State: [x, y, theta]
        # x: horizontal position
        # y: vertical position
        # theta: angle from vertical (positive is anti-clockwise)
        self.state = [0.0, 0.0, 0.0]
#        self.dists = np.zeros((1, sensor.num_rays))
        self.u_diff = 0.0

    def get_robot(self):
        return "./URDFs/husky.urdf"
          
    def sensor(self, base_p, base_o, w, h):
        '''
        Mounts an RGB-D camera on a robot in pybullet

        Parameters
        ----------
        w : Width
        h : Height
        base_p : Base position
        base_o : Base orientation as a quaternion
        Returns
        -------
        rgb : RGB image
        depth : Depth map
        '''
        cam_pos = base_p
        cam_pos[2] += .5  # lifted off the ground a bit
        
        # Rotation matrix
        rot_matrix = pb.getMatrixFromQuaternion(base_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (1, 0, 0) # x-axis
        init_up_vector = (0, 0, 1) # z-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = pb.computeViewMatrix(cam_pos, cam_pos + 0.1 * camera_vector, up_vector)

        # Get Image
        projection_matrix = pb.computeProjectionMatrixFOV(fov=90.0, aspect=1., nearVal=0.3, farVal=3)
        _, _, rgb, depth, _ = pb.getCameraImage(w, h, view_matrix, projection_matrix)

        # Reshape rgb image and drop the alpha layer (#4)
        rgb = np.array(rgb, dtype=np.uint8)
        rgb = rgb[:, :, :3]

        # Reshape depth map
        depth = np.array(depth, dtype=np.float32)

        # Configure observation
        rgb_load = rgb/255 # normalize rgb
        
        ind = 8
        rgb_load = rgb_load[ind, :, :].flatten()
        depth_load = depth[ind, :].flatten()
        obs_ = np.concatenate((rgb_load, depth_load), axis=0)

        # convert to torch tensor
        obs = pt.tensor(obs_)
        obs = obs.type(pt.float)
        obs = Variable(obs)

        return obs
    
    def update_state(self, state, inputs, t):
        '''

        '''
        u_diff = self.u_diff

        # Robot parameters
        r = self.wheel_radius  # Radius of robot wheel
        L = self.wheel_base  # Length between wheels (i.e., width of base)
        dt = self.dt # time step
        v0 = self.forward_speed  # forward speed

        u_diff *= self.gain_u_diff  # force the robot to turn slower/faster, new dynamics
        u_diff = np.maximum(self.u_diff_min, inputs)
        u_diff = np.minimum(self.u_diff_max, inputs)

        ul = v0 / r - u_diff
        ur = v0 / r + u_diff

        # Dynamics:
        # x_dot = -(r/2)*(ul + ur)*sin(theta)
        # y_dot = (r/2)*(ul + ur)*cos(theta)
        # theta_dot = (r/L)*(ur - ul)
        new_state = [0.0, 0.0, 0.0]
        new_state[0] = self.state[0] + dt * (-(r / 2) * (ul + ur) * np.sin(self.state[2]))  # x position
        new_state[1] = self.state[1] + dt * ((r / 2) * (ul + ur) * np.cos(self.state[2]))  # y position
        new_state[2] = self.state[2] + dt * ((r / L) * (ur - ul))
        self.state = new_state

        return pt.tensor([self.state[0], self.state[1], self.state[2], u_diff])