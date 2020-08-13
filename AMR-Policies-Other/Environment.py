import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np

from abc import ABC, abstractmethod


class SetEnvironment():

    def setup_pybullet(self, robot_file, parallel=False):
        robot_radius = self.robot.robot_radius
        if parallel:
            if self.gui:
                print("Warning: Can only have one thread be a gui")
                p = bc.BulletClient(connection_mode=pybullet.GUI)
                visual_shape_id = p.createVisualShape(pybullet.GEOM_SPHERE, radius=robot_radius, rgbaColor=[0, 0, 0, 0])
            else:
                p = bc.BulletClient(connection_mode=pybullet.DIRECT)
                visual_shape_id = -1
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        else:
            if self.gui:
                pybullet.connect(pybullet.GUI)
                p = pybullet
                # This just makes sure that the sphere is not visible (we only use the sphere for collision checking)
                visual_shape_id = p.createVisualShape(pybullet.GEOM_SPHERE, radius=robot_radius, rgbaColor=[0, 0, 0, 0])
            else:
                pybullet.connect(pybullet.DIRECT)
                p = pybullet
                visual_shape_id = -1

        p.loadURDF("./URDFs/plane.urdf")  # Ground plane
        husky = p.loadURDF(robot_file, globalScaling=0.5)  # Load robot from URDF

        col_sphere_id = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=robot_radius)  # Sphere
        mass = 0
        sphere = pybullet.createMultiBody(mass, col_sphere_id, visual_shape_id)

        self.p = p
        self.husky = husky
        self.sphere = sphere

    def set_gui(self, gui):
        self.p.disconnect()
        self.gui = gui
        self.setup_pybullet()

class Environment(ABC):
    
    @abstractmethod
    def generate_obstacles(self):
        pass 


# Discrete Maze Environment
#**********************************************************************************************************************
class GridWorld(Environment):
    '''
    ENVIRONMENT DESCRIPTION:
    Grid World size 20 x 20
    '''
    def __init__(self, size, robot, empty=False, filename='None'):
        self.size = size
        self.robot = robot
        self.empty = empty
        self.data_filename = filename

    def generate_obstacles(self):
        maze = np.zeros((20,20))

        return maze


# Random Obstacle Environment
#**********************************************************************************************************************
class RandomObstacle(Environment, SetEnvironment):
    '''
    ENVIRONMENT DESCRIPTION:
    Maze Navigation environment. Walled maze with two obstacles placed in locations from files. Locations were uniformly
    sampled: y1 drawn from set [y_min + 5.2, y_min + 7.2], y2 drawn from set [y_min + 2, y_min + 8]
    '''
    def __init__(self, robot, parallel=False, gui=False, x_min=-5.0, x_max=5.0, y_min=0.0, y_max=10.0,
                 task=None, mode='train', filename=None):
        
        self.parallel = parallel
        self.gui = gui
        self.robot = robot

        self.height_obs = 100*robot.height

        self.x_lim = [x_min, x_max]
        self.y_lim = [y_min, y_max]

        self.p = None
        self.husky = None
        self.sphere = None
        self.setup_pybullet(self.robot.get_robot())

        self.task = task
        self.mode = mode
        self.data_filename = filename

        if self.mode is 'train':
            self.sample_y1 = np.load("./envs/train_Maze_250_y1.npy")
            self.sample_y2 = np.load("./envs/train_Maze_250_y2.npy")
        else:
            self.sample_y1 = np.load("./envs/test_Maze_20_y1.npy")
            self.sample_y2 = np.load("./envs/test_Maze_20_y2.npy")
        
    def generate_obstacles(self, s):
        if self.parallel:
            self.setup_pybullet(self.robot.get_robot(), self.parallel)

        p = self.p

        x_lim = self.x_lim
        y_lim = self.y_lim
        numObs = 0
        heightObs = self.height_obs
        
        numEnvParts = 9

        linkMasses = [None] * (numObs + numEnvParts) 
        colIdxs = [None] * (numObs + numEnvParts)
        visIdxs = [None] * (numObs + numEnvParts)
        posObs = [None] * (numObs + numEnvParts)
        orientObs = [None] * (numObs + numEnvParts)
        parentIdxs = [None] * (numObs + numEnvParts)
        linkInertialFramePositions = [None] * (numObs + numEnvParts)
        linkInertialFrameOrientations = [None] * (numObs + numEnvParts)
        linkJointTypes = [None] * (numObs + numEnvParts)
        linkJointAxis = [None] * (numObs + numEnvParts)

        for obs in range(numObs + numEnvParts):
            linkMasses[obs] = 0.0
            visIdxs[obs] = -1
            parentIdxs[obs] = 0
            linkInertialFramePositions[obs] = [0, 0, 0]
            linkInertialFrameOrientations[obs] = [0, 0, 0, 1]
            linkJointTypes[obs] = p.JOINT_FIXED
            linkJointAxis[obs] = np.array([0, 0, 1])
            orientObs[obs] = [0, 0, 0, 1]

        # Left wall
        posObs[numObs] = [x_lim[0], (y_lim[0] + y_lim[1] - 1) / 2.0, 0.0]
        colIdxs[numObs] = p.createCollisionShape(p.GEOM_BOX,
                                                 halfExtents=[0.1, (y_lim[1] - y_lim[0] + 1) / 2, heightObs / 2])
        visIdxs[numObs] = p.createVisualShape(p.GEOM_BOX,
                                              halfExtents=[0.1, (y_lim[1] - y_lim[0] + 1) / 2, heightObs / 2],
                                              rgbaColor=[0.5, 0.5, 0.5, 1])

        # Right wall
        posObs[numObs + 1] = [x_lim[1], (y_lim[0] + y_lim[1] - 1) / 2.0, 0.0]
        colIdxs[numObs + 1] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, (y_lim[1] - y_lim[0] + 1) / 2, heightObs / 2])
        visIdxs[numObs + 1] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, (y_lim[1] - y_lim[0] + 1) / 2, heightObs / 2],
                                                  rgbaColor=[0.5, 0.5, 0.5, 1])

        # Top Wall
        orientObs[numObs + 2] = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
        posObs[numObs + 2] = [(x_lim[0] + x_lim[1]) / 2.0, y_lim[1], 0.0]
        colIdxs[numObs + 2] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2])
        visIdxs[numObs + 2] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2],
                                                  rgbaColor=[0.5, 0.5, 0.5, 1])

        # Bottom Wall
        orientObs[numObs + 6] = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
        posObs[numObs + 6] = [(x_lim[0] + x_lim[1]) / 2.0, y_lim[0] - 1, 0.0]
        colIdxs[numObs + 6] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2])
        visIdxs[numObs + 6] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2],
                                                  rgbaColor=[0.5, 0.5, 0.5, 1])
        if self.mode is 'test3':
            # Left wall
            posObs[numObs] = [x_lim[0], (y_lim[0] + y_lim[1] - 1) / 2.0, 0.0]
            colIdxs[numObs] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, (y_lim[1] - y_lim[0] + 1) / 2, heightObs / 2])
            visIdxs[numObs] = p.createVisualShape(p.GEOM_BOX,
                                                         halfExtents=[0.1, (y_lim[1] - y_lim[0] + 1) / 2, heightObs / 2],
                                                         rgbaColor=[0.27, 0.89, 0.96, 1])

            # Right wall
            posObs[numObs + 1] = [x_lim[1], (y_lim[0] + y_lim[1] - 1) / 2.0, 0.0]
            colIdxs[numObs + 1] = p.createCollisionShape(p.GEOM_BOX,
                                                         halfExtents=[0.1, (y_lim[1] - y_lim[0] + 1) / 2, heightObs / 2])
            visIdxs[numObs + 1] = p.createVisualShape(p.GEOM_BOX,
                                                         halfExtents=[0.1, (y_lim[1] - y_lim[0] + 1) / 2, heightObs / 2],
                                                         rgbaColor=[0.49, 0.59, 0.49, 1])

            # Top Wall
            orientObs[numObs + 2] = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
            posObs[numObs + 2] = [(x_lim[0] + x_lim[1]) / 2.0, y_lim[1], 0.0]
            colIdxs[numObs + 2] = p.createCollisionShape(p.GEOM_BOX,
                                                        halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2])
            visIdxs[numObs + 2] = p.createVisualShape(p.GEOM_BOX,
                                                        halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2],
                                                        rgbaColor=[0.74, 0.69, 0.36, 1])

            # Bottom Wall
            orientObs[numObs + 6] = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
            posObs[numObs + 6] = [(x_lim[0] + x_lim[1]) / 2.0, y_lim[0] - 1, 0.0]
            colIdxs[numObs + 6] = p.createCollisionShape(p.GEOM_BOX,
                                                         halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2])
            visIdxs[numObs + 6] = p.createVisualShape(p.GEOM_BOX,
                                                      halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2],
                                                      rgbaColor=[0.5, 0.5, 0.5, 1])

            #Two obstacles
            # Obstacle 1
            posObs[numObs + 3] = [x_lim[0] + 3, self.sample_y1[s], 0]
            colIdxs[numObs + 3] = p.createCollisionShape(p.GEOM_BOX,
                                                         halfExtents=[0.65, 1.5, heightObs / 2])
            visIdxs[numObs + 3] = p.createVisualShape(p.GEOM_BOX,
                                                         halfExtents=[0.65, 1.5, heightObs / 2],
                                                         rgbaColor=[0.69, 0.35, 0.47, 1])

            # Obstacle 2
            posObs[numObs + 4] = [x_lim[1] - 2., self.sample_y2[s], 0]
            colIdxs[numObs + 4] = p.createCollisionShape(p.GEOM_BOX,
                                                         halfExtents=[2, 0.65, heightObs / 2])
            visIdxs[numObs + 4] = p.createVisualShape(p.GEOM_BOX,
                                                         halfExtents=[2, 0.65, heightObs / 2],
                                                         rgbaColor=[0.38, 0.03, 0.63, 1])


        elif self.mode is 'test2':
            # Two obstacles
            # Obstacle 1
            posObs[numObs + 3] = [x_lim[0] + 3, self.sample_y1[s], 0]
            colIdxs[numObs + 3] = p.createCollisionShape(p.GEOM_BOX,
                                                         halfExtents=[0.65, 1.5, heightObs / 2])
            visIdxs[numObs + 3] = p.createVisualShape(p.GEOM_BOX,
                                                      halfExtents=[0.65, 1.5, heightObs / 2],
                                                      rgbaColor=[0, 0, 1, 1])

            # Obstacle 2
            posObs[numObs + 4] = [x_lim[1] - 2., self.sample_y2[s], 0]
            colIdxs[numObs + 4] = p.createCollisionShape(p.GEOM_BOX,
                                                         halfExtents=[2, 0.65, heightObs / 2])
            visIdxs[numObs + 4] = p.createVisualShape(p.GEOM_BOX,
                                                      halfExtents=[2, 0.65, heightObs / 2],
                                                      rgbaColor=[1, 0, 0, 1])
        else:
            # Two obstacles
            # Obstacle 1
            posObs[numObs + 3] = [x_lim[0] + 3, self.sample_y1[s], 0]
            colIdxs[numObs + 3] = p.createCollisionShape(p.GEOM_BOX,
                                                         halfExtents=[0.65, 1.5, heightObs / 2])
            visIdxs[numObs + 3] = p.createVisualShape(p.GEOM_BOX,
                                                      halfExtents=[0.65, 1.5, heightObs / 2],
                                                      rgbaColor=[1, 0, 0, 1])

            # Obstacle 2
            posObs[numObs + 4] = [x_lim[1] - 2., self.sample_y2[s], 0]
            colIdxs[numObs + 4] = p.createCollisionShape(p.GEOM_BOX,
                                                         halfExtents=[2, 0.65, heightObs / 2])
            visIdxs[numObs + 4] = p.createVisualShape(p.GEOM_BOX,
                                                      halfExtents=[2, 0.65, heightObs / 2],
                                                      rgbaColor=[0, 0, 1, 1])
        # Goal Marker
        if self.task is not None:
            posObs[numObs + 5] = [self.task.goal[0], self.task.goal[1], 0]
            visIdxs[numObs + 5] = p.createVisualShape(p.GEOM_CYLINDER,
                                                      radius=0.5, length=2.5,
                                                      rgbaColor=[0., 1., 0., 1])

        obsUid = p.createMultiBody(baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=[0, 0, 0],
                                   baseOrientation=[0, 0, 0, 1], baseInertialFramePosition=[0, 0, 0],
                                   baseInertialFrameOrientation=[0, 0, 0, 1], linkMasses=linkMasses,
                                   linkCollisionShapeIndices=colIdxs, linkVisualShapeIndices=visIdxs,
                                   linkPositions=posObs, linkOrientations=orientObs, linkParentIndices=parentIdxs,
                                   linkInertialFramePositions=linkInertialFramePositions,
                                   linkInertialFrameOrientations=linkInertialFrameOrientations,
                                   linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis)

        p.resetDebugVisualizerCamera(cameraDistance=15., cameraYaw=0., cameraPitch=-85., cameraTargetPosition=[0, 5, 0])

        return obsUid


# Corridor Environment
#**********************************************************************************************************************
class Corridor(Environment, SetEnvironment):
    '''
    ENVIRONMENT DESCRIPTION:
    One red corridor, one green corridor with randomly chosen colored walls
    '''

    def __init__(self, robot, parallel=False, gui=False, x_min=-5.0, x_max=5.0, y_min=0.0, y_max=10.0):
        self.parallel = parallel
        self.gui = gui
        self.robot = robot

        self.height_obs = 100 * robot.height

        self.x_lim = [x_min, x_max]
        self.y_lim = [y_min, y_max]

        self.p = None
        self.husky = None
        self.sphere = None
        self.setup_pybullet(self.robot.get_robot())

    def generate_obstacles(self):
        p = self.p

        x_lim = self.x_lim
        y_lim = self.y_lim
        numObs = 0
        heightObs = self.height_obs
        rgb_range = np.linspace(0, 1, 11)

        numEnvParts = 8

        linkMasses = [None] * (numObs + numEnvParts)
        colIdxs = [None] * (numObs + numEnvParts)
        visIdxs = [None] * (numObs + numEnvParts)
        posObs = [None] * (numObs + numEnvParts)
        orientObs = [None] * (numObs + numEnvParts)
        parentIdxs = [None] * (numObs + numEnvParts)
        linkInertialFramePositions = [None] * (numObs + numEnvParts)
        linkInertialFrameOrientations = [None] * (numObs + numEnvParts)
        linkJointTypes = [None] * (numObs + numEnvParts)
        linkJointAxis = [None] * (numObs + numEnvParts)

        for obs in range(numObs + numEnvParts):
            linkMasses[obs] = 0.0
            visIdxs[obs] = -1
            parentIdxs[obs] = 0
            linkInertialFramePositions[obs] = [0, 0, 0]
            linkInertialFrameOrientations[obs] = [0, 0, 0, 1]
            linkJointTypes[obs] = p.JOINT_FIXED
            linkJointAxis[obs] = np.array([0, 0, 1])
            orientObs[obs] = [0, 0, 0, 1]

        # Left wall
        posObs[numObs] = [x_lim[0], (y_lim[0] + y_lim[1]) / 2.0, 0.0]
        colIdxs[numObs] = p.createCollisionShape(p.GEOM_BOX,
                                                 halfExtents=[0.1, (y_lim[1] - y_lim[0]) / 2.0, heightObs / 2])
        visIdxs[numObs] = p.createVisualShape(p.GEOM_BOX,
                                              halfExtents=[0.1, (y_lim[1] - y_lim[0]) / 2.0, heightObs / 2],
                                              rgbaColor=[np.random.choice(rgb_range, 1), np.random.choice(rgb_range, 1),
                                                         np.random.choice(rgb_range, 1), 1])

        # Right wall
        posObs[numObs + 1] = [x_lim[1], (y_lim[0] + y_lim[1]) / 2.0, 0.0]
        colIdxs[numObs + 1] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, (y_lim[1] - y_lim[0]) / 2.0, heightObs / 2])
        visIdxs[numObs + 1] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, (y_lim[1] - y_lim[0]) / 2.0, heightObs / 2],
                                                  rgbaColor=[np.random.choice(rgb_range, 1),
                                                             np.random.choice(rgb_range, 1),
                                                             np.random.choice(rgb_range, 1), 1])

        # Bottom wall
        orientObs[numObs + 2] = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
        posObs[numObs + 2] = [(x_lim[0] + x_lim[1]) / 2.0, y_lim[0], 0.0]
        colIdxs[numObs + 2] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2])
        visIdxs[numObs + 2] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2],
                                                  rgbaColor=[np.random.choice(rgb_range, 1),
                                                             np.random.choice(rgb_range, 1),
                                                             np.random.choice(rgb_range, 1), 1])

        # Top wall
        orientObs[numObs + 3] = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
        posObs[numObs + 3] = [(x_lim[0] + x_lim[1]) / 2.0, y_lim[1], 0.0]
        colIdxs[numObs + 3] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2])
        visIdxs[numObs + 3] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2],
                                                  rgbaColor=[np.random.choice(rgb_range, 1),
                                                             np.random.choice(rgb_range, 1),
                                                             np.random.choice(rgb_range, 1), 1])

        # Corridor 1
        posObs[numObs + 4] = [-1.1, 1.5, 0.0]
        colIdxs[numObs + 4] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, 1.5, heightObs / 2])
        visIdxs[numObs + 4] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, 1.5, heightObs / 2],
                                                  rgbaColor=[0, 1, 0, 1])

        posObs[numObs + 5] = [-0.1, 1.5, 0.0]
        colIdxs[numObs + 5] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, 1.5, heightObs / 2])
        visIdxs[numObs + 5] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, 1.5, heightObs / 2],
                                                  rgbaColor=[0, 1, 0, 1])
        # Corridor 2
        posObs[numObs + 6] = [0.1, 1.5, 0.0]
        colIdxs[numObs + 6] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, 1.5, heightObs / 2])
        visIdxs[numObs + 6] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, 1.5, heightObs / 2],
                                                  rgbaColor=[1, 0, 0, 1])

        posObs[numObs + 7] = [1.1, 1.5, 0.0]
        colIdxs[numObs + 7] = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[0.1, 1.5, heightObs / 2])
        visIdxs[numObs + 7] = p.createVisualShape(p.GEOM_BOX,
                                                  halfExtents=[0.1, 1.5, heightObs / 2],
                                                  rgbaColor=[1, 0, 0, 1])

        obsUid = p.createMultiBody(baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=[0, 0, 0],
                                   baseOrientation=[0, 0, 0, 1], baseInertialFramePosition=[0, 0, 0],
                                   baseInertialFrameOrientation=[0, 0, 0, 1], linkMasses=linkMasses,
                                   linkCollisionShapeIndices=colIdxs, linkVisualShapeIndices=visIdxs,
                                   linkPositions=posObs, linkOrientations=orientObs, linkParentIndices=parentIdxs,
                                   linkInertialFramePositions=linkInertialFramePositions,
                                   linkInertialFrameOrientations=linkInertialFrameOrientations,
                                   linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis)

        p.resetDebugVisualizerCamera(cameraDistance=15., cameraYaw=0., cameraPitch=-85., cameraTargetPosition=[0, 5, 0])

        return obsUid












