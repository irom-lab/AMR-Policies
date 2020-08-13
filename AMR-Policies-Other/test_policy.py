from Networks import *
import Train
import Robot
import Environment
import Task
import pybullet as pb
import warnings
import matplotlib

# Parameters ***********************************************************************************************************
max_m_dim = 300 # at the memory layer
batch_size = 20 # number of testing envs
input_dim = 1 # actions
output_dim = 17*4 + 1 # observations (RGB-D array size 17 + prev action)
horizon = 80
rnn_horizon = 1 # = horizon if time-varying
load = False
load_file = None # for loading checkpoint models
seed = 0
video = False

# Simulation Parameters ************************************************************************************************
params = {}
params['time_step'] = 0.1 # seconds
params['husky_velocity'] = 2 # meters per second

# ENVIRONMENT INFORMATION
params['y_max']= 10
params['y_min']= 0
params['x_min']= -5
params['x_max']= 5.5

params['filename'] = 'model_trial.pt'
params['mode'] = 'test'
# 'test1': test with colors seen in training; 'test2': swapped colors'; 'test3': new set of colors


# Test *****************************************************************************************************************
warnings.filterwarnings("ignore", category=UserWarning)
pt.manual_seed(seed)
robot = Robot.Husky(forward_speed=params['husky_velocity'], dt=params['time_step'])
task = Task.GoalNav(goal=[3.0, 9.0], alpha=0.0)
env = Environment.RandomObstacle(robot, parallel=False, gui=False,
                      y_max=params['y_max'], y_min=params['y_min'], x_min=params['x_min'], x_max=params['x_max'],
                      task=task, mode=params['mode'])

net = RNN(output_dim, max_m_dim, input_dim, rnn_horizon, seed=seed)
net_path = pt.load('./models/' + params['filename'])
net.load_state_dict(net_path) # some loads might need to be net_path['state_dict'] (e.g., from checkpoint)
net.eval()

states, outputs, mems, inputs, costs, goal_costs, rgb_world = Train.rollout_one(env, net, max_m_dim, horizon, task,
                                                                                batch_size, video)
print("normalized distance to goal: ", costs[horizon].mean().item(), '\pm', costs[horizon].std().item())
print("trajectory cost: ", costs.sum(axis=0).mean().item(), '\pm',
                      costs.sum(axis=0).std().item())
