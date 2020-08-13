import Robot
import Environment
from Networks import *
import Train
import Task
import ray
import warnings

import torch as pt

# Learning Parameters **************************************************************************************************
num_epochs = 3000
max_m_dim = 300 # at the memory layer
batch_size = 250
input_dim = 1 # actions
output_dim = 17*4 + 1 # observations (RGB-D array size 17 + prev action)
horizon = 80
lr = 1e-4
rnn_horizon = 1 # = horizon if time-varying
reg = 0
load = False
load_file = None # for loading checkpoint models
seed = 42
print_int = 50
ckpt_int = 100


# Simulation Parameters ************************************************************************************************
params = {}
params['time_step'] = 0.1 # seconds
params['husky_velocity'] = 2 # meters per second

# ENVIRONMENT INFORMATION
params['y_max']= 10
params['y_min']= 0
params['x_min']= -5
params['x_max']= 5.5

params['filename'] = 'model_trial'
params['mode'] = 'train'
# 'test1': test with colors seen in training; 'test2': swapped colors'; 'test3': new set of colors


# Train ****************************************************************************************************************
warnings.filterwarnings("ignore", category=UserWarning)
robot = Robot.Husky(forward_speed=params['husky_velocity'], dt=params['time_step'])
task = Task.GoalNav(goal=[3.0, 9.0], alpha=0.)
env = Environment.RandomObstacle(robot, parallel=True, gui=False, y_max=params['y_max'], y_min=params['y_min'],
                                 x_min=params['x_min'], x_max=params['x_max'], task=task, mode=params['mode'],
                                 filename=params['filename'])

net = RNN(output_dim, max_m_dim, input_dim, rnn_horizon, seed=seed)

try:
    ray.init()
    Train.train_AMR_one(env, net, num_epochs, rnn_horizon, horizon, max_m_dim, batch_size, task, lr, reg,
                        minibatch_size=0, opt_iters=1, multiprocess=True, load=load, filename=load_file, seed=seed,
                        print_int=print_int, ckpt_int=ckpt_int)

except KeyboardInterrupt:
    ray.shutdown()



