import Robot
from Networks import *
import Environment
import warnings
import numpy as np
import Train
import Task
import torch as pt


# Learning Parameters **************************************************************************************************
num_epochs = 300
max_m_dim = 10 # memory
batch_size = 100
output_dim = 5 # obs
input_dim = 5 # actions
horizon = 8
rnn_horizon = 1
lr = 0.1
reg =  0
seed = 553


# Simulation Parameters ************************************************************************************************
robot = Robot.Automaton()
env = Environment.GridWorld(5, robot, empty=True)
map = env.generate_obstacles()

goal = pt.tensor([[5., 13.]]) # Goal in grid world

task = Task.AutGoalNav(map, goal, alpha=0.)
state = task.sample_initial_dist(empty=True)


# Train ****************************************************************************************************************
warnings.filterwarnings("ignore", category=UserWarning)
pt.manual_seed(seed)
net = RNNDiscrete(batch_size, output_dim, max_m_dim, input_dim, rnn_horizon, seed=seed)
Train.train_AMR_discrete_one(env, net, num_epochs, horizon, max_m_dim, batch_size, task, lr, reg, minibatch_size=0,
                             opt_iters=1, seed=seed)
