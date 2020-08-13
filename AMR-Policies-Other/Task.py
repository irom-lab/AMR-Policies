#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import torch as pt

from abc import ABC, abstractmethod


class Task(ABC):
    @abstractmethod
    def sample_initial_dist(self):
        pass

    @abstractmethod
    def cost(self, state, inputs):
        pass

    @abstractmethod
    def terminal_cost(self, state):
        pass


# Continuous
# **********************************************************************************************************************
class GoalNav(Task):
    def __init__(self, goal, alpha=0.):
        self.goal = goal
        self.alpha = alpha
        self.init_pos = pt.zeros((4))
        self.collision_flag = 0
        self.c_t = 0

    def sample_initial_dist(self):
        self.init_pos = pt.tensor([-4.0, 0., 0., 0.])
        self.collision_flag = 0

        return self.init_pos
     
    def cost(self, state, inputs, collision, t, horizon):
        goal = self.goal
        init_pos = self.init_pos[0:2]

        if collision and not self.collision_flag:
            collision_cost = pt.tensor([(horizon - t)/horizon])
            self.collision_flag = 1
            self.c_t = t
        elif not collision and not self.collision_flag:
            collision_cost = pt.tensor([0.])
        else:
            collision_cost = pt.tensor([(horizon - self.c_t) / horizon])

        goal_cost = pt.tensor([pt.norm(state[0:2] - pt.tensor(goal))/pt.norm(pt.tensor(pt.tensor(goal) - init_pos))])
        cost = self.alpha*collision_cost + (1 - self.alpha)*goal_cost

        return cost, collision_cost, goal_cost
    
    def terminal_cost(self, state):
        goal = self.goal
        init_pos = self.init_pos[0:2]
        goal_cost = pt.tensor([pt.norm(state[0:2] - pt.tensor(goal))/pt.norm(pt.tensor(pt.tensor(goal) - init_pos))])
        cost = goal_cost

        return cost


# Discrete
# **********************************************************************************************************************
class AutGoalNav(Task):
    def __init__(self, map, goal, alpha=0):
        self.map = map
        self.size = map.shape[1]
        self.len = map.shape[0]
        self.alpha = alpha
        self.goal_flag = 0
        self.init_pos = pt.zeros(2)

        self.goal = goal

    def sample_initial_dist(self, empty=False):
        self.collision_flag = 0
        self.goal_flag = 0
        self.init_pos = pt.tensor([9., 9.])

        return self.init_pos

    def cost(self, state, inputs, collision, t, horizon):
        if collision and not self.collision_flag:
            collision_cost = pt.tensor([(horizon - (t))/horizon])
            self.collision_flag = 1
            self.c_t = t
        elif not collision and not self.collision_flag:
            collision_cost = pt.tensor([0.])
        else:
            collision_cost = pt.tensor([(horizon - self.c_t) / horizon])

        goal_cost = pt.tensor([pt.norm(state[0:2] - pt.tensor(self.goal), 1) /
                               pt.norm(pt.tensor(pt.tensor(self.goal) - self.init_pos[0:2]), 1)])
        cost = self.alpha * collision_cost + (1 - self.alpha) * goal_cost

        return cost, collision_cost, goal_cost

    def terminal_cost(self, state):
        goal_cost = pt.tensor([pt.norm(state[0:2] - pt.tensor(self.goal), 1) /
                               pt.norm(pt.tensor(pt.tensor(self.goal) - self.init_pos[0:2]), 1)])

        return goal_cost
