#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as pt
from torch import nn
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Categorical, OneHotCategorical

from abc import abstractmethod

# Continuous (Gaussian)
# **********************************************************************************************************************
class Cell(nn.Module):
    def __init__(self, in_dim, max_m_dim, out_dim):
        super(Cell, self).__init__()

        self.max_m_dim = max_m_dim

        self.enc1 = nn.Linear(in_dim + self.max_m_dim, in_dim + self.max_m_dim)
        self.enc2 = nn.Linear(in_dim + self.max_m_dim, in_dim + self.max_m_dim)
        self.enc3 = nn.Linear(in_dim + self.max_m_dim, self.max_m_dim, bias=False)
        self.dec3 = nn.Linear(self.max_m_dim, out_dim * 2)

    def forward(self, data_in, last_mem):
        combined = pt.cat((data_in, last_mem), 0)

        act1 = nn.ELU()
        x = act1(self.enc1(combined))
        act2 = nn.Tanh()
        x = act2(self.enc2(x))
        act3 = nn.Tanh()

        mem = act3(self.enc3(x))
        x = self.dec3(mem)

        return x, mem


class RNN(Cell):
    def __init__(self, in_dim, max_m_dim, out_dim, t, seed=0):
        super(Cell, self).__init__()
        pt.manual_seed(seed)
        self.scale = 20
        self.rnn = nn.ModuleList([Cell(in_dim, max_m_dim, out_dim) for i in range(t)])

    def forward(self, outputs, prev_mem, t):
        ins, mem = self.rnn[t](outputs, prev_mem)
        outsize = int(len(ins) / 2)

        mean = self.scale * ins[:outsize]
        logcov = ins[outsize:]
        cov = pt.diag(pt.exp(logcov) + 1e-3)

        return (MultivariateNormal(mean, cov).sample().reshape((-1, 1)), mem.reshape((-1, 1)))

    def log_prob(self, outputs, inputs, prev_mem, t):
        ins, mem = self.rnn[t](outputs, prev_mem)
        outsize = int(len(ins) / 2)

        mean = self.scale * ins[:outsize]
        logcov = ins[outsize:]
        cov = pt.diag(pt.exp(logcov) + 1e-3)

        return MultivariateNormal(mean, cov).log_prob(inputs)


# Discrete (Categorical)
# **********************************************************************************************************************
class DiscreteCell(nn.Module):
    def __init__(self, in_dim, max_m_dim, out_dim):
        super(DiscreteCell, self).__init__()
        self.max_m_dim = max_m_dim
        # self.enc1 = nn.Linear(in_dim + max_m_dim, in_dim + max_m_dim)
        self.enc2 = nn.Linear(in_dim + max_m_dim, max_m_dim, bias=False)
        self.dec3 = nn.Linear(self.max_m_dim, out_dim)

    def forward(self, data_in, last_mem):
        if self.max_m_dim == 0:
            combined = data_in
        else:
            combined = pt.cat((data_in, last_mem), 0)

        # act1 = nn.ELU()
        # x = act1(self.enc1(combined))
        act2 = BetaSoftmax(dim=0, beta=100)


        mem = act2(self.enc2(combined))
        act6 = nn.Softmax()
        x = act6(self.dec3(mem))

        return x, mem

    def initMemState(self):
        return pt.zeros(1, self.max_m_dim)


class RNNDiscrete(DiscreteCell):
    def __init__(self, batch_size, in_dim, max_m_dim, out_dim, t, seed=0):
        super(DiscreteCell, self).__init__()
        pt.manual_seed(seed)
        self.rnn = nn.ModuleList([DiscreteCell(in_dim, max_m_dim, out_dim) for i in range(t)])

    def forward(self, outputs, prev_mem, t):
        ins, mem = self.rnn[t](outputs, prev_mem)
        m = Categorical(ins)

        return m.sample(), mem.reshape((-1, 1))

    def log_prob(self, outputs, inputs, prev_mem, t):
        ins, mem = self.rnn[t](outputs, prev_mem)
        m = Categorical(ins)

        return m.log_prob(inputs)

class BetaSoftmax(nn.Module):

    __constants__ = ['dim']

    def __init__(self, dim=None, beta=1):
        super(BetaSoftmax, self).__init__()
        self.dim = dim
        self.beta = beta

    def forward(self, input):
        return nn.functional.softmax(self.beta*input, self.dim)

