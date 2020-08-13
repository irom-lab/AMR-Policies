#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train
"""
import torch as pt
import numpy as np
import pybullet as pb
import ray
import pybullet_utils.bullet_client as bc

from torch.utils.tensorboard import SummaryWriter


def save_ckpt(state, filename, epoch, checkpoint_dir):
    f_path = checkpoint_dir + filename + str(epoch)+'checkpoint.pt'
    pt.save(state, f_path)

def load_ckpt(checkpoint_fpath, model, optimizer):
    checkpoint = pt.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

# Continuous
#**********************************************************************************************************************
def train_AMR_one(env, net, epochs, RNN_horizon, horizon, nmems, batch_size, task, lr, reg, minibatch_size=0,
                  opt_iters=1, multiprocess=False, load=False, filename=None, seed=0, print_int=50, ckpt_int=100):

    writer = SummaryWriter()
    checkpoint_dir = './checkpoints/'

    if minibatch_size == 0:
        minibatch_size = batch_size

    params = list(net.parameters())
    opt = pt.optim.Adam(params, lr=lr)

    if load:
        ckp_path = checkpoint_dir + filename
        net, opt = load_ckpt(ckp_path, net, opt)

    lambda1 = reg

    for epoch in range(epochs):
        print(epoch)
        log_probs = pt.zeros((horizon, minibatch_size))

        if multiprocess:
            states, outputs, mems, inputs, costs, goal_costs = multiprocess_rollout_one(env, net, nmems, horizon, task,
                                                                                        batch_size)
        else:
            states, outputs, mems, inputs, costs, goal_costs, rgb_world = rollout_one(env, net, nmems, horizon, task,
                                                                                      batch_size)

        costs_mean = costs.sum(axis=0).mean()
        cost_std = costs.sum(axis=0).std()

        for iter in range(opt_iters):
            minibatch_idx = np.random.choice(range(batch_size), size=minibatch_size, replace=False)
            outputs_minibatch = outputs[:, :, minibatch_idx]
            mems_minibatch = mems[:, :, minibatch_idx]
            inputs_minibatch = inputs[:, :, minibatch_idx]
            costs_minibatch = costs[:, minibatch_idx]

            for s in range(minibatch_size):
                mem = pt.zeros(nmems)

                for t in range(horizon):
                    log_probs[t, s] = net.log_prob(outputs_minibatch[:, t, s].detach(),
                                                   inputs_minibatch[:, t, s].detach(), mem, 0)

                    mem = mems_minibatch[:, t, s]


            # Prep weights for group LASSO calculation
            m_weights_test = (pt.stack([net.rnn[i].enc3.weight for i in range(RNN_horizon)])).permute(1, 0, 2)
            m_weights_test = m_weights_test.reshape(m_weights_test.shape[0], m_weights_test.shape[2] * RNN_horizon)
            m_size_test = m_weights_test.shape[1]

            group_lasso = pt.sum(np.sqrt(m_size_test) * (pt.sqrt(pt.sum(m_weights_test ** 2, 1))))

            if epoch % print_int == 0:
                print("memory saliency")
                print(pt.tensor([pt.sum(pt.abs(m_weights_test[i])).item() for i in range(nmems)]).sort())
                print("group lasso: ", (lambda1 * group_lasso).item())
                print("normalized distance to goal: ", costs[horizon].mean().item(), '\pm', costs[horizon].std().item())
                print("trajectory cost: ", costs_minibatch.sum(axis=0).mean().item(), '\pm',
                      costs_minibatch.sum(axis=0).std().item())

            opt.zero_grad()
            baseline = (costs_minibatch.sum(axis=0) - costs_mean) / cost_std
            loss = pt.mul(log_probs.sum(axis=0), baseline).mean() + lambda1 * group_lasso

            loss.backward()
            opt.step()

            m_weights_test.detach()
            log_probs = log_probs.detach()

        if epoch % ckpt_int == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': opt.state_dict()
            }
            save_ckpt(checkpoint, env.data_filename, epoch, checkpoint_dir)

        writer.add_scalar('Loss/Total', loss.item(), epoch)
        writer.add_scalar('Loss/Cost', costs_minibatch.sum(axis=0).mean(), epoch)
        writer.add_scalar('Loss/TermDistToGoal', costs[horizon].mean(), epoch)
        writer.add_scalar('Loss/GroupLasso', lambda1*group_lasso, epoch)

@ray.remote(num_return_vals=6)
def rollout_trajectory_one(env, net, nmems, horizon, task, i):

    bc.BulletClient(connection_mode=pb.DIRECT)

    # Generate environment
    obs_uid = env.generate_obstacles(i)

    robot_height = env.robot.height
    traj_states = [task.sample_initial_dist().reshape((-1, 1))]
    env.robot.state = traj_states[0][0:3]

    env.robot.forward_speed = 2.0

    traj_outputs = []
    traj_mem = []
    traj_inputs = []
    traj_costs = []
    traj_goal_costs = []

    mem = pt.zeros(nmems, requires_grad=True).reshape((-1, 1))

    quat = env.p.getQuaternionFromEuler([0., 0., traj_states[0][2] + np.pi / 2])
    env.p.resetBasePositionAndOrientation(env.husky, [traj_states[0][0], traj_states[0][1], 0.], quat)
    env.p.resetBasePositionAndOrientation(env.sphere, [traj_states[0][0], traj_states[0][1], robot_height],
                                          [0, 0, 0, 1])

    goal_flag = 0

    for t in range(horizon):
        pos = [traj_states[t][0], traj_states[t][1], 0]
        quat = env.p.getQuaternionFromEuler([0., 0., traj_states[t][2] + np.pi / 2])

        # Check for collision
        collision = env.p.getClosestPoints(env.sphere, obs_uid, 0.0)

        if collision or goal_flag:
            env.robot.forward_speed = 0.0
            obs = traj_outputs[t-1]

        else:
            sens = env.robot.sensor(pos, quat, w=17, h=15).reshape((-1, 1))
            sens_state = (1/20*traj_states[t][3]).reshape((-1,1))
            obs = pt.cat((sens, sens_state ), 0)

        traj_outputs.append(obs)
        traj_outputs[t].requires_grad_(True)
        ins, mem = net(traj_outputs[t].flatten(), mem.flatten(),0)

        if goal_flag:
            env.robot.forward_speed = 0.0

        traj_mem.append(mem.reshape(-1, 1))
        traj_inputs.append(ins.reshape(-1, 1))
        cost, collision_cost, goal_cost = task.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), collision, t,
                                                    horizon)
        traj_costs.append(cost.reshape((-1, 1)))
        traj_goal_costs.append(goal_cost.reshape((-1, 1)))

        if collision:
            state = traj_states[t]
        else:
            state = env.robot.update_state(traj_states[t].flatten().detach(), traj_inputs[t].flatten().detach(),
                                           t).reshape((-1,1))

        traj_states.append(state)

        env.p.resetBasePositionAndOrientation(env.husky, [traj_states[t][0], traj_states[t][1], 0.], quat)
        env.p.resetBasePositionAndOrientation(env.sphere, [traj_states[t][0], traj_states[t][1], robot_height],
                                              [0, 0, 0, 1])

        if goal_cost < 0.12:
            env.robot.forward_speed = 0.0
            goal_flag = 1

    traj_costs.append(task.terminal_cost(traj_states[-1].flatten()).reshape((-1, 1)))
    env.p.disconnect()

    return pt.cat(traj_states, axis=1), pt.cat(traj_outputs, axis=1), pt.cat(traj_mem, axis=1), \
           pt.cat(traj_inputs, axis=1), pt.cat( traj_costs, axis=1), pt.cat(traj_goal_costs, axis=1)


def multiprocess_rollout_one(env, net, nmems, horizon, task, batch_size):

    states = []
    outputs = []
    mems = []
    inputs = []
    costs = []
    goal_costs = []

    results = [rollout_trajectory_one.remote(env, net, nmems, horizon, task, i) for i in range(batch_size)]

    for r in results:
        result = ray.get(r)
        states.append(result[0])
        outputs.append(result[1])
        mems.append(result[2])
        inputs.append(result[3])
        costs.append(result[4])
        goal_costs.append(result[5])

    states = pt.stack(states, axis=2).detach()
    outputs = pt.stack(outputs, axis=2).detach()
    mems = pt.stack(mems, axis=2).detach()
    inputs = pt.stack(inputs, axis=2).detach()
    costs = pt.stack(costs, axis=2)[0, :, :].detach()
    goal_costs = pt.stack(goal_costs, axis=2)[0, :, :].detach()

    return states, outputs, mems, inputs, costs, goal_costs


def rollout_one(env, net, nmems, horizon, task, batch_size, video):
    states = []
    outputs = []
    mems = []
    inputs = []
    costs = []
    goal_costs = []

    rgb_data = [None] * batch_size
    for s in range(batch_size):
        # Generate environment
        obs_uid = env.generate_obstacles(s)

        robot_height = env.robot.height
        traj_states = [task.sample_initial_dist().reshape((-1, 1))]
        env.robot.state = traj_states[0][0:3]

        env.robot.forward_speed = 2.0

        traj_outputs = []
        traj_mem = []
        traj_inputs = []
        traj_costs = []
        traj_goal_costs = []

        mem = pt.zeros(nmems, requires_grad=True).reshape((-1, 1))

        quat = env.p.getQuaternionFromEuler([0., 0., traj_states[0][2] + np.pi / 2])
        env.p.resetBasePositionAndOrientation(env.husky, [traj_states[0][0], traj_states[0][1], 0.], quat)
        env.p.resetBasePositionAndOrientation(env.sphere, [traj_states[0][0], traj_states[0][1], robot_height],
                                              [0, 0, 0, 1])

        goal_flag = 0

        rgb_world = [None] * horizon
        for t in range(horizon):
            pos = [traj_states[t][0], traj_states[t][1], 0]
            quat = env.p.getQuaternionFromEuler([0., 0., traj_states[t][2] + np.pi / 2])

            if video:
                rgb_world[t], depth_world = mount_world_cam(env)

            # Check for collision
            collision = env.p.getClosestPoints(env.sphere, obs_uid, 0.0)

            if collision or goal_flag:
                env.robot.forward_speed = 0.0
                obs = traj_outputs[t - 1]

            else:
                sens = env.robot.sensor(pos, quat, w=17, h=15).reshape((-1, 1))
                sens_state = (1/20*traj_states[t][3]).reshape((-1,1))
                obs = pt.cat((sens, sens_state ), 0)

            traj_outputs.append(obs)
            traj_outputs[t].requires_grad_(True)
            ins, mem = net(traj_outputs[t].flatten(), mem.flatten(), 0)

            if goal_flag:
                env.robot.forward_speed = 0.0

            traj_mem.append(mem.reshape(-1, 1))
            traj_inputs.append(ins.reshape(-1, 1))
            cost, collision_cost, goal_cost = task.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), collision,
                                                        t, horizon)
            traj_costs.append(cost.reshape((-1, 1)))
            traj_goal_costs.append(goal_cost.reshape((-1, 1)))

            if collision:
                state = traj_states[t]
            else:
                state = env.robot.update_state(traj_states[t].flatten().detach(), traj_inputs[t].flatten().detach(),
                                               t).reshape((-1, 1))

            traj_states.append(state)

            # Update position of pybullet object
            env.p.resetBasePositionAndOrientation(env.husky, [traj_states[t][0], traj_states[t][1], 0.], quat)
            env.p.resetBasePositionAndOrientation(env.sphere, [traj_states[t][0], traj_states[t][1], robot_height],
                                                  [0, 0, 0, 1])

            if goal_cost < 0.12:
                env.robot.forward_speed = 0.0
                goal_flag = 1

        rgb_data[s] = rgb_world
        traj_costs.append(task.terminal_cost(traj_states[-1].flatten()).reshape((-1, 1)))

        states.append(pt.cat(traj_states, axis=1))
        outputs.append(pt.cat(traj_outputs, axis=1))
        mems.append(pt.cat(traj_mem, axis=1))
        inputs.append(pt.cat(traj_inputs, axis=1))
        costs.append(pt.cat(traj_costs, axis=1))
        goal_costs.append(pt.cat(traj_goal_costs, axis=1))

        # Remove obstacles
        env.p.removeBody(obs_uid)

    states = pt.stack(states, axis=2).detach()
    outputs = pt.stack(outputs, axis=2).detach()
    mems = pt.stack(mems, axis=2).detach()
    inputs = pt.stack(inputs, axis=2).detach()
    costs = pt.stack(costs, axis=2)[0, :, :].detach()
    goal_costs = pt.stack(goal_costs, axis=2)[0, :, :].detach()

    return states, outputs, mems, inputs, costs, goal_costs, rgb_data


def mount_world_cam(env, w=50, h=50):
        '''
        Mounts an RGB camera in world in pybullet
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

        p = env.p

        view_matrix = p.computeViewMatrix(cameraEyePosition=[0, 5, 14.5], cameraTargetPosition=[0, 5, 0],
                                          cameraUpVector=[0, 1, 0])

        # Get Image
        projection_matrix = p.computeProjectionMatrixFOV(fov=90.0, aspect=1., nearVal=0.1, farVal=15)
        _, _, rgb, depth, _ = p.getCameraImage(1000, 1000, view_matrix, projection_matrix)

        return rgb, depth


# Discrete
#**********************************************************************************************************************
def train_AMR_discrete_one(env, net, epochs, horizon, nmems, batch_size, task, lr, reg, minibatch_size=0, opt_iters=1,
                           seed=0):

    if minibatch_size == 0:
        minibatch_size = batch_size

    params = list(net.parameters())
    opt = pt.optim.Adam(params, lr=lr)

    lambda1 = reg

    for epoch in range(epochs):
        print(epoch)

        states, outputs, mems, inputs, costs, goal_costs = automaton_rollout_one(env, net, nmems, horizon, task,
                                                                                 batch_size)

        costs_mean = costs.sum(axis=0).mean()
        cost_std = costs.sum(axis=0).std()



        log_probs = pt.zeros((horizon, minibatch_size))

        for iter in range(opt_iters):
            minibatch_idx = np.random.choice(range(batch_size), size=minibatch_size, replace=False)
            outputs_minibatch = outputs[:, :, minibatch_idx]
            mems_minibatch = mems[:, :, minibatch_idx]
            inputs_minibatch = inputs[:, :, minibatch_idx]
            costs_minibatch = costs[:, minibatch_idx]

            for s in range(minibatch_size):
                mem = pt.zeros(nmems)
                for t in range(horizon):
                    log_probs[t, s] = net.log_prob(outputs[:, t, s].detach(), inputs_minibatch[:, t, s], mem, 0)
                    mem = mems_minibatch[:, t, s]

            m_weights = (pt.stack([net.rnn[i].enc2.weight for i in range(1)])).permute(1, 0, 2)
            m_weights = m_weights.reshape(m_weights.shape[0], m_weights.shape[2] * 1)
            m_size = m_weights.shape[1]

            group_lasso = pt.sum(np.sqrt(m_size) * (pt.sqrt(pt.sum(m_weights ** 2, 1))))

            print("group lasso: ", (lambda1 * group_lasso).item())
            print("memory saliency: ", pt.tensor([pt.sum(pt.abs(m_weights[i])).item() for i in range(nmems)]))
            print("trajectory cost: ", costs_mean.item())
            print("First 10 policies: ")
            print(inputs.squeeze()[:, 0:10])
            print("First 10 mem_states: ")
            print(pt.stack([pt.argmax(mems.squeeze(), dim=0)[i, 0:10] for i in range(horizon)]))

            opt.zero_grad()
            baseline = (costs_minibatch.sum(axis=0)- costs_mean)/ cost_std
            loss = pt.mul(log_probs.sum(axis=0), baseline).mean() + lambda1 * group_lasso
            loss.backward()
            opt.step()

            log_probs = log_probs.detach()

def automaton_rollout_one(env, net, nmems, horizon, task, batch_size):
    states = []
    outputs = []
    mems = []
    inputs = []
    costs = []
    goal_costs = []

    for s in range(batch_size):
        # Generate environment
        map = env.generate_obstacles()

        traj_states = [task.sample_initial_dist().reshape((-1, 1))]
        state = traj_states[0][0:2]

        traj_outputs = []
        traj_mem = []
        traj_inputs = []
        ins_map = []
        traj_costs = []
        traj_goal_costs = []


        goal_flag = 0
        mem = pt.zeros(nmems, requires_grad=True).reshape((-1, 1))
        mem[0] = 1.
        for t in range(horizon):

            # Check for collision
            if map[int(traj_states[t][0]), int(traj_states[t][1])] == 1:
                collision = 1
            else:
                collision = 0

            obs = env.robot.sensor(map, traj_states[t], task.goal).reshape((-1, 1))
            traj_outputs.append(obs)
            traj_outputs[t].requires_grad_(True)
            ins, mem = net(traj_outputs[t].flatten(), mem.flatten(), 0)
            traj_mem.append(mem.reshape((-1,1)))

            if collision:
                ins = pt.tensor([4])

            if goal_flag:
                ins = pt.tensor([4])

            ins_map.append(env.robot.ins_map(ins).reshape((-1,1)))
            traj_inputs.append(ins.reshape((-1,1)))
            cost, collision_cost, goal_cost = task.cost(traj_states[t].flatten(), ins_map[t].flatten(), collision,
                                                        t, horizon)

            traj_costs.append(cost.reshape((-1, 1)))
            traj_states.append(env.robot.update_state(traj_states[t].flatten().detach(),
                                                      ins_map[t].flatten().detach()).reshape((-1, 1)))
            traj_goal_costs.append(goal_cost.reshape((-1,1)))

            if pt.equal(goal_cost, pt.tensor([0.])):
                goal_flag = 1

        traj_costs.append(task.terminal_cost(traj_states[-1].flatten()).reshape((-1, 1)))
        states.append(pt.cat(traj_states, axis=1))
        outputs.append(pt.cat(traj_outputs, axis=1))
        mems.append(pt.cat(traj_mem, axis=1))
        inputs.append(pt.cat(traj_inputs, axis=1))
        costs.append(pt.cat(traj_costs, axis=1))
        goal_costs.append(pt.cat(traj_goal_costs, axis=1))

    states = pt.stack(states, axis=2).detach()
    outputs = pt.stack(outputs, axis=2).detach()
    mems = pt.stack(mems, axis=2).detach()
    inputs = pt.stack(inputs, axis=2).detach()
    costs = pt.stack(costs, axis=2)[0, :, :].detach()
    goal_costs = pt.stack(goal_costs, axis=2)[0, :, :].detach()

    return states, outputs, mems, inputs, costs, goal_costs