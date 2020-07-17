# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gin

from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers
from gibson2.data_utils.utils import get_train_models
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv, NavigateRandomEnvSim2Real, NavigateRandomInitEnvSim2Real
import gibson2


@gin.configurable
def load(config_file,
         model_id=None,
         env_type='gibson',
         sim2real_track='static',
         env_mode='headless',
         action_timestep=1.0 / 5.0,
         physics_timestep=1.0 / 40.0,
         device_idx=0,
         random_position=False,
         random_height=False,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None,
         random_init_m=0,
         seed=0):
    config_file = os.path.join(os.path.dirname(gibson2.__file__), config_file)
    if env_type == 'gibson':
        if random_position:
            env = NavigateRandomEnv(config_file=config_file,
                                    mode=env_mode,
                                    action_timestep=action_timestep,
                                    physics_timestep=physics_timestep,
                                    device_idx=device_idx,
                                    random_height=random_height)
        else:
            env = NavigateEnv(config_file=config_file,
                              mode=env_mode,
                              action_timestep=action_timestep,
                              physics_timestep=physics_timestep,
                              device_idx=device_idx)
    elif env_type == 'gibson_sim2real':
        env = NavigateRandomEnvSim2Real(config_file=config_file,
                                        mode=env_mode,
                                        action_timestep=action_timestep,
                                        physics_timestep=physics_timestep,
                                        device_idx=device_idx,
                                        track=sim2real_track)
    elif env_type == 'gibson_meg':
        env = NavigateRandomInitEnvSim2Real(config_file=config_file,
                                        mode=env_mode,
                                        action_timestep=action_timestep,
                                        physics_timestep=physics_timestep,
                                        device_idx=device_idx,
                                        track=sim2real_track,
                                        random_init_m=random_init_m,
                                        seed=seed)
    else:
        assert False, 'unknown env_type: {}'.format(env_type)

    discount = env.discount_factor
    max_episode_steps = env.max_step

    return wrap_env(
        env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        time_limit_wrapper=wrappers.TimeLimit,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map,
        auto_reset=True
    )


@gin.configurable
def wrap_env(env,
             discount=1.0,
             max_episode_steps=0,
             gym_env_wrappers=(),
             time_limit_wrapper=wrappers.TimeLimit,
             env_wrappers=(),
             spec_dtype_map=None,
             auto_reset=True):
    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)
    env = gym_wrapper.GymWrapper(
        env,
        discount=discount,
        spec_dtype_map=spec_dtype_map,
        match_obs_space_dtype=True,
        auto_reset=auto_reset,
        simplify_box_bounds=True
    )

    if max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in env_wrappers:
        env = wrapper(env)

    return env

def get_train_models_wrapper():
    return get_train_models()
