#!/bin/bash

# AMR-Changes: This file is modified from TF-agents gibson_sim2real branch:
# agents/tf_agents/agents/sac/examples/v1/train_single_env.sh

gpu_c="0"
gpu_g="0"
robot="turtlebot"

if [[ -z "${CONFIG_FILE}" ]]; then
  config_file="../examples/configs/"$robot"_AMR.yaml"
else
  config_file="${CONFIG_FILE}"
fi

if [[ -z "${SIM2REAL_TRACK}" ]]; then
  sim2real_track="static" #make 'interactive' for objects to appear in hallway
else
  sim2real_track="${SIM2REAL_TRACK}"
fi

if [[ -z "${LOG_DIR}" ]]; then
  log_dir="test"
else
  log_dir="${LOG_DIR}"
fi

echo "config_file:" $config_file
echo "sim2real_track:" $sim2real_track
echo "log_dir:" $log_dir

python -u train_eval_rnn_m_reinforce.py \
    --root_dir $log_dir \
    --env_type gibson_m \
    --sim2real_track $sim2real_track \
    --config_file $config_file \
    --collect_episodes_per_iteration 1 \
    --learning_rate 1e-4 \
    --train_steps_per_iteration 1 \
    --replay_buffer_capacity 1000 \
    --num_eval_episodes 10 \
    --actor_rnn_size 100 \
    --random_init_m 2 \
    --seed 0 \
    --env_mode 'headless' \
    --eval_interval 50 \
    --AMR_regularizer 0 \
    --gpu_c $gpu_c \
    --gpu_g $gpu_g \
    --num_parallel_environments 3
