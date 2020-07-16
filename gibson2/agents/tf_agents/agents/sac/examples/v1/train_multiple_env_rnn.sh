#!/bin/bash

gpu_c="0"
gpu_g="0"
robot="locobot"

if [[ -z "${CONFIG_FILE}" ]]; then
  config_file="../examples/configs/"$robot"_p2p_nav_house.yaml"
else
  config_file="${CONFIG_FILE}"
fi

if [[ -z "${SIM2REAL_TRACK}" ]]; then
  sim2real_track="static"
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

python -u train_eval_rnn.py \
    --root_dir $log_dir \
    --reload_interval 5000 \
    --env_type gibson_sim2real \
    --sim2real_track $sim2real_track \
    --config_file $config_file \
    --initial_collect_episodes 1 \
    --collect_episodes_per_iteration 1 \
    --batch_size 64 \
    --train_steps_per_iteration 1 \
    --replay_buffer_capacity 10000 \
    --num_eval_episodes 10 \
    --eval_interval 10000000 \
    --gpu_c $gpu_c \
    --gpu_g $gpu_g \
    --num_parallel_environments 2
