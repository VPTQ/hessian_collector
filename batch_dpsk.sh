#!/bin/bash

CKPT_PATH=~/yangwang/DeepSeek-V3-mp4/
CONFIG_PATH=~/yangwang/DeepSeek-V3/inference/configs/config_671B.json

for seed in {0..20}
do
    echo "Starting run with seed ${seed}"
    
    torchrun --nnodes 1 --nproc-per-node 4 deepseek_hessian_collector.py \
        --ckpt_path ${CKPT_PATH} \
        --config ${CONFIG_PATH} \
        --save_path Hessians-Deepseek-V3-8192-256-seed-${seed} \
        --max_samples 256 \
        --ctx_size 8192 \
        --sample_proc 1 \
        --seed ${seed}
        
    echo "Completed run with seed ${seed}"
    
	sleep 10
done

echo "All runs completed!"