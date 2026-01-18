#!/bin/bash
# Wait for SAITS to complete, then start GIMCC on GPU 0
# Usage: nohup ./scripts/start_gimcc_after_saits.sh &

SAITS_PID=4702
LOG_DIR="logs/final_exp"

echo "Waiting for SAITS (PID: $SAITS_PID) to complete..."

while ps -p $SAITS_PID > /dev/null 2>&1; do
    sleep 60
    echo "$(date): SAITS still running..."
done

echo "$(date): SAITS completed. Starting GIMCC..."

eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate tslib_env

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model gimcc \
    --dataset metr-la \
    --seeds 42,123,456 \
    --epochs 100 \
    --batch_size 16 \
    --device cuda:0 \
    --tensorboard \
    --log_dir $LOG_DIR

echo "$(date): GIMCC completed."
