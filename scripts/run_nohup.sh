#!/bin/bash
#
# VSF Research Platform - Background Experiment Runner
#
# 원격 서버에서 SSH 연결 끊어도 계속 실행되는 버전
#
# Usage:
#   chmod +x scripts/run_nohup.sh
#   ./scripts/run_nohup.sh
#
#   # 진행 상황 확인
#   tail -f logs/parallel_*/gpu*.log
#
#   # 프로세스 확인
#   ps aux | grep train.py

set -e

SEEDS="42,123,456"
EPOCHS=100
DATASET="metr-la"
LOG_DIR="logs/parallel_$(date +%Y%m%d_%H%M%S)"

mkdir -p $LOG_DIR

echo "============================================================"
echo "    VSF Background Experiment Runner"
echo "============================================================"
echo "  Log dir: $LOG_DIR"
echo "============================================================"

# GPU 0: FDW + GinAR
nohup bash -c "
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model fdw --dataset $DATASET --seeds $SEEDS --epochs $EPOCHS --batch_size 32 --log_dir $LOG_DIR --tensorboard && \
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model ginar --dataset $DATASET --seeds $SEEDS --epochs $EPOCHS --batch_size 32 --log_dir $LOG_DIR --tensorboard
" > "$LOG_DIR/gpu0.log" 2>&1 &
echo "Started GPU 0 (FDW, GinAR) - PID: $!"

# GPU 1: CSDI
nohup bash -c "
    CUDA_VISIBLE_DEVICES=1 python scripts/train.py --model csdi --dataset $DATASET --seeds $SEEDS --epochs $EPOCHS --batch_size 8 --log_dir $LOG_DIR --tensorboard
" > "$LOG_DIR/gpu1.log" 2>&1 &
echo "Started GPU 1 (CSDI) - PID: $!"

# GPU 2: SRDI + SAITS
nohup bash -c "
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py --model srdi --dataset $DATASET --seeds $SEEDS --epochs $EPOCHS --batch_size 8 --log_dir $LOG_DIR --tensorboard && \
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py --model saits --dataset $DATASET --seeds $SEEDS --epochs $EPOCHS --batch_size 16 --log_dir $LOG_DIR --tensorboard
" > "$LOG_DIR/gpu2.log" 2>&1 &
echo "Started GPU 2 (SRDI, SAITS) - PID: $!"

echo ""
echo "============================================================"
echo "All experiments started in background!"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/gpu0.log"
echo "  tail -f $LOG_DIR/gpu1.log"
echo "  tail -f $LOG_DIR/gpu2.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep train.py"
echo ""
echo "You can now safely disconnect from SSH."
echo "============================================================"
