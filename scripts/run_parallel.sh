#!/bin/bash
#
# VSF Research Platform - Parallel Experiment Runner (Bash)
#
# Usage:
#   chmod +x scripts/run_parallel.sh
#   ./scripts/run_parallel.sh
#
# 3개 GPU에서 모든 모델을 병렬 실행

set -e

# ============================================================================
# Configuration
# ============================================================================
SEEDS="42,123,456"
EPOCHS=100
DATASET="metr-la"
LOG_DIR="logs/parallel_$(date +%Y%m%d_%H%M%S)"

# ============================================================================
# GPU 0: FDW + GinAR (가벼운 모델)
# ============================================================================
run_gpu0() {
    echo "[GPU 0] Starting FDW..."
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --model fdw \
        --dataset $DATASET \
        --seeds $SEEDS \
        --epochs $EPOCHS \
        --batch_size 32 \
        --log_dir $LOG_DIR \
        --tensorboard

    echo "[GPU 0] Starting GinAR..."
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --model ginar \
        --dataset $DATASET \
        --seeds $SEEDS \
        --epochs $EPOCHS \
        --batch_size 32 \
        --log_dir $LOG_DIR \
        --tensorboard

    echo "[GPU 0] Done!"
}

# ============================================================================
# GPU 1: CSDI (Diffusion - 단독)
# ============================================================================
run_gpu1() {
    echo "[GPU 1] Starting CSDI..."
    CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
        --model csdi \
        --dataset $DATASET \
        --seeds $SEEDS \
        --epochs $EPOCHS \
        --batch_size 8 \
        --log_dir $LOG_DIR \
        --tensorboard

    echo "[GPU 1] Done!"
}

# ============================================================================
# GPU 2: SRDI + SAITS
# ============================================================================
run_gpu2() {
    echo "[GPU 2] Starting SRDI..."
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
        --model srdi \
        --dataset $DATASET \
        --seeds $SEEDS \
        --epochs $EPOCHS \
        --batch_size 8 \
        --log_dir $LOG_DIR \
        --tensorboard

    echo "[GPU 2] Starting SAITS..."
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
        --model saits \
        --dataset $DATASET \
        --seeds $SEEDS \
        --epochs $EPOCHS \
        --batch_size 16 \
        --log_dir $LOG_DIR \
        --tensorboard

    echo "[GPU 2] Done!"
}

# ============================================================================
# Main
# ============================================================================
echo "============================================================"
echo "    VSF Parallel Experiment Runner"
echo "============================================================"
echo "  Dataset: $DATASET"
echo "  Seeds:   $SEEDS"
echo "  Epochs:  $EPOCHS"
echo "  Log dir: $LOG_DIR"
echo "============================================================"
echo ""

mkdir -p $LOG_DIR

echo "Starting parallel experiments on 3 GPUs..."
echo ""

# 3개 GPU에서 병렬 실행 (&로 백그라운드 실행, wait로 대기)
run_gpu0 > "$LOG_DIR/gpu0.log" 2>&1 &
PID0=$!

run_gpu1 > "$LOG_DIR/gpu1.log" 2>&1 &
PID1=$!

run_gpu2 > "$LOG_DIR/gpu2.log" 2>&1 &
PID2=$!

echo "Running experiments..."
echo "  GPU 0 PID: $PID0 (FDW, GinAR)"
echo "  GPU 1 PID: $PID1 (CSDI)"
echo "  GPU 2 PID: $PID2 (SRDI, SAITS)"
echo ""
echo "Logs: $LOG_DIR/gpu*.log"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_DIR/gpu0.log"
echo "  tail -f $LOG_DIR/gpu1.log"
echo "  tail -f $LOG_DIR/gpu2.log"
echo ""

# 모든 프로세스 완료 대기
wait $PID0
echo "✅ GPU 0 completed"

wait $PID1
echo "✅ GPU 1 completed"

wait $PID2
echo "✅ GPU 2 completed"

echo ""
echo "============================================================"
echo "    All experiments completed!"
echo "    Results in: $LOG_DIR"
echo "============================================================"
