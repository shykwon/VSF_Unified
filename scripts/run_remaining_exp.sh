#!/bin/bash
#
# Remaining Experiments: CSDI, SRDI, GIMCC only
# FDW, GinAR 완료됨 - 제외
# 모델별 batch_size 최적화
#

set -e

# MIG 환경 설정 - NVML 에러 해결
export CUDA_VISIBLE_DEVICES="MIG-0d6f4ada-0e40-5207-8a91-9a379ae60bd2"
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

DATASET="solar"
SEEDS="42 123 456"
HORIZON=12
MISSING_RATE=0.75
EPOCHS=100
LOG_DIR="logs/quick_exp"
CONFIG="configs/gpu_a100_mig.yaml"

mkdir -p $LOG_DIR

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "Remaining Experiments 시작: $START_TIME"
echo "모델: CSDI, SRDI, GIMCC (FDW, GinAR 완료)"
echo "============================================================"

# 모델별 batch_size 설정
declare -A BATCH_SIZES
# CSDI/SRDI: MIG 20GB에서 OOM 발생 - 나중에 별도 실행
# BATCH_SIZES[csdi]=8
# BATCH_SIZES[srdi]=4
BATCH_SIZES[gimcc]=256  # GPU 15% -> 더 높게

MODELS="gimcc"
TOTAL=3
COUNT=0

for model in $MODELS; do
    BATCH_SIZE=${BATCH_SIZES[$model]}

    for seed in $SEEDS; do
        COUNT=$((COUNT + 1))

        EXP_NAME="${model}_${DATASET}_h${HORIZON}_mr${MISSING_RATE}_s${seed}"
        LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

        echo "[$COUNT/$TOTAL] $EXP_NAME (batch_size=$BATCH_SIZE)"
        echo "  시작: $(date '+%H:%M:%S')"

        # 실험 실행 (batch_size override)
        python3 scripts/train.py \
            --model $model \
            --dataset $DATASET \
            --seed $seed \
            --seq_out $HORIZON \
            --missing_rate $MISSING_RATE \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --config $CONFIG \
            > "$LOG_FILE" 2>&1

        # 결과 추출
        if grep -q "ObservedMAE:" "$LOG_FILE"; then
            OBS_MAE=$(grep "ObservedMAE:" "$LOG_FILE" | tail -1 | awk '{print $2}')
            MAE=$(grep "MAE (Full):" "$LOG_FILE" | tail -1 | awk '{print $3}')
            echo "  완료: MAE=$MAE, ObservedMAE=$OBS_MAE"
        elif grep -q "MAE:" "$LOG_FILE"; then
            MAE=$(grep "MAE:" "$LOG_FILE" | tail -1 | awk '{print $2}')
            echo "  완료: MAE=$MAE"
        else
            echo "  에러 발생 - 로그 확인: $LOG_FILE"
        fi
        echo ""
    done
done

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "Remaining Experiments 완료!"
echo "시작: $START_TIME"
echo "종료: $END_TIME"
echo "로그: $LOG_DIR/"
echo "============================================================"
