#!/bin/bash
#
# Quick Experiment: missing_rate=0.75, horizon=12, all models
# 총 15개 실험 (5 models × 3 seeds)
#

set -e

# 설정
MODELS="fdw ginar csdi srdi gimcc"
DATASET="solar"
SEEDS="42 123 456"
HORIZON=12
MISSING_RATE=0.75
EPOCHS=100
LOG_DIR="logs/quick_exp"
CONFIG="configs/gpu_a100_mig.yaml"

# 로그 디렉토리 생성
mkdir -p $LOG_DIR

# 시작 시간
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "Quick Experiment 시작: $START_TIME"
echo "설정: horizon=$HORIZON, missing_rate=$MISSING_RATE"
echo "============================================================"

# 총 실험 수
TOTAL=15
COUNT=0

for model in $MODELS; do
    for seed in $SEEDS; do
        COUNT=$((COUNT + 1))

        EXP_NAME="${model}_${DATASET}_h${HORIZON}_mr${MISSING_RATE}_s${seed}"
        LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

        echo "[$COUNT/$TOTAL] $EXP_NAME"
        echo "  시작: $(date '+%H:%M:%S')"

        # 실험 실행
        python3 scripts/train.py \
            --model $model \
            --dataset $DATASET \
            --seed $seed \
            --seq_out $HORIZON \
            --missing_rate $MISSING_RATE \
            --epochs $EPOCHS \
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
            echo "  완료: (결과 확인 필요)"
        fi
        echo ""
    done
done

# 완료
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "Quick Experiment 완료!"
echo "시작: $START_TIME"
echo "종료: $END_TIME"
echo "로그: $LOG_DIR/"
echo "============================================================"
