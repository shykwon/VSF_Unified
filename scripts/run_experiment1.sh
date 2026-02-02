#!/bin/bash
#
# 실험 1: 재현성 검증 + Oracle
# 자동 연속 실행 스크립트
#
# 사용법:
#   chmod +x scripts/run_experiment1.sh
#   tmux new -s exp1
#   ./scripts/run_experiment1.sh
#   # Ctrl+B, D 로 분리 후 노트북 꺼도 됨
#

set -e

# 설정
MODELS="fdw ginar csdi srdi gimcc"
DATASETS="solar"  # 137 노드, 빠른 실험
SEEDS="42 123 456"
HORIZONS="3 6 12 24"
MISSING_RATES="0.0 0.25"  # Oracle + 기본
EPOCHS=100
LOG_DIR="logs/experiment1"
CONFIG="configs/gpu_a100_mig.yaml"  # 증가된 batch_size 적용
SKIP=6  # 이미 완료된 실험 수 (7번부터 시작)

# 로그 디렉토리 생성
mkdir -p $LOG_DIR

# 시작 시간
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "실험 1 시작: $START_TIME"
echo "============================================================"

# 총 실험 수 계산
TOTAL=0
for m in $MODELS; do
    for d in $DATASETS; do
        for s in $SEEDS; do
            for h in $HORIZONS; do
                for mr in $MISSING_RATES; do
                    TOTAL=$((TOTAL + 1))
                done
            done
        done
    done
done
echo "총 실험 수: $TOTAL"
echo ""

# 실험 실행
COUNT=0
for model in $MODELS; do
    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            for horizon in $HORIZONS; do
                for missing_rate in $MISSING_RATES; do
                    COUNT=$((COUNT + 1))

                    # 이미 완료된 실험 건너뛰기
                    if [ $COUNT -le $SKIP ]; then
                        continue
                    fi

                    # 실험 이름
                    EXP_NAME="${model}_${dataset}_h${horizon}_mr${missing_rate}_s${seed}"
                    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

                    echo "[$COUNT/$TOTAL] $EXP_NAME"
                    echo "  시작: $(date '+%H:%M:%S')"

                    # 실험 실행 (증가된 batch_size 적용)
                    python3 scripts/train.py \
                        --model $model \
                        --dataset $dataset \
                        --seed $seed \
                        --seq_out $horizon \
                        --missing_rate $missing_rate \
                        --epochs $EPOCHS \
                        --config $CONFIG \
                        > "$LOG_FILE" 2>&1

                    # 결과 추출
                    if grep -q "MAE:" "$LOG_FILE"; then
                        MAE=$(grep "MAE:" "$LOG_FILE" | tail -1 | awk '{print $2}')
                        echo "  완료: MAE=$MAE"
                    else
                        echo "  완료: (결과 확인 필요)"
                    fi
                    echo ""
                done
            done
        done
    done
done

# 완료
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "실험 1 완료!"
echo "시작: $START_TIME"
echo "종료: $END_TIME"
echo "로그: $LOG_DIR/"
echo "============================================================"
