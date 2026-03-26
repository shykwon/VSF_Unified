#!/bin/bash
# Weather Dataset Full Experiment
# 5 models × 3 seeds, horizon=12, missing_rate=0.75

LOG_DIR="logs/weather_exp"
COMMON="--dataset weather --seq_out 12 --missing_rate 0.75 --epochs 100 --config configs/gpu_a100_mig.yaml"

echo "============================================================"
echo "Weather Experiment 시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "설정: horizon=12, missing_rate=0.75, 21 nodes"
echo "============================================================"

# --- FDW (3 seeds) ---
for SEED in 42 123 456; do
    echo ""
    echo "[FDW] seed=$SEED 시작: $(date '+%H:%M:%S')"
    python3 scripts/train.py --model fdw --seed $SEED --batch_size 64 $COMMON \
        > "$LOG_DIR/fdw_weather_s${SEED}.log" 2>&1
    if [ $? -eq 0 ]; then
        MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/fdw_weather_seed${SEED}_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}\")" 2>/dev/null)
        echo "  완료: $MAE"
    else
        echo "  실패!"
    fi
done

# --- GinAR (3 seeds) ---
for SEED in 42 123 456; do
    echo ""
    echo "[GinAR] seed=$SEED 시작: $(date '+%H:%M:%S')"
    python3 scripts/train.py --model ginar --seed $SEED --batch_size 64 $COMMON \
        > "$LOG_DIR/ginar_weather_s${SEED}.log" 2>&1
    if [ $? -eq 0 ]; then
        MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/ginar_weather_seed${SEED}_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}\")" 2>/dev/null)
        echo "  완료: $MAE"
    else
        echo "  실패!"
    fi
done

# --- GIMCC (3 seeds) ---
for SEED in 42 123 456; do
    echo ""
    echo "[GIMCC] seed=$SEED 시작: $(date '+%H:%M:%S')"
    python3 scripts/train.py --model gimcc --seed $SEED --batch_size 256 $COMMON \
        > "$LOG_DIR/gimcc_weather_s${SEED}.log" 2>&1
    if [ $? -eq 0 ]; then
        MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/gimcc_weather_seed${SEED}_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}\")" 2>/dev/null)
        echo "  완료: $MAE"
    else
        echo "  실패!"
    fi
done

# --- CSDI (3 seeds) ---
for SEED in 42 123 456; do
    echo ""
    echo "[CSDI] seed=$SEED 시작: $(date '+%H:%M:%S')"
    python3 scripts/train.py --model csdi --seed $SEED --batch_size 16 $COMMON \
        > "$LOG_DIR/csdi_weather_s${SEED}.log" 2>&1
    if [ $? -eq 0 ]; then
        MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/csdi_weather_seed${SEED}_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}\")" 2>/dev/null)
        echo "  완료: $MAE"
    else
        echo "  실패!"
    fi
done

# --- SRDI (3 seeds) ---
for SEED in 42 123 456; do
    echo ""
    echo "[SRDI] seed=$SEED 시작: $(date '+%H:%M:%S')"
    python3 scripts/train.py --model srdi --seed $SEED --batch_size 8 $COMMON \
        > "$LOG_DIR/srdi_weather_s${SEED}.log" 2>&1
    if [ $? -eq 0 ]; then
        MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/srdi_weather_seed${SEED}_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}\")" 2>/dev/null)
        echo "  완료: $MAE"
    else
        echo "  실패!"
    fi
done

echo ""
echo "============================================================"
echo "Weather Experiment 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
