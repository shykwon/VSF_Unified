#!/bin/bash
# Weather Dataset Quick Experiment - seed 42 only
# 5 models × 1 seed

LOG_DIR="logs/weather_exp"
COMMON="--dataset weather --seq_out 12 --missing_rate 0.75 --epochs 100 --config configs/gpu_a100_mig.yaml --seed 42"

echo "============================================================"
echo "Weather Quick Experiment: $(date '+%Y-%m-%d %H:%M:%S')"
echo "5 models x seed=42, horizon=12, missing_rate=0.75"
echo "============================================================"

echo ""
echo "[1/5] FDW 시작: $(date '+%H:%M:%S')"
python3 scripts/train.py --model fdw --batch_size 64 $COMMON > "$LOG_DIR/fdw_s42.log" 2>&1
if [ $? -eq 0 ]; then
    MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/fdw_weather_seed42_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}, RMSE={d['test_metrics']['RMSE']:.4f}\")" 2>/dev/null)
    echo "  FDW 완료: $MAE ($(date '+%H:%M:%S'))"
else
    echo "  FDW 실패!"
    tail -5 "$LOG_DIR/fdw_s42.log"
fi

echo ""
echo "[2/5] GinAR 시작: $(date '+%H:%M:%S')"
python3 scripts/train.py --model ginar --batch_size 64 $COMMON > "$LOG_DIR/ginar_s42.log" 2>&1
if [ $? -eq 0 ]; then
    MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/ginar_weather_seed42_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}, RMSE={d['test_metrics']['RMSE']:.4f}\")" 2>/dev/null)
    echo "  GinAR 완료: $MAE ($(date '+%H:%M:%S'))"
else
    echo "  GinAR 실패!"
    tail -5 "$LOG_DIR/ginar_s42.log"
fi

echo ""
echo "[3/5] GIMCC 시작: $(date '+%H:%M:%S')"
python3 scripts/train.py --model gimcc --batch_size 256 $COMMON > "$LOG_DIR/gimcc_s42.log" 2>&1
if [ $? -eq 0 ]; then
    MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/gimcc_weather_seed42_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}, RMSE={d['test_metrics']['RMSE']:.4f}\")" 2>/dev/null)
    echo "  GIMCC 완료: $MAE ($(date '+%H:%M:%S'))"
else
    echo "  GIMCC 실패!"
    tail -5 "$LOG_DIR/gimcc_s42.log"
fi

echo ""
echo "[4/5] CSDI 시작: $(date '+%H:%M:%S')"
python3 scripts/train.py --model csdi --batch_size 16 $COMMON > "$LOG_DIR/csdi_s42.log" 2>&1
if [ $? -eq 0 ]; then
    MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/csdi_weather_seed42_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}, RMSE={d['test_metrics']['RMSE']:.4f}\")" 2>/dev/null)
    echo "  CSDI 완료: $MAE ($(date '+%H:%M:%S'))"
else
    echo "  CSDI 실패!"
    tail -5 "$LOG_DIR/csdi_s42.log"
fi

echo ""
echo "[5/5] SRDI 시작: $(date '+%H:%M:%S')"
python3 scripts/train.py --model srdi --batch_size 8 $COMMON > "$LOG_DIR/srdi_s42.log" 2>&1
if [ $? -eq 0 ]; then
    MAE=$(python3 -c "import json,glob; f=sorted(glob.glob('logs/srdi_weather_seed42_*/results.json'))[-1]; d=json.load(open(f)); print(f\"MAE={d['test_metrics']['MAE']:.4f}, RMSE={d['test_metrics']['RMSE']:.4f}\")" 2>/dev/null)
    echo "  SRDI 완료: $MAE ($(date '+%H:%M:%S'))"
else
    echo "  SRDI 실패!"
    tail -5 "$LOG_DIR/srdi_s42.log"
fi

echo ""
echo "============================================================"
echo "완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
