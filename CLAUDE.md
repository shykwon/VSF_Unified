# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Variable Subset Forecasting (VSF) 연구 플랫폼. 다양한 시계열 예측 모델(FDW, GinAR, GIMCC, SRDI, CSDI, SAITS)을 통합 인터페이스로 실험할 수 있는 환경.

**목표**: 6개 모델을 동일한 조건에서 공정하게 비교 실험

## Quick Start (신규 서버)

```bash
# 1. 셋업 (venv 생성 + 패키지 설치 + GPU 확인)
chmod +x scripts/setup_server.sh
./scripts/setup_server.sh

# 2. 데이터셋 다운로드
source venv/bin/activate
python scripts/download_datasets.py

# 3. 테스트 실행 (debug 모드)
python scripts/train.py --model fdw --dataset metr-la --debug

# 4. 전체 실험 (3 GPU 병렬)
python scripts/run_parallel.py --dry-run  # 명령어 확인
python scripts/run_parallel.py            # 실제 실행
```

## Commands

```bash
# 단일 모델 학습
python scripts/train.py --model fdw --dataset metr-la --epochs 100

# Multi-seed 실험
python scripts/train.py --model fdw --seeds 42,123,456 --dataset metr-la

# YAML config 사용
python scripts/train.py --config configs/default.yaml

# TensorBoard
python scripts/train.py --model fdw --dataset metr-la --tensorboard
tensorboard --logdir logs/

# 병렬 실행 (3 GPU)
python scripts/run_parallel.py                    # Interactive
./scripts/run_nohup.sh                            # Background (SSH 끊어도 실행)

# Wrapper 테스트
python tests/test_wrappers.py
```

## Project Structure

```
research_VSF/
├── src/
│   ├── core/                    # 핵심 추상화 레이어
│   │   ├── model.py             # BaseVSFModel
│   │   ├── dataset.py           # BaseVSFDataset, WindowedVSFDataset
│   │   ├── trainer.py           # UnifiedTrainer
│   │   ├── metrics.py           # Metrics, MaskedMetrics
│   │   └── utils.py             # set_seed, get_device
│   ├── models/                  # 모델 래퍼들
│   │   ├── fdw/wrapper.py
│   │   ├── ginar/wrapper.py
│   │   ├── gimcc/wrapper.py
│   │   ├── csdi/wrapper.py
│   │   ├── srdi/wrapper.py
│   │   └── saits/wrapper.py
│   └── data/
│       ├── loader.py            # load_dataset()
│       └── scaler.py            # StandardScaler
├── external/                    # 원본 모델 코드 (수정 금지)
├── scripts/
│   ├── train.py                 # 메인 학습 스크립트
│   ├── run_parallel.py          # 병렬 실행
│   ├── run_nohup.sh             # 백그라운드 실행
│   ├── setup_server.sh          # 서버 셋업
│   └── download_datasets.py     # 데이터 다운로드
├── configs/                     # YAML 설정 파일
├── data/raw/                    # 데이터셋 저장 위치
└── logs/                        # 실험 결과
```

## Architecture

### Unified Interface

```
BaseVSFDataset.__getitem__() → Dict{'x', 'y', 'mask'}  # Shape: (Time, Node, Channel)
BaseVSFModel.forward(batch)  → Dict{'pred', 'loss'?}   # Shape: (Batch, Time, Node, Channel)
```

### Data Flow

```
Raw Data (T, N, C)
    ↓ load_dataset() + StandardScaler
WindowedVSFDataset (sliding window)
    ↓ DataLoader
batch Dict{'x': (B,T,N,C), 'y': (B,T,N,C), 'mask': (B,T,N,C)}
    ↓ ModelWrapper.forward(batch)
External Model (in external/)
    ↓
Output → {'pred': (B, T_out, N, C), 'loss': optional}
```

### Loss Function Strategy

| 모델 | Loss 계산 위치 | Trainer 동작 |
|------|---------------|--------------|
| CSDI, SRDI, SAITS, GIMCC | 내부 (forward에서 반환) | `output['loss']` 사용 |
| FDW, GinAR | 외부 (Trainer에서 계산) | `masked_mae(pred, true)` |

## Models

| Model | Type | Paper | GPU Memory |
|-------|------|-------|------------|
| FDW | Forecasting | KDD 2022 | ~2-3GB |
| GinAR | Forecasting | KDD 2024 | ~2-3GB |
| CSDI | Diffusion | - | ~6-8GB |
| SRDI | Diffusion | ICLR 2025 | ~6-8GB |
| SAITS | Attention | - | ~3-4GB |
| GIMCC | Graph+Causal | KDD 2025 | ~4-6GB |

## Key Config Parameters

```python
config = {
    'num_nodes': 207,      # Number of sensors/variables
    'seq_in_len': 12,      # Input sequence length
    'seq_out_len': 12,     # Prediction horizon
    'in_dim': 1,           # Input feature dimension
    'hidden_dim': 32,      # Hidden layer size
    'device': 'cuda',      # or 'cpu'
    'loss_fn': 'masked_mae',  # or 'mse'
}
```

## GPU Allocation (GTX 1080 x3)

```
GPU 0: FDW + GinAR     (batch_size=32, 가벼운 모델)
GPU 1: CSDI           (batch_size=8, Diffusion)
GPU 2: SRDI + SAITS   (batch_size=8~16)
```

## Experiment Output

```
logs/
└── fdw_metr-la_seed42_20250114_160000/
    ├── config.json       # 실험 설정 (재현용)
    ├── best_model.pth    # 최적 모델 체크포인트
    ├── results.json      # Test 결과 (MAE, RMSE, MAPE)
    ├── train_log.csv     # 에폭별 기록
    └── tensorboard/      # TensorBoard 로그
```

## Current Status

### ✅ 완료
- 6개 모델 래퍼 구현 (FDW, GinAR, GIMCC, CSDI, SRDI, SAITS)
- 통합 Trainer (UnifiedTrainer)
- 통합 데이터 파이프라인
- Seed 관리 (재현성)
- 병렬 실행 스크립트
- TensorBoard 로깅

### ⚠️ 미완료
- VSF 시나리오별 마스킹 전략

## Important Notes

1. **Mask Semantics**: Dataset은 `mask=1`이 observed, SAITS는 `missing_mask=1`이 missing (wrapper에서 변환됨)

2. **Loss Function**: FDW/GinAR 원논문은 `masked_mae` 사용 (기본값으로 설정됨)

3. **Diffusion Models**: CSDI/SRDI는 batch_size를 8로 낮춰야 8GB GPU에서 동작

4. **External Code**: `external/` 디렉토리는 원본 코드이므로 수정 금지, wrapper로만 사용

---

## ⚠️ Temporary Modifications (GTX 1080 Ti 11GB 환경)

아래 수정사항은 **GTX 1080 Ti (11GB)** 환경에서 OOM/호환성 문제 해결을 위해 적용됨.
**32GB+ GPU** 또는 **더 나은 환경**에서는 원복을 권장.

### 1. SRDI Wrapper Config (src/models/srdi/wrapper.py:63-82)

**문제**: SRDI Diffusion 모델이 기본 설정으로 OOM 발생
**수정**: config 축소 (성능 저하 가능성 있음)

```python
# 원본 (고성능 GPU용)
"model": {
    "timeemb": 128,
    "featureemb": 16,
},
"diffusion": {
    "layers": 4,
    "channels": 64,
    "nheads": 8,
    "diffusion_embedding_dim": 128,
}

# 현재 (11GB GPU용)
"model": {
    "timeemb": 64,       # Reduced
    "featureemb": 8,     # Reduced
},
"diffusion": {
    "layers": 2,         # Reduced from 4
    "channels": 32,      # Reduced from 64
    "nheads": 4,         # Reduced from 8
    "diffusion_embedding_dim": 64,  # Reduced
}
```

**원복 조건**: 32GB+ VRAM GPU 사용 시

---

### 2. SRDI compute() Function (external/SRDI/Model.py:102-138)

**문제**: `correlation_matrix = torch.zeros(x, y, y)` 가 (batch×seq, 207, 207) 크기로 ~10GB 메모리 사용
**수정**: 전체 matrix 저장 대신 이전 timestep만 유지하여 on-the-fly 계산

```python
# 원본 (메모리 비효율)
correlation_matrix = torch.zeros(x, y, y).to(device)
for i in range(x):
    correlation_matrix[i] = cosine_similarity(...)
for i in range(len(correlation_matrix)):
    loss += torch.abs(correlation_matrix[i] - correlation_matrix[i-1])

# 현재 (메모리 효율)
prev_corr = None
for i in range(x):
    curr_corr = cosine_similarity(...)
    if prev_corr is not None:
        loss += torch.abs(curr_corr - prev_corr)
    prev_corr = curr_corr
```

**원복 조건**: 32GB+ VRAM GPU 사용 시 (원본이 약간 더 빠를 수 있음)

---

### 3. GIMCC edge_weight (external/gimcc/models/graph/graph_utils.py:6-31)

**문제**: PyTorch Geometric 2.3+ 에서 SAGEConv가 `edge_weight` 필수로 요구
**수정**: `create_pyg_data()`에 `edge_weight=torch.ones(num_edges)` 추가

```python
# 원본
data = Data(x=node_features, edge_index=edge_index)

# 현재
edge_weight = torch.ones(num_edges, dtype=torch.float)
data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
```

**원복 조건**: PyTorch Geometric < 2.3 사용 시 (선택적, 현재 코드도 호환됨)

---

### 원복 체크리스트

| 파일 | 수정 내용 | 원복 시점 |
|------|----------|----------|
| `src/models/srdi/wrapper.py` | diffusion config 축소 | 32GB+ GPU |
| `external/SRDI/Model.py` | compute() 메모리 최적화 | 32GB+ GPU (선택적) |
| `external/gimcc/.../graph_utils.py` | edge_weight 추가 | PyG < 2.3 (선택적) |

## Useful Commands

```python
# Python에서 사용
from src.core import set_seed, get_device, UnifiedTrainer
from src.data.loader import load_dataset

set_seed(42)
train_ds, val_ds, test_ds, scaler = load_dataset('metr-la', 'data/raw', mode='all')
```
