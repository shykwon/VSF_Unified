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

# 4. 전체 실험 (단일 GPU 순차 실행)
python scripts/train.py --model fdw --dataset metr-la --config configs/gpu_a100_mig.yaml
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

# 순차 실행 (단일 GPU)
python scripts/train.py --model fdw --dataset metr-la --config configs/gpu_a100_mig.yaml
# Background 실행: nohup python scripts/train.py ... &

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
| SRDI | Diffusion+MAML | ICLR 2025 | ~8-12GB |
| SAITS | Attention | - | ~3-4GB |
| GIMCC | Graph+Causal | KDD 2025 | ~4-6GB |

### SRDI Architecture (Full Implementation)

SRDI는 3개의 컴포넌트로 구성됨:
1. **CSDI_vsf**: Diffusion 기반 imputation 모듈
2. **gtnet**: MTGNN 기반 forecaster
3. **MAML**: Test-time adaptation을 위한 meta-learning

```
Training:  Input -> MAML(CSDI_vsf) -> Imputed -> gtnet -> Forecast
Test:      각 샘플마다 MAML adaptation 수행 후 예측
```

**의존성**: `learn2learn>=0.2.0` (MAML 구현)

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

## GPU Allocation (A100 MIG 20GB x1)

```
단일 GPU: 순차 실행
- FDW, GinAR: batch_size=64 (가벼운 모델)
- CSDI: batch_size=12 (Diffusion)
- SRDI: batch_size=8 (가장 무거움)
- SAITS, GIMCC: batch_size=32
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

## Experiment Design

실험 설계 문서: **[docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md)**

| 실험 | 목적 | Runs |
|------|------|------|
| 1. 재현성 + Oracle | Baseline + Upper Bound 확립 | 240 |
| 2. Imputation Ablation | Mean 대체 시 성능 비교 | 24 |
| 3. Semantic Ambiguity | Zero → Learnable token | 36 |
| 4. Missing Rate Robustness | 25/50/75/90% Sensor failure | 120 |
| 5. Computational Cost | 시간/메모리/파라미터 | 자동 |

**총 420 runs** | 모델: FDW, GinAR, CSDI, SRDI, GIMCC (SAITS 제외)

> SAITS는 imputation 전용 모델로 forecasting 미지원하여 실험에서 제외

---

## Current Status

### ✅ 완료
- 6개 모델 래퍼 구현 (FDW, GinAR, GIMCC, CSDI, SRDI, SAITS)
- 통합 Trainer (UnifiedTrainer)
- 통합 데이터 파이프라인
- Seed 관리 (재현성)
- TensorBoard 로깅
- 실험 설계 문서화

### ⚠️ 미완료
- Learnable token 구현 (GIMCC, GinAR) - 실험 3용
- Mean imputation 모드 - 실험 2용
- Cost logging - 실험 5용

### ✅ 최근 완료
- MAPE metric (`Metrics.MAPE`, `MaskedMetrics.MaskedMAPE`)
- Multi-horizon 지원 (`--seq_out 3/6/12/24`)
- Oracle mode (`--missing_rate 0`)
- Sensor failure 마스킹 (`--missing_rate 0.25/0.5/0.75/0.9 --missing_pattern sensor`)

## Important Notes

1. **Mask Semantics**: Dataset은 `mask=1`이 observed, SAITS는 `missing_mask=1`이 missing (wrapper에서 변환됨)

2. **Loss Function**: FDW/GinAR 원논문은 `masked_mae` 사용 (기본값으로 설정됨)

3. **Diffusion Models**: A100 MIG 20GB에서 CSDI는 batch_size=12, SRDI는 batch_size=8 권장

4. **External Code**: `external/` 디렉토리는 원본 코드이므로 수정 금지, wrapper로만 사용

---

## GPU Profile System

현재 서버: **A100 MIG 20GB** (단일 GPU)

```bash
# 현재 환경 (A100 MIG 20GB) - 기본값
python scripts/train.py --model fdw --gpu_profile a100

# 설정 파일 사용
python scripts/train.py --config configs/gpu_a100_mig.yaml --model fdw
```

| GPU Profile | hidden_dim | conv_channels | layers | batch_size |
|-------------|------------|---------------|--------|------------|
| 1080ti | 32 | 32 | 3 | 8-32 |
| a100 (MIG 20GB) | 64 | 64 | 5 | 8-64 |

설정 파일: `configs/gpu_a100_mig.yaml`

---

## ⚠️ Legacy: GTX 1080 Ti (11GB) 수정사항

아래는 `--gpu_profile 1080ti` 사용 시 자동 적용되는 설정입니다.

### 1. SRDI Full Implementation (Level 3)

SRDI는 이제 원본 구조를 완전히 재현:
- **CSDI_vsf**: Diffusion 기반 imputation
- **gtnet**: MTGNN 기반 forecaster
- **MAML**: Test-time adaptation (learn2learn 필요)

GPU Profile에 따라 diffusion config 자동 조정:

```python
# 4090/A100 (24GB+) - 논문 기본 설정
"diffusion": {
    "layers": 4,
    "channels": 64,
    "nheads": 8,
    "diffusion_embedding_dim": 32,  # base_forecasting.yaml
}

# 1080ti (11GB) - 축소 설정
"diffusion": {
    "layers": 2,
    "channels": 32,
    "nheads": 4,
    "diffusion_embedding_dim": 32,
}
```

**의존성**: `pip install learn2learn` (없으면 MAML 없이 동작)

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

### 3. GIMCC edge_weight (external/gimcc/models/graph/)

**문제**: PyTorch Geometric 2.3+ 에서 SAGEConv가 `edge_weight` 필수로 요구
**수정 1**: `graph_utils.py` - `create_pyg_data()`에 `edge_weight=torch.ones(num_edges)` 추가
**수정 2**: `subgraph_matching.py` - `SkipLastGNN.forward()`에서 edge_weight 추출 및 전달

```python
# graph_utils.py 수정
edge_weight = torch.ones(num_edges, dtype=torch.float)
data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)

# subgraph_matching.py 수정 (line 143-173)
edge_weight = getattr(data, 'edge_weight', None)  # 추가
x = self.convs[i](curr_emb, edge_index, edge_weight)  # edge_weight 전달
```

**원복 조건**: PyTorch Geometric < 2.3 사용 시 (선택적, 현재 코드도 호환됨)

---

### 원복 체크리스트

| 파일 | 수정 내용 | 원복 시점 |
|------|----------|----------|
| `src/models/srdi/wrapper.py` | diffusion config 축소 | 32GB+ GPU |
| `external/SRDI/Model.py` | compute() 메모리 최적화 | 32GB+ GPU (선택적) |
| `external/gimcc/.../graph_utils.py` | edge_weight 추가 | PyG < 2.3 (선택적) |
| `external/gimcc/.../subgraph_matching.py` | edge_weight 전달 | PyG < 2.3 (선택적) |

## Useful Commands

```python
# Python에서 사용
from src.core import set_seed, get_device, UnifiedTrainer
from src.data.loader import load_dataset

set_seed(42)
train_ds, val_ds, test_ds, scaler = load_dataset('metr-la', 'data/raw', mode='all')
```
