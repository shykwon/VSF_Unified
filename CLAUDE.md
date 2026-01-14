# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Variable Subset Forecasting (VSF) 연구 플랫폼. 다양한 시계열 예측 모델(FDW, GinAR, GIMCC, SRDI, CSDI, SAITS)을 통합 인터페이스로 실험할 수 있는 환경.

## Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run wrapper integration tests
python tests/test_wrappers.py
```

## Architecture

### Core Abstraction Layer (`src/core/`)

모든 모델과 데이터셋이 따르는 통합 인터페이스:

```
BaseVSFDataset.__getitem__() → Dict{'x', 'y', 'mask'}  # Shape: (Time, Node, Channel)
BaseVSFModel.forward(batch)  → Dict{'pred', ...}       # Shape: (Batch, Time, Node, Channel)
```

### Data Flow

```
Raw Data (T, N, C)
    ↓
WindowedVSFDataset (sliding window)
    ↓
DataLoader → batch Dict{'x': (B,T,N,C), 'y': (B,T,N,C), 'mask': (B,T,N,C)}
    ↓
ModelWrapper.forward(batch) → permute to model-specific format
    ↓
External Model (in external/)
    ↓
Output permute back → {'pred': (B, T_out, N, C)}
```

### Model Wrapper Pattern (`src/models/*/wrapper.py`)

각 래퍼는 external/ 의 원본 모델을 import하고, 입출력 shape을 통합 포맷으로 변환:

| Wrapper | External Path | Input Transform |
|---------|---------------|-----------------|
| FDWWrapper | `external/google_vsf_time_series` | `(B,T,N,C) → (B,C,N,T)` |
| GinARWrapper | `external/ginar` | Direct `(B,T,N,C)` |
| CSDIWrapper | `external/csdi` | Model-specific |
| SAITSWrapper | `external/saits` | Model-specific |

### Key Config Parameters

```python
config = {
    'num_nodes': 207,      # Number of variables/sensors
    'seq_in_len': 12,      # Input sequence length
    'seq_out_len': 12,     # Prediction horizon
    'in_dim': 1,           # Input feature dimension
    'hidden_dim': 32,      # Hidden layer size
    'adj_mx': None,        # Graph adjacency matrix (required for graph models)
    'device': 'cpu',
}
```

## External Models

`external/` 디렉토리는 원본 논문 코드를 그대로 보관. 수정하지 않고 wrapper를 통해 사용.

- **FDW** (KDD 2022): VSF 문제 최초 정의
- **GinAR** (KDD 2024): Variable Missing 시나리오
- **GIMCC** (KDD 2025): Graph Coupling
- **SRDI** (ICLR 2025): 최신 Imputation
- **CSDI**: Diffusion 기반 Imputation
- **SAITS**: Self-Attention 기반 Imputation

## Datasets

VSF 표준 데이터셋: METR-LA, PEMS-BAY, Solar, ETT 계열
- 데이터 shape: `(Time, Node, Channel)`
- Mask: 0/1 indicator (1=observed)
