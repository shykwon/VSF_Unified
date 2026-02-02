# VSF 실험 설계 문서 (v2)

> 최종 수정: 2026-01-31

## 연구 목표

**Research Question:**
> "VSF 모델들의 결측치 처리 방식은 실제로 효과적인가? 더 나은 대안은 무엇인가?"

**목적:** 기존 VSF 논문들의 허점을 찾고, 개선 방향을 제시하여 연구 주제 도출

---

## 공통 설정

| 항목 | 값 |
|------|-----|
| **모델** | FDW, GinAR, CSDI, SRDI, GIMCC (5개) |
| **데이터셋** | METR-LA, PEMS-BAY |
| **Seeds** | 42, 123, 456 |
| **Horizons** | 3, 6, 12, 24 step |
| **Metrics** | Masked MAE, Full MAE, Full RMSE, Full MAPE |
| **GPU** | A100 MIG 20GB (단일) |

> **Note:** SAITS는 imputation 전용 모델로 forecasting (horizon) 미지원하여 제외

---

## 실험 1: 재현성 검증 + Oracle (Baseline)

### 목적
- 기존 모델들의 논문 보고 성능 재현
- **Oracle (Upper Bound)** 설정: 결측 없을 때의 이론적 최대 성능
- 모든 후속 실험의 기준점(baseline) 확립

### 실험 1-a: Oracle (Upper Bound)
```
Missing Rate = 0% (결측 없음)
→ 모델의 순수 예측 능력만 측정
→ 이론적 상한선 (완벽한 imputation 시 도달 가능한 성능)
```

### 실험 1-b: 논문 기본 설정
```
Missing Rate = 논문 기본값
→ 기존 논문 성능 재현
→ Oracle 대비 Gap 측정
```

### 설정
| 항목 | 값 |
|------|-----|
| Seeds | 42, 123, 456 |
| Horizons | 3, 6, 12, 24 |
| Missing Rate | 0% (Oracle), 논문 기본값 |

### Metrics
| Metric | 설명 | 수식 |
|--------|------|------|
| Masked MAE | 결측 위치만 평가 | `mean(\|pred - true\| * mask)` |
| Full MAE | 전체 위치 평가 | `mean(\|pred - true\|)` |
| Full RMSE | 전체 위치 평가 | `sqrt(mean((pred - true)^2))` |
| Full MAPE | 전체 위치 평가 | `mean(\|pred - true\| / \|true\|) * 100` |

### Oracle Gap 계산
```python
# Oracle 대비 성능 gap (낮을수록 좋음)
Oracle_Gap (%) = (Model_MAE - Oracle_MAE) / Oracle_MAE * 100

# 예시: Oracle MAE = 2.5, Model MAE = 3.0
# Gap = (3.0 - 2.5) / 2.5 * 100 = 20%
# → "결측 처리로 인해 20% 성능 저하"
```

### Output
```
모델별 × 데이터셋별 × Horizon별:
- Oracle 성능 (Mean ± Std)
- 논문 설정 성능 (Mean ± Std)
- Oracle Gap (%)
- 논문 보고 성능과의 차이
```

### 실험 수
```
5 모델 × 2 데이터셋 × 3 seeds × 4 horizons × 2 conditions = 240 runs
```

---

## 실험 2: Imputation Ablation Study

### 목적
- 복잡한 imputation 모듈이 실제로 얼마나 기여하는지 정량화
- "단순 mean으로 대체해도 충분한가?" 검증

### 대상 모델
| 모델 | 원본 Imputation 방식 |
|------|---------------------|
| CSDI | Diffusion-based |
| SRDI | Diffusion + MAML |

### 실험 2-a: Mean 대체
```python
# 원본: 복잡한 imputation
imputed = model.impute(x, mask)

# 대체: Feature-wise mean
feature_mean = x[mask == 1].mean(dim=0)  # 노드별 평균
imputed = x.clone()
imputed[mask == 0] = feature_mean
```

### 실험 2-b: Contribution 정량화
```
Δ Performance = (Mean 대체 MAE) - (원본 MAE)
Contribution (%) = Δ Performance / (Mean 대체 MAE) * 100
```

### 가설
> H1: 복잡한 imputation은 단순 mean 대비 유의미한 성능 향상을 제공한다
> H0: 성능 차이가 없거나 미미하다 (< 5%)

### 실험 수
```
2 모델 × 2 데이터셋 × 3 seeds × 2 conditions = 24 runs
```

---

## 실험 3: Semantic Ambiguity 해결

### 목적
- Zero filling의 근본적 문제 검증 (0 = 실제값? 결측?)
- Learnable token으로 "결측"의 의미를 학습하게 하여 성능 개선

### 대상 모델
| 모델 | 현재 방식 |
|------|----------|
| GIMCC | Zero filling |
| GinAR | Zero filling |

### 방법: Learnable Missing Token
```python
class ImprovedModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 단일 learnable token (Option A)
        self.missing_token = nn.Parameter(torch.randn(1, hidden_dim))

    def forward(self, x, mask):
        # 결측 위치에 learnable token 삽입
        x_filled = x.clone()
        x_filled[mask == 0] = self.missing_token
        return self.backbone(x_filled)
```

### 비교군
| 방식 | 설명 |
|------|------|
| Zero | 기존 (0으로 채움) |
| Mean | Feature-wise mean |
| Learnable | 학습 가능한 단일 토큰 |

### 가설
> H1: Learnable token이 zero/mean filling 대비 성능이 우수하다
> H2: 모델이 "결측"의 semantic을 학습하여 더 robust해진다

### 실험 수
```
2 모델 × 2 데이터셋 × 3 seeds × 3 methods = 36 runs
```

---

## 실험 4: Missing Rate Robustness

### 목적
- 다양한 결측률에서 모델의 robustness 검증
- 성능이 급격히 저하되는 임계점(critical point) 식별
- 기존 논문들이 다루지 않는 극단적 상황 테스트

### Missing Rate
| Rate | 설명 |
|------|------|
| 25% | 경미한 결측 |
| 50% | 중간 수준 |
| 75% | 심각한 결측 |
| 90% | 극단적 상황 |

### Missing Pattern: Sensor Failure
```python
def generate_sensor_failure_mask(num_nodes, missing_rate, seq_len):
    """
    특정 노드가 전 시간대에서 완전히 결측되는 패턴
    실제 센서 고장 상황을 시뮬레이션
    """
    num_missing_nodes = int(num_nodes * missing_rate)
    missing_nodes = np.random.choice(num_nodes, num_missing_nodes, replace=False)

    mask = np.ones((seq_len, num_nodes))
    mask[:, missing_nodes] = 0  # 선택된 노드는 전 시간대 결측

    return mask

# 예시: METR-LA (207 nodes)
# 25% → 52개 노드 완전 결측
# 50% → 104개 노드 완전 결측
# 75% → 155개 노드 완전 결측
# 90% → 186개 노드 완전 결측
```

### 분석 포인트
1. **Degradation Curve:** Missing rate 증가에 따른 성능 저하 곡선
2. **Critical Point:** 성능이 급격히 무너지는 임계점
3. **Model Ranking 변화:** 낮은 rate vs 높은 rate에서 순위 변동
4. **Oracle Gap 확대:** Missing rate에 따른 Oracle 대비 gap 변화

### 실험 수
```
5 모델 × 2 데이터셋 × 3 seeds × 4 rates = 120 runs
```

---

## 실험 5: Computational Cost

### 목적
- 성능뿐 아니라 효율성 관점에서 모델 비교
- "성능 1% 향상에 10배 비용이면 실용적인가?" 분석

### Metrics
| Metric | 측정 방법 | 단위 |
|--------|----------|------|
| **Train Time** | 1 epoch 평균 | 초 (s) |
| **Inference Time** | Test set 전체 추론 | 초 (s) |
| **Peak Memory** | 학습 중 최대 GPU 메모리 | GB |
| **Parameters** | 학습 가능한 파라미터 수 | M (백만) |

### 측정 코드
```python
import time
import torch

# 학습 시간
start = time.time()
trainer.train_epoch()
train_time = time.time() - start

# 추론 시간
start = time.time()
with torch.no_grad():
    predictions = model.predict(test_loader)
inference_time = time.time() - start

# GPU 메모리
peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB

# 파라미터 수
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
```

### Output
```
모델별 Cost-Performance Trade-off 분석
- Pareto frontier 시각화
- 효율성 점수: Performance / Cost
```

---

## 실험 스토리라인

```
┌─────────────────────────────────────────────────────────────┐
│  실험 1: 재현성 검증 + Oracle                                  │
│  → Oracle (Upper Bound): 결측 없을 때 이론적 최대 성능          │
│  → Baseline 확립, 논문 성능 검증                              │
│  → Oracle Gap 정량화: "결측으로 인한 성능 손실"                 │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  실험 2: Imputation Ablation                                 │
│  → "복잡한 imputation이 정말 필요한가?"                        │
│  → Mean 대체 시 Oracle Gap 변화 분석                          │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  실험 3: Semantic Ambiguity                                  │
│  → Zero filling의 문제점 검증                                 │
│  → Learnable token으로 Oracle Gap 축소 가능성                 │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  실험 4: Missing Rate Robustness                             │
│  → 극단적 상황에서 Oracle Gap 확대 패턴 분석                    │
│  → 기존 논문들의 미검증 영역                                   │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  실험 5: Computational Cost                                  │
│  → 성능 vs 효율성 trade-off                                   │
│  → Oracle Gap 축소 대비 비용 분석                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 총 실험 수

| 실험 | 구성 | Runs |
|------|------|------|
| 실험 1 | 5 모델 × 2 데이터셋 × 3 seeds × 4 horizons × 2 (Oracle + 기본) | 240 |
| 실험 2 | 2 모델 × 2 데이터셋 × 3 seeds × 2 conditions | 24 |
| 실험 3 | 2 모델 × 2 데이터셋 × 3 seeds × 3 methods | 36 |
| 실험 4 | 5 모델 × 2 데이터셋 × 3 seeds × 4 rates | 120 |
| **합계** | | **420 runs** |

※ 실험 5는 모든 실험 과정에서 자동 기록

---

## 구현 체크리스트

| 항목 | 상태 | 파일 | 비고 |
|------|------|------|------|
| Multi-seed | ✅ 완료 | `train.py` | `--seeds 42,123,456` |
| Multi-horizon | ✅ 완료 | `train.py` | `--seq_out 3/6/12/24` (SAITS 제외) |
| MAPE metric | ✅ 완료 | `src/core/metrics.py` | `Metrics.MAPE`, `MaskedMetrics.MaskedMAPE` |
| Oracle mode | ✅ 완료 | `train.py`, `loader.py` | `--missing_rate 0` |
| Sensor failure mask | ✅ 완료 | `src/data/loader.py` | `--missing_rate 0.25/0.5/0.75/0.9` |
| Mean imputation | ⬜ 구현 필요 | Wrapper 수정 | 실험 2용 |
| Learnable token | ⬜ 구현 필요 | GIMCC, GinAR wrapper | 실험 3용 |
| Cost logging | ⬜ 추가 필요 | `src/core/trainer.py` | 실험 5용 |

---

## 예상 결과물

1. **재현성 보고서:** 논문 vs 실제 성능 비교표
2. **Ablation 분석:** Imputation 기여도 정량화
3. **Semantic Ambiguity 논문 초안:** Learnable token 효과 검증
4. **Robustness 분석:** Missing rate별 성능 저하 곡선
5. **효율성 분석:** Cost-Performance Pareto frontier

---

## 연구 주제 후보 (실험 결과에 따라)

1. **Imputation 경량화:** 복잡한 모듈이 불필요하다면 → 효율적 대안 제시
2. **Semantic-aware Missing Handling:** Learnable token이 효과적이라면 → 방법론 확장
3. **Robust VSF:** 극단적 missing rate에서도 강건한 모델 설계
4. **Efficient VSF:** 성능-효율 균형점을 찾는 연구
