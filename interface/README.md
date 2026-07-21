# interface — 모듈 간 데이터 계약

이 프로젝트는 담당이 3개로 나뉘어 있고, 각자 자기 폴더에서 독립적으로 작업합니다.

```
carbon-forecast-LSTM/   탄소강도 24시간 예측        (LSTM 담당)
load_balancer/          어느 리전에서 실행할지       (로드밸런서 담당)
scheduler/              그 리전에서 언제 실행할지     (스케줄러 담당)
interface/              ← 위 셋을 이어주는 계약 계층
```

모듈끼리 서로의 내부 구현(모델 구조, 최적화 알고리즘 등)을 알 필요가 없도록,
**주고받는 데이터의 형식과 이름만** 이 폴더에 모아둡니다.
한쪽 구현이 바뀌어도 여기 계약만 지키면 나머지는 건드릴 필요가 없습니다.

---

## 데이터 흐름

```
[LSTM]  ──리전별 24h 예측──▶  carbon_forecast_api  ──▶  [로드밸런서] · [스케줄러]
                                                              │
[로드밸런서]  ──job별 배정 리전──▶  lb_assignment  ──────────▶  [스케줄러]
```

---

## 1. `regions.py` — 리전 표기 통합

같은 리전을 세 모듈이 **다른 이름**으로 부르고 있었습니다. 이게 가장 큰 연결 문제였습니다.

| 리전 | 로드밸런서 | LSTM | 표준(채택) |
|---|---|---|---|
| 미 서부 | `US_West` | `US-CAL-CISO` | `US-CAL-CISO` |
| 미 중부 | `US_Central` | `US-TEX-ERCO` | `US-TEX-ERCO` |
| 미 동부 | `US_East` | `US-NY-NYIS` | `US-NY-NYIS` |
| 프랑스 | `France` | `FR` | `FR` |
| 독일 | `Germany` | `DE` | `DE` |
| 한국 | `Korea` | `KR` | `KR` |
| 인도 | `India` | `IN` | `IN` |
| 일본 | `Japan` | `JP` | `JP` |

**표준은 LSTM의 zone 코드를 따릅니다.** LSTM이 실제 학습된 모델 파일을 그 코드로
저장해두었기 때문입니다 (`carbon-forecast-LSTM/models/KR_lstm.pt` 등).

```python
from interface.regions import REGIONS, to_region, to_iso3, label

to_region("Korea")        # -> "KR"     (로드밸런서 표기)
to_region("KR")           # -> "KR"     (이미 표준)
to_region("IN-NO")        # -> "IN"     (과거 스케줄러 코드도 호환)
to_iso3("KR")             # -> "KOR"    (지도용 국가코드, 미국 3리전은 모두 USA)
label("US-NY-NYIS")       # -> "US East (New York)"
```

---

## 2. `carbon_forecast_api.py` — LSTM 경계

스케줄러는 torch·scaler·168시간 입력 같은 LSTM 내부를 몰라야 합니다. 이 함수 하나만 씁니다.

```python
from interface import carbon_forecast_api

forecast = carbon_forecast_api.get_forecast(t_hour=12, horizon=24)
# -> {"KR": [24개 값], "FR": [...], ...}   단위 gCO₂/kWh, index 0 = 기준 시각
```

**2단계 폴백**으로 동작합니다.

1. **실제 LSTM** — torch 설치 + `models/*.pt` 존재 + 과거 탄소강도 데이터가 있으면 진짜 예측
2. **더미** — 위 조건이 안 되면 사인파 + 노이즈 (개발·데모용)

현재 어느 쪽인지는 `backend_info()`로 확인합니다.

```python
carbon_forecast_api.try_init_lstm(carbon_csv="…/carbon_intensity.csv")  # 1단계 시도
carbon_forecast_api.backend_info()   # "실제 LSTM 모델 …" 또는 "더미 예측 …"
```

> LSTM 쪽 원래 시그니처는 `get_forecast_at(t, models, scalers, all_df)` 이며,
> 이 어댑터가 그 호출과 리전 코드 변환을 대신 처리합니다.

---

## 3. `lb_assignment.py` — 로드밸런서 경계

로드밸런서가 job별로 "어느 리전에서 돌릴지" 정한 결과를 읽습니다.
**스케줄러는 리전을 스스로 고르지 않습니다.**

지원 형식 2가지 (자동 인식):

| | 파일 | 원본 리전 | 배정 리전 |
|---|---|---|---|
| A | `load_balancer/05_프레임워크/results/assign_*.csv` | `origin` | `assigned` |
| B | `scheduler/data/job/jobs_routed_alpha_auto.csv` | `region` | `배정` |

```python
from interface.lb_assignment import load_assignments, attach_to_jobs

a = load_assignments("…/assign_alpha_auto.csv")
# -> {"j_002120": {"origin": "IN", "assigned": "FR"}, …}

attach_to_jobs(jobs, a)
# job["region"]        <- origin    (비교군1 baseline)
# job["carbon_region"] <- assigned  (비교군2·3)
```

---

## 계약 요약

| 주는 쪽 | 받는 쪽 | 내용 | 형식 |
|---|---|---|---|
| LSTM | 로드밸런서·스케줄러 | 리전별 향후 24h 탄소강도 | `{리전: [24개 float]}` gCO₂/kWh |
| 로드밸런서 | 스케줄러 | job별 실행 리전 | `{job_name: {origin, assigned}}` |
| 스케줄러 | (결과) | job별 실행 시각·배출량 | `scheduled_start`, `carbon_emitted`, `slo_satisfied` |

리전 이름은 **모든 경계에서 `regions.to_region()`을 거쳐 표준 코드로 정규화**됩니다.
