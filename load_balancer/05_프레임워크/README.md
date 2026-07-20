# 탄소 인지 로드밸런서 — 시뮬레이션 프레임워크

`탄소인지_로드밸런서_정리.md`의 설계(LSTM 예측 + ILP 라우팅)를 jobs.csv 워크로드로
구현한 프레임워크. 아직 실제 탄소강도 데이터가 없어 **합성 데이터로 동작**하며,
실데이터가 오면 CSV 한 장만 교체하면 된다.

## 실행법

```bash
cd 05_프레임워크
.venv/bin/python run_experiments.py      # baseline + α 스윕 (약 30초)
.venv/bin/streamlit run app.py           # 웹 대시보드 (브라우저 자동 오픈)
```

## 파일 구조

| 파일 | 역할 |
|---|---|
| `config.py` | 리전 순서·UTC 오프셋·레이턴시 행렬 로더 |
| `gen_carbon.py` | **임의 탄소강도 생성기** (리전별 현실적 프로파일, 15분 × 8일) |
| `simulator.py` | 핵심 시뮬레이터 — 슬롯(15분)마다 ILP(PuLP/CBC)로 job 배정 |
| `run_experiments.py` | baseline + α ∈ {0, .25, .5, .75, 1} 일괄 실행 → `results/` |
| `app.py` | Streamlit 대시보드 (입력 데이터 / 전후 비교 / α 스윕) |
| `data/carbon_intensity.csv` | 탄소강도 시계열 — **실데이터로 교체할 파일** |
| `results/` | summary.json + run별 배정 결과 |

## 실데이터 교체 방법

`data/carbon_intensity.csv`를 같은 스키마로 갈아끼우면 끝:

```
time_s, US_West, US_Central, US_East, France, Germany, Korea, India, Japan
0.0,    245.3,  410.2, ...                     ← 단위 gCO2/kWh, t=0 = 2026-01-01 00:00 UTC
900.0,  ...                                    ← 등간격이면 해상도는 자유 (15분 권장)
```

## LSTM 연결 지점

`simulator.py` → `CarbonSeries.forecast(t)`. 지금은 실측값을 그대로 반환하는
**perfect forecast placeholder**. LSTM이 준비되면 이 메서드만 예측값을 반환하도록
교체 (인터페이스: 시각 t → 리전 8개 예측값 배열).

## 현재 가정 (실데이터/결정 필요한 것)

1. **탄소강도 = 합성** — `gen_carbon.py`의 리전별 base/일주기/노이즈는 대략적 현실 수준
2. **용량 C_j** = baseline 최대 동시실행 × 1.2 (균일), headroom 0.8 — 실제 서버 수를 알면 교체
3. **job 전력 = 1 kW 균일** — 상대 비교용
4. **job은 제출 즉시 시작** (시간 이동 없음 — 그건 스케줄러 담당), 네트워크 지연만 비용
5. 예측 = perfect forecast (LSTM 자리)

## 설계 반영 사항 (정리 노트 §4~§8)

- 목적함수 `min Σ (α·M̃ + (1−α)·l̃)·x`, 매 슬롯 max 정규화
- 용량 headroom 0.8 (쏠림 완화) + slack 페널티 1000 (infeasibility 방어)
- 15분 재최적화 주기 (MPC receding horizon의 골격)
- 레이턴시는 Azure 공식 baseline 고정 (244ms 정규화)
- **이동 거리 정책** (선행 연구의 이동 제한 규칙, 쏠림 방어의 주 수단):
  제한 없음 / 2500km(대륙 내) / 1200km(프랑스↔독일·한국↔일본만) 3종 비교.
  리전 좌표 기반 대권거리 (`config.distance_matrix`), `SimConfig.dist_max_km`.
  용량은 근거 있는 실측값이 없으므로 느슨한 안전망(baseline 피크×1.2)으로만 사용.
