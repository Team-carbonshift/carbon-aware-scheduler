# 탄소 인지 로드밸런서 — 시뮬레이션 프레임워크

`탄소인지_로드밸런서_정리.md`의 설계(LSTM 예측 + ILP 라우팅) 구현.
**1년 실데이터로 동작**: 입력은 전부 `../01_데이터/` (jobs.csv + lstm_eval/), 스키마는
그쪽 README 참고.

## 실행법 (처음 클론했을 때)

```bash
cd load_balancer/02_프레임워크
python3 -m venv .venv                    # 1) 가상환경 (최초 1회)
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py           # 2) 웹 대시보드 (브라우저 자동 오픈)
```

실험을 직접 다시 돌리고 싶을 때 (1년치 전체 ~45분):

```bash
.venv/bin/python run_experiments.py      # baseline + 고정 α 5개 + α=auto
```

## 파일 구조

| 파일 | 역할 |
|---|---|
| `config.py` | 경로·리전 순서·UTC 오프셋·레이턴시 행렬 로더 |
| `simulator.py` | 핵심 엔진 — 1시간 슬롯마다 ILP(PuLP/CBC) 배정, α=auto 무릎점 선택 |
| `run_experiments.py` | baseline + α 스윕 + auto 일괄 실행 → `results/` + `../03_라우팅결과/` |
| `app.py` | Streamlit 대시보드 (입력 데이터 / 전후 비교 / 파레토 사후 평가) |
| `results/` | 대시보드 분석용 내부 산출물 (summary.json, run별 배정·슬롯 기록) |

## 데이터 흐름

```
01_데이터/jobs.csv ─┐
01_데이터/lstm_eval/ ┤→ simulator.CarbonSeries ─→ run_experiments ─→ results/ → app.py
  (y_true 실측 ·     │   · 탄소 회계 = y_true 적분              └→ 03_라우팅결과/ (스케줄러 인계)
   y_pred LSTM 예측) ┘   · 라우팅 예측 = y_pred (h=1)
```

- **예측 = LSTM 사전 계산값(y_pred)**: torch 없이 룩업으로 동작. perfect forecast와
  비교하려면 `CarbonSeries(use_lstm_pred=False)`.
- **α=auto**: 매 슬롯 α 후보 11개(0.1 간격)의 (지연, 배출) 곡선에서 이상점 최소 거리
  (무릎점)의 α를 자동 선택. 가중치 w 불사용.

## 현재 가정

1. **용량 C_j** = baseline 최대 동시실행 × 1.2 (균일), headroom 0.8 — 실측값 나오면 교체
2. **job 전력 = 1 kW 균일** — 상대 비교용
3. **job은 제출 즉시 시작** (시간 이동 없음 — 그건 스케줄러 담당), 네트워크 지연만 비용
4. 1월 1~7일 탄소는 1월 8일 프로파일 반복 (데이터 시작 전 구간, 합의 규약)

## 설계 반영 사항 (정리 노트 §4~§8)

- 목적함수 `min Σ (α·M̃ + (1−α)·l̃)·x`, 매 슬롯 max 정규화
- 용량 headroom 0.8 (쏠림 완화) + slack 페널티 1000 (infeasibility 방어)
- 1시간 재최적화 주기 — 라우팅 행렬을 1시간마다 갱신 (MPC receding horizon의 골격)
- 레이턴시는 Azure 공식 baseline 고정 (244ms 정규화)
- 네트워크 지연 SLO 상한(`SimConfig.l_net_max`)은 옵션 (기본 미사용 — 지연 억제는
  α=auto의 무릎점 선택이 담당)
