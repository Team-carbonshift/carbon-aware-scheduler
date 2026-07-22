# load_balancer — 탄소 인지 로드밸런서

LSTM 탄소강도 예측 + ILP 최적화로 job을 **탄소가 깨끗한 리전으로 라우팅**하는 모듈.
매 1시간 슬롯마다 지연↔탄소 교환 곡선의 무릎점으로 α를 자동 선택(auto)한다.

> **1년 실데이터 결과: 탄소 29,183 → 12,640 kg (−56.7%), 평균 지연 35.7ms, 드롭 0**

## 폴더 구조 (번호 = 파이프라인 순서)

```
load_balancer/
├── 00_자료/          연구 참고자료 (논문 · 논문분석 · 설계 노트) — 파이프라인과 무관
├── 01_데이터/        ★ 입력: jobs.csv(1년 워크로드) + lstm_eval/(탄소 실측·LSTM 예측)
├── 02_프레임워크/    엔진: 시뮬레이터(simulator.py) · 실험 러너 · Streamlit 대시보드
│   └── results/     분석용 산출물 (summary, run별 기록, 시간별 절감량)
└── 03_라우팅결과/    ★ 출력: jobs_routed_*.csv — 스케줄러 인계용 납품물
```

각 폴더의 README에 파일별 스키마·규약이 정리되어 있다:
[01_데이터](01_데이터/README.md) ·
[02_프레임워크](02_프레임워크/README.md) ·
[results](02_프레임워크/results/README.md) ·
[03_라우팅결과](03_라우팅결과/README.md)

## 빠른 시작

```bash
cd load_balancer/02_프레임워크
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # 최초 1회
.venv/bin/streamlit run app.py           # 대시보드 (results가 있으면 바로 뜸)
.venv/bin/python run_experiments.py      # 실험 재실행 (1년치 전체 ~40분)
```

## 동작 요약

1. 매 1시간 슬롯: 그 시간 제출 job + LSTM 1시간 예측(y_pred)을 수집
2. α 후보 11개(0~1, 0.1 간격)로 각각 ILP 배정 → (평균 지연, 예상 배출) 곡선
3. 두 축 정규화 후 이상점 (0,0) 최소 거리 = **무릎점 α** 채택 (가중치 w 없음)
4. 배정 확정 → 탄소 회계는 실측(y_true) 적분으로 정산
5. 결과: `03_라우팅결과/jobs_routed_auto.csv` (+ 고정 α 5개 비교 run)

스케줄러와의 역할 경계: 로드밸런서 = **공간 이동**(어느 리전), 스케줄러 =
`assigned_region` 안에서의 **시간 이동**(언제 실행, deferrable job 대상).
