# 01_데이터 — 로드밸런서 입력함 (1년 실데이터, 단일 세트)

```
01_데이터/
├── jobs.csv                 1년치 job 워크로드 146,000개 (8리전 × 365일 × 50/일)
├── lstm_eval/               ★ 탄소강도 실측 + LSTM 예측 (리전별 8개)
│   └── {KR,FR,DE,IN,JP,US-CAL-CISO,US-TEX-ERCO,US-NY-NYIS}_eval_records.csv
├── 8x8레이턴시표.csv         리전 간 지연 상수 (고정)
├── 8x8레이턴시_정규화.csv    위의 정규화판 (참고용)
└── 생성기/                   합성 job 생성기 (gen_jobs.py, N_DAYS=365)
```

## lstm_eval 스키마

```
timestamp, horizon, y_true, y_pred, abs_err
2025-01-08 00:00:00, 1, 219.0, 206.8, 12.2
```

- **timestamp = 예측 대상 시각** (y_true는 시각당 유일), horizon = 몇 시간 전 발행 예측인지
- 기간: 2025-01-08 ~ 2025-12-31, 1시간 해상도 (리전당 8,569시각 × horizon 1~24)
- **y_true** = 실측 탄소강도 → 탄소 회계 + perfect forecast 재료
- **y_pred(horizon=1)** = LSTM 1시간 예측 → 라우팅 결정에 사용 (torch 불필요, 룩업)

## 시간축 규약

- jobs의 **t=0 ↔ 2025-01-01 00:00 UTC**
- 탄소 데이터가 없는 **1월 1~7일은 1월 8일의 시간대별 값을 반복 사용** (합의 규약)

## 출력

- `../03_라우팅결과/jobs_routed_*.csv` — 스케줄러 인계용 (jobs.csv + alpha + assigned_region)
- `../02_프레임워크/results/` — 대시보드 분석용 내부 산출물 (직접 볼 일 없음)
