# results/ — 분석용 산출물 (대시보드가 읽는 내부 데이터)

`run_experiments.py` 실행 시 자동 생성. run 7개(baseline + 고정 α 5개 + auto) × 파일 2종 + 요약 2개
+ 정적 그림 5장. 스케줄러 인계용 납품물은 여기가 아니라 `../../03_라우팅결과/`.

## figures/ — 발표·문서용 정적 이미지 (PNG, `export_figures.py`)

| 파일 | 내용 |
|---|---|
| `pareto_curve.png` | 고정 α 곡선 + ★auto (정사각형) — 핵심 결과 한 장 |
| `cumulative.png` | 누적 배출 baseline vs auto — 연간 절감량이 벌어지는 그림 |
| `alpha_timeline.png` | 슬롯별 무릎점 α 1년 + 7일 이동평균 |
| `daily_savings.png` | 일별 절감량 막대 |
| `region_load.png` | 리전별 처리 job 수 전/후 |

## summary.json — run별 성적표

run 이름(`baseline`, `alpha_0.5`, `alpha_auto`, …)을 키로 하는 JSON:

```jsonc
{
  "alpha_auto": {
    "metrics": {
      "total_carbon_kg": 12639.5,   // 총 탄소 (실측 강도 적분)
      "avg_latency_ms": 35.7,       // 평균 네트워크 지연
      "p95_latency_ms": ...,        // p95 지연
      "home_ratio": 0.563,          // 홈 리전 처리 비율
      "alpha": 0.50,                // auto면 슬롯 α 평균, 고정이면 그 값
      "alpha_mode": "auto",         // "auto" | "fixed"
      "dropped": 0,                 // 전 리전 만석으로 못 받은 job 수
      "capacity": ..., "headroom": 0.8,
      "region_load": { "France": ..., ... }   // 리전별 처리 job 수
    },
    "routing_matrix": [[...]]       // 8×8, [출발][처리] = job 개수 (리전 순서는 config.REGIONS)
  }
}
```

## assign_<run>.csv — job별 장부 (146,000행)

```
job_name, submit_time, origin, assigned, k, duration, latency_ms, carbon_g, dropped
```

- `origin` → `assigned`: 출발 리전 → 처리 리전 (다르면 이동한 job)
- `latency_ms`: 그 경로의 네트워크 지연 (홈 처리 = 0)
- `carbon_g`: 이 job의 **실제 배출량** — 실행 구간 [submit, submit+duration] 동안
  처리 리전의 실측 탄소강도(y_true)를 적분 × 1kW
- `dropped`: True면 배정 실패 (assigned 비어 있음)

## slots_<run>.csv — 1시간 슬롯별 기록 (8,760행)

```
time_s, alpha, emission_g_per_h, run_US_West, …, run_Japan
```

- `alpha`: 그 슬롯에 적용된 α — **auto run에서는 무릎점 선택 결과라 시간마다 다름**
- `emission_g_per_h`: 슬롯 시작 시점 배출률 (시각화용 근사 — 누적 계산에는 쓰지 말 것,
  누적은 assign의 carbon_g가 정확)
- `run_<리전>` 8열: 그 시각 각 리전에서 실행 중이던 job 수 (용량 점유 추적)

## hourly_savings.csv — 시간별 절감량 (8,760행)

```
time_s, baseline_g, auto_g, saved_g, saved_pct, alpha
```

- 같은 job 집합을 baseline과 auto가 어떻게 처리했는지의 시간별 대조
  (배출은 job 제출 시각 기준 귀속)
- `saved_g` = baseline_g − auto_g (음수 = LSTM 예측이 빗나가 오히려 더 배출한 슬롯)
- Excel에서 바로 열어 그래프 그리기 좋은 형태

## 시간축 규약 (모든 파일 공통)

- `time_s`/`submit_time`은 초 단위 UTC 절대축, **t=0 = 2025-01-01 00:00 UTC**
- 슬롯 = 1시간 (3600초), 슬롯 번호 = `time_s // 3600`
