# 03_라우팅결과 — 스케줄러 인계용 최종 산출물

로드밸런서(02_프레임워크)의 최종 출력. **스케줄러는 이 폴더의 CSV를 입력으로 사용한다.**
`run_experiments.py`를 실행하면 자동 재생성된다.

## 파일 (각 146,000행 = 1년치 job 전체)

| 파일 | 의미 |
|---|---|
| **`jobs_routed_auto.csv`** | ★ **기본 인계본** — 매 슬롯(1시간) 파레토 무릎점으로 α를 자동 선택한 결과. `alpha` 열이 시간마다 다름 |
| `jobs_routed_0.csv` `_0.25` `_0.5` `_0.75` `_1.csv` | 고정 α 비교 run 5개 — 파레토 곡선용 대조군. `alpha` 열이 상수 |

## 스키마 (jobs.csv 원본 8열 + 2열 추가)

```
job_name, submit_time, duration, region, k, L_max, submit_local_hour, band,
alpha,            ← 이 job이 배정된 1시간 슬롯에 적용된 α
assigned_region   ← 로드밸런서가 배정한 처리 리전 (비어 있으면 드롭)
```

- `region` = 출발(제출) 리전, `assigned_region` = 실제 처리 리전. 두 값이 다르면 이동한 job.
- 시간 값은 전부 초 단위 UTC 절대축, **t=0 = 2025-01-01 00:00 UTC** (탄소 실데이터의 시간축).
- 라우팅은 LSTM의 1시간 전 발행 예측(y_pred, horizon=1)을 기반으로 결정됨.
- 스케줄러는 `assigned_region` 안에서 시간 이동(deferrable: `L_max ≥ 3600`)을 결정하면 된다.

## 참고 성적 (2025년 1년치, run 결과)

| | 총 탄소 | 평균 지연 |
|---|---|---|
| baseline (라우팅 없음) | 29,183 kg | 0 ms |
| **auto** | **12,640 kg (−56.7%)** | 35.7 ms |

세부 지표·시간별 절감량은 `../02_프레임워크/results/` 참고 (스키마는 그쪽 README.md).
