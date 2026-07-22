# 03_라우팅결과 — 스케줄러 인계용 최종 산출물

로드밸런서(02_프레임워크)의 최종 출력. **스케줄러는 이 폴더의 CSV를 입력으로 사용한다.**
`run_experiments.py`를 실행하면 자동 재생성된다.

## 파일

| 파일 | 의미 |
|---|---|
| `jobs_routed_auto.csv` | **α = auto** (매 슬롯 파레토 무릎점 자동 선택) — 기본 인계본 |
| `jobs_routed_0.5.csv` 등 | 고정 α run (0, 0.1, 0.2, 0.25, 0.3, …, 0.9, 1) — 비교용 |

## 스키마 (jobs.csv 원본 8열 + 2열 추가)

```
job_name, submit_time, duration, region, k, L_max, submit_local_hour, band,
alpha,            ← 이 job이 배정된 1시간 슬롯에 적용된 α
assigned_region   ← 로드밸런서가 배정한 처리 리전 (비어 있으면 드롭)
```

- `region` = 출발(제출) 리전, `assigned_region` = 실제 처리 리전. 두 값이 다르면 이동한 job.
- 시간 값은 전부 초 단위 UTC 절대축 (t=0 = 2026-01-01 00:00 UTC, jobs.csv 규약과 동일).
- 스케줄러는 `assigned_region` 안에서 시간 이동(deferrable: `L_max ≥ 3600`)을 결정하면 된다.
