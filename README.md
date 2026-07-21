# Carbon-Aware Scheduler

탄소 배출이 낮은 **리전(공간)** 과 **시간대(시간)** 로 클라우드 job을 옮겨 실행하는
탄소 인식 스케줄링 시스템입니다.

## 구성

담당별로 최상위 폴더가 나뉘어 있고, 그 사이는 `interface/`가 이어줍니다.

| 폴더 | 담당 | 역할 |
|---|---|---|
| [`carbon-forecast-LSTM/`](carbon-forecast-LSTM/) | LSTM | 리전별 향후 24시간 탄소강도 예측 |
| [`load_balancer/`](load_balancer/) | 로드밸런서 | job을 **어느 리전**에서 실행할지 (공간 이동) |
| [`scheduler/`](scheduler/) | 스케줄러 | 그 리전에서 **언제** 실행할지 (시간 이동) |
| [`interface/`](interface/) | 공통 | 위 셋이 주고받는 **데이터 계약** |

## 전체 흐름

```
[LSTM]  ──리전별 24h 탄소강도 예측──▶  [로드밸런서]  ──job별 배정 리전──▶  [스케줄러]
                                       어느 리전?                        언제 실행?
                                                                            │
                                                          결과: 총 탄소 / 지연 / SLO 위반
```

## 먼저 읽을 문서

- [`interface/README.md`](interface/README.md) — **모듈 간 계약** (리전 표기 통합, 예측·배정 데이터 형식)
- [`scheduler/README.md`](scheduler/README.md) — time-shift 알고리즘, 비교군, 대시보드 사용법

## 빠른 실행 (스케줄러 대시보드)

```bash
cd scheduler
pip install -r requirements.txt
streamlit run scheduler/gui.py
```
