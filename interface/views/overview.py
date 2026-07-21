"""전체 개요 — 3개 모듈이 어떻게 이어지는지, 지금 무엇이 연결돼 있는지."""

import os
import sys

import streamlit as st

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from interface import carbon_forecast_api as api  # noqa: E402
from interface.regions import REGIONS, label  # noqa: E402

st.title("Carbon-Aware Scheduler")
st.caption("탄소가 낮은 리전(공간)과 시간대(시간)로 job을 옮겨 실행하는 시스템")

st.markdown(
    """
```
[LSTM]  ──리전별 24h 탄소강도 예측──▶  [로드밸런서]  ──job별 배정 리전──▶  [스케줄러]
                                        어느 리전?                        언제 실행?
```
왼쪽 사이드바에서 각 단계 화면으로 이동할 수 있습니다.
"""
)

st.divider()
st.subheader("연결 상태")

status = api.status()
lb_csv = os.path.join(_REPO_ROOT, "load_balancer", "05_프레임워크",
                      "results", "assign_alpha_auto.csv")
lb_ok = os.path.exists(lb_csv)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**LSTM 예측**")
    if status["ready"]:
        st.success("연결됨 — 실제 모델")
    else:
        st.warning("미연결 — 더미 사용")
    st.caption(api.backend_info())
with c2:
    st.markdown("**로드밸런서 배정**")
    if lb_ok:
        st.success("연결됨 — 배정 CSV")
    else:
        st.warning("배정 결과 없음")
    st.caption("results/assign_alpha_auto.csv 를 스케줄러가 그대로 사용")
with c3:
    st.markdown("**스케줄러 time-shift**")
    st.success("동작")
    st.caption("L_max 안에서 탄소·지연 가중 score 최소 시각 선택")

if status["ready"] and status["placeholder_cfe_re"]:
    st.info(
        "LSTM 입력 중 `cfe_pct`·`re_pct`(무탄소/재생에너지 비중)는 실측 데이터가 아직 없어 "
        "탄소강도로부터 만든 **임시 추정값**을 쓰고 있습니다. 실측 CSV가 확보되면 교체됩니다."
    )

st.divider()
st.subheader("리전 (8개)")
st.caption("모듈마다 표기가 달라 interface/regions.py 가 표준 코드로 통일합니다.")

cols = st.columns(4)
for i, r in enumerate(REGIONS):
    cols[i % 4].markdown(f"`{r}` — {label(r)}")

with st.expander("모듈 간 계약 요약"):
    st.markdown(
        """
| 주는 쪽 | 받는 쪽 | 내용 | 형식 |
|---|---|---|---|
| LSTM | 로드밸런서·스케줄러 | 리전별 향후 24h 탄소강도 | `{리전: [24개 float]}` gCO₂/kWh |
| 로드밸런서 | 스케줄러 | job별 실행 리전 | `{job_name: {origin, assigned}}` |
| 스케줄러 | (결과) | job별 실행 시각·배출량 | `scheduled_start`, `carbon_emitted`, `slo_satisfied` |

자세한 내용은 `interface/README.md` 참고.
        """
    )
