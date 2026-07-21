"""LSTM 탄소강도 예측 화면 — 실제 모델이 내놓는 24시간 예측을 확인한다."""

import os
import sys

import pandas as pd
import streamlit as st

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from interface import carbon_forecast_api as api  # noqa: E402
from interface.regions import REGIONS, label  # noqa: E402

st.title("LSTM 탄소강도 예측")
st.caption("carbon-forecast-LSTM/models 의 학습된 모델이 향후 24시간 탄소강도를 예측합니다.")

status = api.status()

if status["ready"]:
    st.success(api.backend_info())
else:
    st.warning(api.backend_info())

c1, c2, c3 = st.columns(3)
c1.metric("모델 상태", "연결됨" if status["ready"] else "미연결")
c2.metric("예측 가능 시작", str(status["forecastable_from"] or "-")[:16])
c3.metric("이력 끝", str(status["history_end"] or "-")[:16])

if status["placeholder_cfe_re"]:
    st.info(
        "**참고** — LSTM은 입력으로 `carbon_intensity`와 함께 `cfe_pct`(무탄소 비중)·"
        "`re_pct`(재생에너지 비중) 실측값을 요구합니다. 현재 저장소에는 그 실측 데이터가 없어 "
        "탄소강도로부터 **임시 추정값**을 만들어 넣고 있습니다. "
        "데이터 파이프라인 담당이 실측 CSV를 주면 그대로 교체됩니다."
    )

st.divider()

# ── 예측 시점 선택 ─────────────────────────────────────────
st.subheader("예측 조회")

max_hour = 24 * 9
default_hour = 170 if status["ready"] else 12
t_hour = st.number_input(
    "예측 기준 시각 (시뮬레이션 t, 시간 단위)",
    min_value=0, max_value=max_hour, value=default_hour, step=1,
    help="LSTM은 이 시점 이전 168시간 이력이 있어야 동작합니다. 없으면 더미로 폴백합니다.",
)

forecast = api.get_forecast(t_hour=int(t_hour))
used = api.last_backend()

if used == "lstm":
    st.success(f"이 예측은 **실제 LSTM 모델**이 생성했습니다 (t = {int(t_hour)}h)")
else:
    st.warning(
        f"이 시점(t = {int(t_hour)}h)은 168시간 이력이 없어 **더미**로 대체했습니다. "
        f"실제 LSTM은 t ≥ 168h 구간에서 동작합니다."
    )

df = pd.DataFrame(forecast)
df.index = [f"+{h}h" for h in range(len(df))]
df = df[[r for r in REGIONS if r in df.columns]]
df.columns = [f"{r} ({label(r)})" for r in df.columns]

st.markdown("**향후 24시간 예측 (gCO₂/kWh)**")
st.line_chart(df)

st.markdown("**예측값 표** (행 = 리전, 열 = 몇 시간 후)")
st.dataframe(df.T.style.format("{:.0f}"), width="stretch")

with st.expander("이 예측이 스케줄러에서 어떻게 쓰이나"):
    st.markdown(
        """
스케줄러는 job을 미룰 수 있는 시간(`L_max`) 안에서 이 예측값을 보고
**탄소가 가장 낮은 시각**을 고릅니다.

```
score(t) = α · (탄소 비용) + (1 - α) · (지연 비용)
t* = argmin score(t)
```

즉 위 곡선이 낮게 내려가는 시간대로 job이 이동합니다.
자세한 내용은 `scheduler/README.md` 3절을 참고하세요.
        """
    )
