"""최종 통합 화면 — 세계지도 위에서 공간 이동(로드밸런서)과 시간 이동(스케줄러)을 함께 본다.

구성 (앞으로 하나씩 확장):
  1) 큰 세계지도
  2) 로드밸런서 배정: job이 원래 리전(origin)에서 다른 나라(assigned)로 보내지는 것을
     나라 간 곡선 화살표로 표시. 화살표 굵기 = 그 경로로 간 job 수.
  3) 도착 나라에서의 시간 스케줄링: 선택한 도착 나라의 24시간 탄소강도 예측을 보여주고,
     탄소가 낮은 시간대(스케줄러가 job을 미룰 목표 구간)를 표시.

성능: 146,000개(365일) 전체를 라이브로 스케줄링하면 매우 느리므로,
     슬라이더로 고른 시각 주변의 '창(window)'에 든 job만 그린다.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from interface.regions import REGION_LABELS, to_iso3, to_region

# 리전별 대략 중심 좌표 (lon, lat) — 화살표 끝점·나라 라벨 위치
REGION_COORD = {
    "US-CAL-CISO": (-120, 37),
    "US-TEX-ERCO": (-99, 31),
    "US-NY-NYIS":  (-74, 41),
    "FR": (2, 47),
    "DE": (10, 51),
    "KR": (127, 37),
    "IN": (78, 22),
    "JP": (139, 37),
}

LB_RESULTS = os.path.join(_REPO_ROOT, "load_balancer", "02_프레임워크", "results",
                          "assign_alpha_auto.csv")

ROUTE_COLOR = "#2e7d32"   # 이동 화살표
STAY_COLOR = "#9aa0a6"    # 제자리 실행


@st.cache_data(show_spinner=False)
def load_assign(path):
    """로드밸런서 배정 CSV -> 표준 리전 코드 + 시간(h) 컬럼 추가."""
    df = pd.read_csv(path)
    df["origin_std"] = df["origin"].map(to_region)
    df["assigned_std"] = df["assigned"].map(to_region)
    df["submit_h"] = df["submit_time"] / 3600.0
    df["moved"] = df["origin_std"] != df["assigned_std"]
    return df


def _arc(lon0, lat0, lon1, lat1, n=24, bump=0.18):
    """두 지점을 잇는 완만한 호(arc) 좌표. 중간을 위로 살짝 띄워 곡선처럼 보이게 한다."""
    t = np.linspace(0, 1, n)
    lon = lon0 + (lon1 - lon0) * t
    lat = lat0 + (lat1 - lat0) * t
    lat = lat + np.sin(np.pi * t) * abs(lon1 - lon0) * bump
    return lon, lat


def draw_map(window_df, height=620):
    """창 안의 job들을 세계지도에 그린다: 도착 나라 강조 + origin→assigned 화살표."""
    fig = go.Figure()

    # (1) 도착 나라 강조 — assigned 기준 job 수
    dest_counts = window_df["assigned_std"].value_counts().to_dict()
    iso_counts = {}
    for region, n in dest_counts.items():
        iso_counts[to_iso3(region)] = iso_counts.get(to_iso3(region), 0) + n
    if iso_counts:
        fig.add_trace(go.Choropleth(
            locations=list(iso_counts), locationmode="ISO-3",
            z=list(iso_counts.values()),
            colorscale=[[0, "#e8f5e9"], [1, "#2e7d32"]],
            showscale=False, marker_line_color="white", marker_line_width=0.5,
            hovertext=[f"{k}: {v} job 도착" for k, v in iso_counts.items()],
            hoverinfo="text",
        ))

    # (2) 이동 경로 화살표 — (origin, assigned)별 집계, 굵기 = job 수
    moved = window_df[window_df["moved"]]
    routes = moved.groupby(["origin_std", "assigned_std"]).size()
    if len(routes):
        wmax = routes.max()
        for (o, a), n in routes.items():
            if o not in REGION_COORD or a not in REGION_COORD:
                continue
            lon0, lat0 = REGION_COORD[o]
            lon1, lat1 = REGION_COORD[a]
            lon, lat = _arc(lon0, lat0, lon1, lat1)
            fig.add_trace(go.Scattergeo(
                lon=lon, lat=lat, mode="lines",
                line=dict(width=1 + 5 * (n / wmax), color=ROUTE_COLOR),
                opacity=0.6, hoverinfo="text",
                hovertext=f"{REGION_LABELS.get(o, o)} → {REGION_LABELS.get(a, a)} : {n} job",
                showlegend=False,
            ))
            # 도착점 화살촉(마커)
            fig.add_trace(go.Scattergeo(
                lon=[lon1], lat=[lat1], mode="markers",
                marker=dict(size=6, color=ROUTE_COLOR, symbol="circle"),
                hoverinfo="skip", showlegend=False,
            ))

    # (3) 출발점(제자리 포함) 표시
    for region, (lon, lat) in REGION_COORD.items():
        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat], mode="markers",
            marker=dict(size=4, color=STAY_COLOR),
            hoverinfo="text", hovertext=REGION_LABELS.get(region, region),
            showlegend=False,
        ))

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), dragmode=False,
        geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True,
                 coastlinecolor="#d0d0d0", showland=True, landcolor="#f5f5f5",
                 showocean=True, oceancolor="#eaf2f8", projection_type="natural earth"),
    )
    return fig


# ─────────────────────── 화면 ───────────────────────
# 지도를 화면 최상단까지 끌어올리기 위해 메인 컨테이너 상단 여백 제거
st.markdown(
    "<style>[data-testid='stMainBlockContainer']{padding-top:1rem;}</style>",
    unsafe_allow_html=True)

if not os.path.exists(LB_RESULTS):
    st.error(f"로드밸런서 배정 결과를 찾을 수 없습니다: {LB_RESULTS}")
    st.stop()

df = load_assign(LB_RESULTS)

# 자리 잡기용: 고정 시점(Day 1, 6시간 창)으로 지도만 그린다. 조절 UI 없음.
window = df[df["submit_h"] < 6]

st.plotly_chart(draw_map(window, height=720), width="stretch",
                config={"displayModeBar": False})
