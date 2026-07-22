"""
탄소 인지 로드밸런서 — 웹 대시보드 (Streamlit)

실행: .venv/bin/streamlit run app.py
탭: ① 입력 데이터 ② 전/후 비교 ③ α 스윕 · 모드 비교
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (BASE_DIR, RESULTS_DIR, JOBS_CSV, REGIONS,
                    L_NET_MAX_MS, UTC_OFFSET, load_latency_matrix)

st.set_page_config(page_title="탄소 인지 로드밸런서", page_icon="🌍", layout="wide")

# ── 팔레트 (검증된 8색 고정 순서 — 리전 순서에 1:1 매핑, 순환 금지) ──
REGION_COLORS = dict(zip(REGIONS, [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]))
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BASELINE_GRAY = "#898781"
ACCENT = "#2a78d6"

LAYOUT = dict(
    font=dict(family="system-ui, -apple-system, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=GRID, zeroline=False),
    yaxis=dict(gridcolor=GRID, zeroline=False),
    hovermode="closest",
)


@st.cache_data
def load_all(mtime: float):
    """mtime: summary.json 수정 시각 — 캐시 키에 포함되어 결과 파일이 바뀌면
    자동 무효화. (주의: 매개변수를 _mtime처럼 밑줄로 시작하면 Streamlit이
    캐시 키에서 제외해버려 무효화가 작동하지 않음)"""
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    from simulator import CarbonSeries
    carbon = CarbonSeries().frame()   # 실데이터 우선 (time_s + 리전 8열)
    jobs = pd.read_csv(JOBS_CSV)
    latency = load_latency_matrix()
    slots = {name: pd.read_csv(RESULTS_DIR / f"slots_{name}.csv")
             for name in summary}
    # 시간별 라우팅 뷰용: job별 배정 + 제출 시각 (구버전 CSV엔 submit_time이
    # 없으므로 jobs.csv와 조인해 항상 확보)
    assigns = {}
    for name in summary:
        a = pd.read_csv(RESULTS_DIR / f"assign_{name}.csv")
        if "submit_time" not in a.columns:
            a = a.merge(jobs[["job_name", "submit_time"]], on="job_name")
        assigns[name] = a
    return summary, carbon, jobs, latency, slots, assigns


T0 = pd.Timestamp("2025-01-01")  # t=0 ↔ 탄소 실데이터의 2025년 축 (UTC)


def ts(series_s):
    """초 단위 시각 → 실제 날짜시각 (1년치로 늘어나도 그대로 동작)."""
    return T0 + pd.to_timedelta(series_s, unit="s")


# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 탄소 인지 LB")
    st.caption("LSTM + ILP 라우팅 시뮬레이터 · 1년 실데이터 · Azure 8리전")
    if st.button("🔄 실험 다시 실행 (α 스윕)"):
        with st.spinner("baseline + α 5개 실행 중… (~30초)"):
            r = subprocess.run([sys.executable, str(BASE_DIR / "run_experiments.py")],
                               capture_output=True, text=True, cwd=BASE_DIR)
        st.code(r.stdout or r.stderr)
        st.cache_data.clear()
    st.divider()
    st.markdown(
        "**현재 가정 (placeholder)**\n"
        "- 탄소강도: 합성 데이터 (`gen_carbon.py`)\n"
        "- 예측: perfect forecast (LSTM 자리)\n"
        "- 용량: baseline 피크 × 1.2, headroom 0.8\n"
        "- job 전력 1 kW 균일"
    )

if not (RESULTS_DIR / "summary.json").exists():
    st.warning("결과가 없습니다. 먼저 `python run_experiments.py`를 실행하세요.")
    st.stop()

summary, carbon, jobs, latency, slots, assigns = load_all(
    (RESULTS_DIR / "summary.json").stat().st_mtime)
BASE_M = summary["baseline"]["metrics"]

def _alpha_key(run_name):
    """정렬 키: 숫자 α 오름차순, auto는 맨 뒤."""
    part = run_name.split("_")[1]
    return 1.5 if part == "auto" else float(part)


ALPHA_RUNS = sorted([k for k in summary if k.startswith("alpha_") and "_l" not in k],
                    key=_alpha_key)
AUTO = next((k for k in ALPHA_RUNS if k.split("_")[1] == "auto"), None)
AM = summary[AUTO]["metrics"] if AUTO else None
if not ALPHA_RUNS:
    st.warning("결과 파일이 예전 버전인 것 같아요 — 사이드바의 **🔄 실험 다시 실행**을 "
               "누르거나 터미널에서 `python run_experiments.py`를 실행해 주세요.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["① 입력 데이터", "② 전 / 후 비교", "③ α 스윕 · 모드 비교"])

# ═════════════════════ ① 입력 데이터 ═════════════════════
with tab1:
    st.subheader("리전별 탄소강도 (실측, 1시간 해상도)")
    st.caption("소스: 03_데이터/lstm_eval의 y_true. 1월 1~7일은 1월 8일 프로파일 반복.")
    fig = go.Figure()
    for r in REGIONS:
        fig.add_trace(go.Scatter(
            x=ts(carbon.time_s), y=carbon[r], name=r, mode="lines",
            line=dict(color=REGION_COLORS[r], width=2),
            hovertemplate=f"{r}: %{{y:.0f}} g/kWh<br>%{{x|%m-%d %H시}}<extra></extra>"))
    fig.update_layout(**LAYOUT, height=420, legend=dict(orientation="h", y=1.12),
                      xaxis_title="시각 (UTC)", yaxis_title="gCO₂/kWh")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("리전 간 레이턴시 (ms)")
        fig = go.Figure(go.Heatmap(
            z=latency, x=REGIONS, y=REGIONS, colorscale=BLUE_SEQ,
            text=latency.astype(int), texttemplate="%{text}",
            hovertemplate="%{y} → %{x}: %{z} ms<extra></extra>", showscale=False))
        fig.update_layout(**LAYOUT, height=420, yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("job 제출 분포 (리전 × UTC 시각)")
        pivot = (jobs.assign(h=((jobs.submit_time // 3600) % 24).astype(int))
                 .groupby(["region", "h"]).size().unstack(fill_value=0)
                 .reindex(index=REGIONS, columns=range(24), fill_value=0))
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(range(24)), y=REGIONS, colorscale=BLUE_SEQ,
            hovertemplate="%{y} · UTC %{x}시: %{z}개<extra></extra>",
            colorbar=dict(title="개")))
        fig.update_layout(**LAYOUT, height=420, yaxis_autorange="reversed",
                          xaxis_title="UTC 시각")
        st.plotly_chart(fig, width="stretch")
        st.caption("UTC 기준이라 리전별 봉우리가 시차만큼 어긋나 보입니다 "
                   "(각 리전의 현지 낮이 서로 다른 UTC 시간대에 위치).")

    with st.expander("jobs.csv 미리보기 / 통계"):
        st.dataframe(jobs.head(50), width="stretch")
        st.write(f"총 {len(jobs):,}개 · 리전별 {len(jobs)//8}개 균등 · "
                 f"deferrable(L_max≥1h) {(jobs.L_max >= 3600).mean()*100:.1f}%")

# ═════════════════════ ② 전 / 후 비교 ═════════════════════
with tab2:
    alpha_pick = st.radio("α 선택 (0 = 레이턴시 중심 ←→ 1 = 탄소 중심)",
                          ALPHA_RUNS,
                          index=ALPHA_RUNS.index(AUTO) if AUTO else 0,
                          horizontal=True,
                          format_func=lambda k: f"α = {k.split('_')[1]}")
    M = summary[alpha_pick]["metrics"]

    k1, k2, k3, k4 = st.columns(4)
    d_carbon = (M["total_carbon_kg"] / BASE_M["total_carbon_kg"] - 1) * 100
    k1.metric("총 탄소 배출", f"{M['total_carbon_kg']:,.0f} kg",
              f"{d_carbon:+.1f}% vs baseline", delta_color="inverse")
    k2.metric("평균 네트워크 지연", f"{M['avg_latency_ms']:.1f} ms",
              f"{M['avg_latency_ms'] - BASE_M['avg_latency_ms']:+.1f} ms",
              delta_color="inverse")
    k3.metric("홈 리전 처리 비율", f"{M['home_ratio']*100:.1f} %",
              f"{(M['home_ratio'] - 1)*100:+.1f} %p")
    k4.metric("드롭된 job", f"{M['dropped']}", "전량 처리" if M["dropped"] == 0 else "확인 필요")

    st.divider()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("시간대별 배출률 — 전(baseline) vs 후")
        sb, sa = slots["baseline"], slots[alpha_pick]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts(sb.time_s), y=sb.emission_g_per_h / 1000,
                                 name="baseline (전)", mode="lines",
                                 line=dict(color=BASELINE_GRAY, width=2)))
        fig.add_trace(go.Scatter(x=ts(sa.time_s), y=sa.emission_g_per_h / 1000,
                                 name=f"탄소 인지 LB (후)", mode="lines",
                                 line=dict(color=ACCENT, width=2)))
        fig.update_layout(**LAYOUT, height=380, legend=dict(orientation="h", y=1.12),
                          xaxis_title="시각 (UTC)", yaxis_title="배출률 (kg CO₂/h)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("라우팅 행렬 (출발 → 처리)")
        rm = summary[alpha_pick]["routing_matrix"]
        fig = go.Figure(go.Heatmap(
            z=rm, x=REGIONS, y=REGIONS, colorscale=BLUE_SEQ,
            hovertemplate="%{y} → %{x}: %{z}개<extra></extra>", showscale=False))
        fig.update_layout(**LAYOUT, height=380, yaxis_autorange="reversed",
                          xaxis_title="처리 리전", yaxis_title="출발 리전")
        st.plotly_chart(fig, width="stretch")
        st.caption("baseline은 대각선 100%. 대각선 밖 = 탄소를 위해 이동한 job.")

    st.subheader("누적 탄소 배출 — 실시간 관점")
    st.caption("매 시각의 결정이 쌓여 벌어지는 격차. 시간을 되감지 않고 읽을 수 있는, "
               "실시간 운영 그대로의 그림입니다.")
    fig = go.Figure()
    cum_runs = [("baseline", "baseline (전)", BASELINE_GRAY, "solid"),
                (alpha_pick, f"탄소 인지 LB (후)", ACCENT, "solid")]
    if AUTO and alpha_pick != AUTO:
        cum_runs.append((AUTO, "α=auto 참고", INK, "dot"))
    for name, label, color, dash in cum_runs:
        # job별 실측 배출(carbon_g)을 제출 시각 기준으로 누적 — 총량이 metrics와 일치
        a = assigns[name].sort_values("submit_time")
        fig.add_trace(go.Scatter(x=ts(a.submit_time), y=a.carbon_g.cumsum() / 1000.0,
                                 name=label, mode="lines",
                                 line=dict(color=color, width=2, dash=dash)))
    fig.update_layout(**LAYOUT, height=360, legend=dict(orientation="h", y=1.12),
                      xaxis_title="시각 (UTC)", yaxis_title="누적 배출 (kg CO₂)")
    st.plotly_chart(fig, width="stretch")

    asel = assigns[alpha_pick].copy()
    asel["h"] = (asel.submit_time // 3600).astype(int)  # 슬롯 번호 (경과 h)
    asel["dt"] = ts(asel.submit_time)                   # 실제 날짜시각 (UTC)
    asel["dow"] = asel.dt.dt.dayofweek                  # 0=월 … 6=일
    asel["hod"] = asel.dt.dt.hour                       # UTC 시각 (0~23)

    if M.get("alpha_mode") == "auto":
        st.subheader("슬롯별 자동 선택 α (파레토 무릎점)")
        sa_df = slots[alpha_pick]
        aa = sa_df[sa_df.alpha.notna()]
        fig = go.Figure(go.Scatter(
            x=ts(aa.time_s), y=aa.alpha, mode="lines+markers",
            line=dict(color=ACCENT, width=2), marker=dict(size=5, color=ACCENT),
            hovertemplate="%{x|%m-%d %H시}: α=%{y:.2f}<extra></extra>"))
        fig.update_layout(**LAYOUT, height=280, xaxis_title="시각 (UTC)",
                          yaxis_title="α", yaxis_range=[-0.05, 1.05])
        st.plotly_chart(fig, width="stretch")
        st.caption(f"매 슬롯(1시간) α 후보 11개(0~1, 0.1 간격)의 (평균 지연, 예상 배출) "
                   f"파레토 곡선에서 무릎점을 자동 선택. 평균 α = {M['alpha']:.2f}. "
                   "평가 가중치 w 없이 곡선의 기하학만 사용합니다.")

        with st.expander("Q. auto의 α는 어떤 원리로 계산되나요?"):
            st.markdown(
                "매 1시간 슬롯마다 다음 4단계를 반복합니다:\n\n"
                "1. **재료 수집** — 그 시간에 제출된 job들 + 리전별 탄소강도. "
                "탄소강도는 지금은 슬롯 시작 시점의 **실측값**을 그대로 사용합니다 "
                "(perfect forecast placeholder — LSTM이 준비되면 '과거 24시간 → 다음 "
                "1시간 예측'으로 이 부분만 교체. 즉 현재 성적은 예측 오차 0 가정의 상한선).\n"
                "2. **후보 곡선 그리기** — α 후보 11개(0, 0.1, …, 1)마다 ILP 배정을 "
                "각각 계산해 (평균 지연, 예상 배출) 점 11개를 찍음. 이게 '이 시간의 "
                "지연↔탄소 교환 곡선'.\n"
                "3. **무릎점 선택** — 두 축을 각각 0~1로 정규화(단위 맞추기용, 가중치 "
                "아님)한 뒤, 이상점 (0,0)에서 **유클리드 거리 √(x²+y²)가 최소**인 점을 "
                "채택: α\\* = argmin √(x_α² + y_α²). 곡선이 '급격한 개선 → 미미한 개선'으로 "
                "꺾이는 코너가 수학적으로 이 지점입니다 (다목적 최적화의 이상점 최소 "
                "거리법).\n"
                "4. **적용** — 그 α의 배정을 확정하고, 다음 슬롯에서 1번부터 다시.\n\n"
                "부가 규칙: 드롭이 최소인 후보들 안에서만 선택, 거리 동률이면 작은 α"
                "(지연 우선).")

        with st.expander("Q. 왜 α가 왔다갔다 하나요? — 0인 슬롯도, 1에 가까운 슬롯도 있음"):
            st.markdown(
                "α는 미리 정해두는 튜닝 값이 아니라 **매 시간 새로 내리는 결정의 결과**입니다. "
                "매 슬롯 \"지금 job을 옮기면 지연 1ms당 탄소를 얼마나 살 수 있나\"라는 "
                "**교환 비율**이 달라지고, 무릎점은 그 비율이 급락하는 지점이라 시간마다 "
                "다르게 나오는 게 정상입니다.\n\n"
                "**α가 1에 가까운 시간** — \"지금 옮기면 많이 벌린다\":\n"
                "- 리전 간 탄소 격차가 큼 (예: 프랑스는 새벽 원자력 잉여로 ~50 g/kWh, "
                "인도는 ~700 g/kWh)\n"
                "- 깨끗한 리전에 용량 여유가 있음\n"
                "- 배치에 옮길 가치가 큰 job(장기 실행)이 포함됨\n\n"
                "**α = 0인 시간** — \"이번 시간은 옮길 이유가 없다\":\n"
                "- 리전 간 격차가 작아 옮겨도 얻는 게 거의 없음\n"
                "- 배치의 job들이 이미 깨끗한 리전에서 출발\n"
                "- 깨끗한 리전 용량이 앞 슬롯의 장기 job들로 이미 차 있음\n"
                "- 배치가 작아 선택지가 '안 옮김 vs 대륙 간 대량 이동' 둘뿐(계단형 곡선) "
                "→ 중간 거래가 없어 보수적으로 '안 옮김'을 택함\n\n"
                "즉 이 진동은 노이즈가 아니라 **탄소강도 지형의 시간 변화를 따라가는 "
                "신호**입니다. 고정 α는 이 차이를 무시하고 매시간 같은 강도로 밀어붙이기 "
                "때문에, 격차가 없는 시간에 지연만 낭비합니다 — auto가 파레토 곡선 위쪽(★)에 "
                "있는 이유가 바로 이것입니다.")

    st.subheader("리전별 처리 job 수 — 전 vs 후")
    fc1, fc2 = st.columns([2.6, 1.4])
    sel_months = fc1.segmented_control("월 (중복 선택)", list(range(1, 13)),
                                       selection_mode="multi",
                                       format_func=lambda m: f"{m}월")
    h_lo, h_hi = fc2.slider("시간대 (UTC)", 0, 24, (0, 24))
    sel_origin = st.segmented_control("출발 리전 (중복 선택)", REGIONS,
                                      selection_mode="multi")
    if sel_origin and len(sel_origin) == 1:
        off = UTC_OFFSET[sel_origin[0]]
        st.caption(f"💡 {sel_origin[0]} 현지 시각 = UTC{off:+g}h → 현지 낮(8~20시) ≈ "
                   f"UTC {(8 - off) % 24:g}~{(20 - off) % 24:g}시. "
                   "시간대 슬라이더로 이 구간을 잡으면 '그 리전의 낮'을 보는 셈입니다.")

    with st.expander("🕐 UTC ↔ 리전 현지 시각 대조표 — 슬라이더 선택 구간이 각 리전의 몇 시인지"):
        st.caption("모든 그래프·필터의 축은 UTC 하나로 통일되어 있고, "
                   "이 표가 리전별 현지 시각으로 번역해 줍니다.")
        conv = pd.DataFrame([{
            "리전": r,
            "UTC 오프셋": f"UTC{UTC_OFFSET[r]:+g}",
            f"선택 구간 (UTC {h_lo}~{h_hi}시)의 현지 시각":
                f"{(h_lo + UTC_OFFSET[r]) % 24:g}시 ~ {(h_hi + UTC_OFFSET[r]) % 24:g}시",
            "현지 낮 (8~20시) ≈ UTC":
                f"{(8 - UTC_OFFSET[r]) % 24:g}~{(20 - UTC_OFFSET[r]) % 24:g}시",
        } for r in REGIONS])
        st.dataframe(conv, width="stretch", hide_index=True)

    filt = asel[(asel.hod >= h_lo) & (asel.hod < h_hi)]
    if sel_months:
        filt = filt[filt.dt.dt.month.isin(sel_months)]
    if sel_origin:
        filt = filt[filt.origin.isin(sel_origin)]

    if filt.empty:
        st.info(f"필터 조건에 맞는 job이 없습니다. 현재 데이터 범위: "
                f"{asel.dt.min():%Y-%m-%d} ~ {asel.dt.max():%Y-%m-%d} — "
                "월 필터는 1년치 데이터에서 의미가 생깁니다.")
    else:
        # 선택 구간 요약 지표 (탄소는 같은 job들의 baseline 배출과 비교)
        ok = filt[~filt.dropped]
        bmap = assigns["baseline"].set_index("job_name").carbon_g
        base_kg = filt.job_name.map(bmap).sum() / 1000.0
        after_kg = filt.carbon_g.sum() / 1000.0
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("선택 구간 job", f"{len(filt):,}개")
        mc2.metric("탄소 (후)", f"{after_kg:,.1f} kg",
                   f"{(after_kg / base_kg - 1) * 100:+.1f}% vs baseline"
                   if base_kg > 0 else None, delta_color="inverse")
        mc3.metric("평균 지연 (후)", f"{ok.latency_ms.mean():.1f} ms" if len(ok) else "—")
        mc4.metric("이동 job", f"{int((ok.origin != ok.assigned).sum()):,}개")

        hod_cnt = filt.groupby("hod").size().reindex(range(24), fill_value=0)
        fig = go.Figure(go.Bar(x=list(range(24)), y=hod_cnt.values,
                               marker_color=ACCENT,
                               hovertemplate="UTC %{x}시: %{y}개<extra></extra>"))
        fig.update_layout(**LAYOUT, height=180,
                          xaxis_title="선택 구간의 시간대(UTC) 분포", yaxis_title="job 수")
        st.plotly_chart(fig, width="stretch")

    before = filt.origin.value_counts().reindex(REGIONS, fill_value=0)
    after = filt[~filt.dropped].assigned.value_counts().reindex(REGIONS, fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=REGIONS, y=before.values,
                         name="baseline (전 = 출발 리전)", marker_color=BASELINE_GRAY))
    fig.add_trace(go.Bar(x=REGIONS, y=after.values,
                         name="탄소 인지 LB (후)", marker_color=ACCENT))
    fig.update_layout(**LAYOUT, height=340, barmode="group", bargap=0.25,
                      legend=dict(orientation="h", y=1.15), yaxis_title="처리 job 수")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"필터 적용: {len(filt):,}개 / 전체 {len(asel):,}개 "
               "(버튼 미선택 = 전체 · 다시 누르면 해제). 시간 필터는 전부 UTC 기준. "
               "탄소강도가 낮은 리전으로 부하가 이동하는 정도가 α에 따라 달라집니다.")

    st.subheader("job별 배정 내역 — jobs.csv + 그 시각의 α + 배정 리전")
    st.caption("행 = job 하나: 언제 제출됐고, 그 슬롯의 α가 얼마였고, 그래서 어디로 "
               "배정됐는지. 위 필터가 그대로 적용됩니다.")
    alpha_by_h = {}
    if "alpha" in slots[alpha_pick].columns:
        sa_df = slots[alpha_pick]
        alpha_by_h = {int(t // 3600): a for t, a in
                      zip(sa_df.time_s, sa_df.alpha) if pd.notna(a)}
    # jobs.csv 원본 열 그대로 + 뒤에 α·배정 나라 2열만 추가
    view = filt.sort_values("submit_time")
    tbl = jobs.set_index("job_name").loc[view.job_name].reset_index()
    tbl["α"] = view.h.map(alpha_by_h).values
    tbl["배정"] = view.assigned.fillna("(드롭)").values
    SHOW_MAX = 5000  # 1년치(수십만 행)여도 화면은 가볍게, 전체는 CSV로
    if len(tbl) > SHOW_MAX:
        st.info(f"{len(tbl):,}행 중 앞 {SHOW_MAX:,}행만 표시합니다. "
                "전체는 아래 CSV로 받아 Excel에서 보세요.")
    st.dataframe(tbl.head(SHOW_MAX), width="stretch", hide_index=True, height=420)
    st.download_button(f"⬇️ 전체 {len(tbl):,}행 CSV 다운로드 (Excel 호환)",
                       tbl.to_csv(index=False).encode("utf-8-sig"),
                       f"jobs_routed_{alpha_pick}.csv", "text/csv")

# ═════════════════════ ③ α 스윕 · 모드 비교 ═════════════════════
with tab3:
    st.subheader("파레토 곡선 — 사후 평가 (7일 집계)")
    st.caption("run 하나가 점 하나 (x=평균 지연, y=총 탄소). **파란 곡선은 같은 7일을 "
               "α만 바꿔 여러 번 재생해야 얻어지는 사후(hindsight) 기준선**이고, "
               "auto는 매 시각 그 시점 정보만으로 실시간 달성한 값입니다. "
               "실시간 관점의 그림(누적 배출·시간별 라우팅)은 ②탭에 있습니다.")

    if AUTO:
        st.success(
            f"⭐ **α = auto (슬롯별 파레토 무릎점)** — 총 탄소 {AM['total_carbon_kg']:,.0f} kg "
            f"(baseline 대비 {(1 - AM['total_carbon_kg']/BASE_M['total_carbon_kg'])*100:.1f}% 절감) · "
            f"평균 지연 {AM['avg_latency_ms']:.1f} ms · 슬롯 평균 α = {AM['alpha']:g}")

    def saving_pct(m):  # baseline 대비 탄소 절감률 (%) — 높을수록 좋음
        return (1 - m["total_carbon_kg"] / BASE_M["total_carbon_kg"]) * 100

    FIXED_RUNS = [k for k in ALPHA_RUNS if k != AUTO]  # 곡선은 고정 α만으로
    xs = [summary[k]["metrics"]["avg_latency_ms"] for k in FIXED_RUNS]
    ys = [saving_pct(summary[k]["metrics"]) for k in FIXED_RUNS]
    labels = [f"α={k.split('_')[1]}" for k in FIXED_RUNS]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers+text", text=labels, textposition="bottom right",
        textfont=dict(color=INK), name="고정 α 스윕",
        line=dict(color=ACCENT, width=2), marker=dict(size=10, color=ACCENT),
        hovertemplate="%{text}: 지연 %{x:.1f}ms · 절감 %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[BASE_M["avg_latency_ms"]], y=[0],
        mode="markers+text", text=["baseline"], textposition="top right",
        textfont=dict(color=MUTED), name="baseline (절감 0%)",
        marker=dict(size=12, color=BASELINE_GRAY, symbol="diamond")))
    if AUTO:
        fig.add_trace(go.Scatter(
            x=[AM["avg_latency_ms"]], y=[saving_pct(AM)],
            mode="markers+text", text=["★ α=auto"], textposition="top left",
            textfont=dict(color=INK, size=14), name="α = auto (무릎점)",
            marker=dict(size=14, color=INK, symbol="star")))
    fig.update_layout(**LAYOUT, height=440, legend=dict(orientation="h", y=1.1),
                      xaxis_title="평균 네트워크 지연 (ms) — 오른쪽일수록 비쌈",
                      yaxis_title="탄소 절감률 (% vs baseline) — 위일수록 좋음")
    st.plotly_chart(fig, width="stretch")
    st.caption("**좌상단이 이상적** (지연은 적게, 절감은 많이). ★ = auto가 실제로 도달한 "
               "지점 — 곡선보다 위에 떠 있다면 같은 지연 예산으로 고정 α보다 더 아꼈다는 뜻. "
               "auto의 슬롯별 α 변화는 ②탭에서 볼 수 있습니다.")

    st.divider()
    st.subheader("세 가지 운영 모드")
    modes = [("alpha_0", "🏠 지역(레이턴시) 중심", "α=0 — 항상 가장 가까운 리전"),
             ("alpha_0.5", "⚖️ 균형", "α=0.5 — 탄소·지연 절충"),
             ("alpha_1", "🌱 탄소 중심", "α=1 — 항상 가장 깨끗한 리전")]
    cols = st.columns(3)
    for col, (key, title, desc) in zip(cols, modes):
        m = summary[key]["metrics"]
        with col:
            st.markdown(f"**{title}**  \n{desc}")
            st.metric("총 탄소", f"{m['total_carbon_kg']:,.0f} kg",
                      f"{(m['total_carbon_kg']/BASE_M['total_carbon_kg']-1)*100:+.1f}%",
                      delta_color="inverse")
            st.metric("평균 지연", f"{m['avg_latency_ms']:.1f} ms")
            st.metric("홈 리전 비율", f"{m['home_ratio']*100:.0f} %")

    st.divider()
    st.subheader("전체 결과 표")
    rows = []
    for name in ["baseline"] + ALPHA_RUNS:
        m = summary[name]["metrics"]
        rows.append(dict(
            run=name, 모드=m["mode"],
            α=("—" if m["mode"] != "ilp"
               else f"auto (평균 {m['alpha']:g})" if m.get("alpha_mode") == "auto"
               else f"{m['alpha']:g}"),
            총탄소_kg=m["total_carbon_kg"],
            평균지연_ms=m["avg_latency_ms"], p95지연_ms=m["p95_latency_ms"],
            홈리전=f"{m['home_ratio']*100:.1f}%", 드롭=m["dropped"],
            **{"탄소절감(vs baseline)":
               f"{(1 - m['total_carbon_kg']/BASE_M['total_carbon_kg'])*100:.1f}%"}))
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("고정 α run들은 auto의 비교 기준입니다. auto = 매 슬롯 파레토 무릎점 α 자동 선택.")
