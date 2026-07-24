"""탄소 인식 time-shift 스케줄러를 브라우저에서 테스트하는 Streamlit GUI.

실행:
    streamlit run scheduler/gui.py

레이아웃:
    - 왼쪽 사이드바: 시뮬레이션 실행 + 시점(일자/시각) 조절 + 자동 재생
    - 오른쪽 메인: 지도(나라별 실행 job 수) + 현재 실행 중 job 목록 +
      요청→실행(time-shift) 타임라인. 시점을 바꾸거나 자동재생하면 즉시 갱신.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scheduler import carbon_forecast, data_loader, metrics, simulator
from scheduler.config import MODES, ZONE_LABELS, ZONE_TO_ISO3

st.set_page_config(page_title="탄소 인식 스케줄러", layout="wide", initial_sidebar_state="expanded")

_SCHED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_SCHED_ROOT)

JOB_DIR = os.path.join(_SCHED_ROOT, "data", "job")
ROUTED_CSV_PATH = os.path.join(JOB_DIR, "jobs_routed_alpha_auto.csv")
JOBS_CSV_PATH = os.path.join(JOB_DIR, "jobs.csv")

# 2025년 1년치: 로드밸런서와 같은 job 목록 + 같은 배정 결과를 그대로 사용
YEAR_JOBS_CSV = os.path.join(_REPO_ROOT, "load_balancer", "01_데이터", "jobs.csv")
YEAR_ASSIGN_CSV = os.path.join(_REPO_ROOT, "load_balancer", "02_프레임워크",
                               "results", "assign_alpha_auto.csv")
USING_YEAR = os.path.exists(YEAR_JOBS_CSV) and os.path.exists(YEAR_ASSIGN_CSV)
USING_ROUTED = os.path.exists(ROUTED_CSV_PATH)

# 시뮬레이션 t=0 이 가리키는 실제 시각 — 모두 UTC(협정 세계시) 기준.
# 2025년 1년치: 로드밸런서·LSTM eval_records와 동일한 축.
YEAR_BASE_TIME = pd.Timestamp("2025-01-01 00:00:00")
# 7일치 폴백: jobs.csv 생성 규약(README_jobs.md)의 t=0.
WEEK_BASE_TIME = pd.Timestamp("2026-01-01 00:00:00")

COLOR_SHIFT = "#2e7d32"       # 실행 구간(초록)
COLOR_WAIT = "#c9c9c9"        # 요청~실행 대기(점선 회색)
COLOR_SUBMIT = "#e08600"      # 요청 시각 마커(주황)
PLAY_INTERVAL_SEC = 0.6

# 나라별 대략적 중심 좌표 (lon, lat) — 지도 위 job 개수 숫자 위치
CENTROIDS = {
    "USA": (-98, 39), "FRA": (2, 47), "DEU": (10, 51),
    "KOR": (128, 36), "IND": (79, 22), "JPN": (138, 37),
}


def load_jobs():
    """2025년 1년치(로드밸런서와 동일 데이터)를 우선 사용."""
    if USING_YEAR:
        return data_loader.load_jobs_with_assignment(YEAR_JOBS_CSV, YEAR_ASSIGN_CSV), "year"
    if USING_ROUTED:
        return data_loader.load_routed_jobs_csv(ROUTED_CSV_PATH), "week"
    return data_loader.load_jobs_csv(JOBS_CSV_PATH), "week"


def run_simulation():
    jobs, scope = load_jobs()
    st.session_state["data_scope"] = scope
    sim_horizon = max(j["deadline"] for j in jobs) + 24
    # 탄소 회계는 실측 시계열로 해야 한다. 예측(LSTM)으로 판단하고 채점은 더미로 하면
    # 판단 기준과 채점 기준이 어긋나 절감률이 음수로 나온다.
    carbon_series, is_real = carbon_forecast.load_actual_series(int(sim_horizon) + 48)
    results = simulator.run_all_modes(
        jobs, carbon_series, modes=["carbon_lb_immediate", "carbon_lb_timeshift"]
    )
    st.session_state["results_by_mode"] = results
    st.session_state["carbon_series"] = carbon_series
    st.session_state["carbon_is_real"] = is_real
    st.session_state["horizon_hours"] = sim_horizon
    st.session_state["n_jobs_run"] = len(jobs)


def running_jobs(results, t):
    return [r for r in results
            if r["scheduled_start"] <= t < r["scheduled_start"] + r["duration"]]


def country_job_counts(jobs_at_t):
    counts = {}
    for r in jobs_at_t:
        iso = ZONE_TO_ISO3.get(r["region"], r["region"])
        counts[iso] = counts.get(iso, 0) + 1
    return counts


def sim_base():
    """시뮬레이션 t=0 에 대응하는 실제 시각 (UTC).

    2025년 1년치는 로드밸런서·LSTM과 같은 2025-01-01 00:00 UTC 기준,
    7일치 폴백 데이터는 job 생성 규약대로 2026-01-01 00:00 UTC 기준.
    """
    if st.session_state.get("data_scope") == "year":
        return YEAR_BASE_TIME
    return WEEK_BASE_TIME


def to_utc(h):
    """시뮬레이션 시각(시간 단위) -> 실제 UTC datetime."""
    return sim_base() + pd.Timedelta(hours=float(h))


def fmt_dt(h):
    """시뮬레이션 시각 -> 'YYYY-MM-DD HH:MM' (UTC)."""
    return to_utc(h).strftime("%Y-%m-%d %H:%M")


def fmt_date(h):
    """시뮬레이션 시각 -> 'YYYY-MM-DD' (UTC)."""
    return to_utc(h).strftime("%Y-%m-%d")


def draw_map(counts, color):
    fig = go.Figure()
    iso3 = list(counts.keys())
    # 빈 상태에서도 지도(지리) 형태를 유지하기 위해 항상 scattergeo 트레이스를 둔다.
    pts = [(CENTROIDS[c][0], CENTROIDS[c][1], counts[c]) for c in iso3 if c in CENTROIDS]
    if iso3:
        fig.add_trace(go.Choropleth(
            locations=iso3, locationmode="ISO-3", z=[1] * len(iso3),
            colorscale=[[0, color], [1, color]], showscale=False,
            marker_line_color="white", marker_line_width=0.6, hoverinfo="location",
        ))
    fig.add_trace(go.Scattergeo(
        lon=[p[0] for p in pts], lat=[p[1] for p in pts],
        text=[str(p[2]) for p in pts], mode="text",
        textfont=dict(size=18, color="white", family="Arial Black"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=175, dragmode=False,
        geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False,
                 showcoastlines=True, coastlinecolor="#d0d0d0",
                 showland=True, landcolor="#f2f2f2", projection_type="natural earth"),
    )
    return fig


def _clip(x0, x1, lo=0.0, hi=24.0):
    a, b = max(x0, lo), min(x1, hi)
    return (a, b) if b > a else None


def draw_timeline(jobs_at_t, t, day_start):
    """선택한 날(0~24시) 기준으로 요청(주황◆)→대기(점선)→실행(초록)을 job별로 표시.

    x축은 항상 그 날의 0~24시로 고정. 전날 요청/시작한 부분은 왼쪽 끝(0시)에서 잘림.
    """
    js = sorted(jobs_at_t, key=lambda r: r["scheduled_start"])[:15]
    fig = go.Figure()
    for r in js:
        jid = r["job_id"]
        sub = r["submit_time"] - day_start
        start = r["scheduled_start"] - day_start
        fin = start + r["duration"]
        wseg = _clip(sub, start)
        if wseg:  # 대기 구간
            fig.add_trace(go.Scatter(
                x=list(wseg), y=[jid, jid], mode="lines",
                line=dict(color=COLOR_WAIT, width=3, dash="dot"),
                showlegend=False, hoverinfo="skip"))
        eseg = _clip(start, fin)
        if eseg:  # 실행 구간
            fig.add_trace(go.Scatter(
                x=list(eseg), y=[jid, jid], mode="lines",
                line=dict(color=COLOR_SHIFT, width=9),
                showlegend=False, hoverinfo="text",
                hovertext=f"{jid}<br>요청 {fmt_dt(r['submit_time'])}"
                          f"<br>실행 {fmt_dt(r['scheduled_start'])}"))
        if 0 <= sub <= 24:  # 요청 마커 (그 날 안에 요청된 경우만)
            fig.add_trace(go.Scatter(
                x=[sub], y=[jid], mode="markers",
                marker=dict(color=COLOR_SUBMIT, size=9, symbol="diamond"),
                showlegend=False, hoverinfo="text",
                hovertext=f"요청 {fmt_dt(r['submit_time'])}"))
    fig.add_vline(x=t - day_start, line=dict(color="#d33", width=1.5, dash="dash"))
    fig.update_layout(
        height=210, margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(autorange="reversed", title=None), plot_bgcolor="white",
    )
    fig.update_xaxes(range=[0, 24], tickvals=list(range(0, 25, 3)),
                     title="시각 (시, UTC)", gridcolor="#eee")
    return fig


def carbon_up_to(results, t):
    return sum(r["carbon_emitted"] for r in results if r["scheduled_start"] <= t)


TABLE_COLS = ["job_id", "k", "지역", "요청", "실행 시작", "종료 예정",
              "즉시실행(gCO₂)", "time-shift(gCO₂)", "절감(gCO₂)"]


def jobs_table(jobs_at_t, t, imm_by_id):
    """imm_by_id: {job_id: 즉시실행 시 배출량}.

    즉시실행(gCO₂)  : 안 미루고 요청 즉시 실행했을 때의 배출량
    time-shift(gCO₂): 실제로 time-shift 해서 실행한 배출량(=실제 배출)
    절감(gCO₂)      : 즉시실행 - time-shift (시간 이동으로 아낀 양)
    실행 중 job이 없어도 컬럼(헤더)은 항상 유지해 레이아웃이 흔들리지 않게 한다.
    """
    rows = []
    for r in jobs_at_t:
        region = r["region"]
        finish = r["scheduled_start"] + r["duration"]
        shift_c = r["carbon_emitted"]
        imm_c = imm_by_id.get(r["job_id"], shift_c)
        rows.append({
            "job_id": r["job_id"],
            "k": r["k"],
            "지역": f"{region} ({ZONE_LABELS.get(region, region)})",
            "요청": fmt_dt(r["submit_time"]),
            "실행 시작": fmt_dt(r["scheduled_start"]),
            "종료 예정": fmt_dt(finish),
            "즉시실행(gCO₂)": round(imm_c, 1),
            "time-shift(gCO₂)": round(shift_c, 1),
            "절감(gCO₂)": round(imm_c - shift_c, 1),
        })
    df = pd.DataFrame(rows, columns=TABLE_COLS)
    if not df.empty:
        df = df.sort_values("절감(gCO₂)", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────── 사이드바 ───────────────────────
# 페이지에 들어오면 (로드밸런서처럼) 자동으로 1년치 시뮬레이션이 준비되어 있게 한다.
if st.session_state.get("results_by_mode") is None:
    with st.spinner("2025년 1년치 시뮬레이션 실행 중… (최초 1회)"):
        run_simulation()
    st.session_state["playing"] = False

with st.sidebar:
    st.header("시뮬레이션 설정")

    scope = st.session_state.get("data_scope")
    n_jobs = st.session_state.get("n_jobs_run", 0)
    days = int(st.session_state.get("horizon_hours", 0)) // 24
    st.caption(f"{'2025년 1년치' if scope == 'year' else '7일치'} · job {n_jobs:,}개 · {days}일")
    st.caption(carbon_forecast.backend_info())

    if st.button("다시 실행", width="stretch"):
        with st.spinner("실행 중..."):
            run_simulation()
        st.session_state["playing"] = False

    st.divider()

    has_results = st.session_state.get("results_by_mode") is not None
    max_day = max(1, int(st.session_state.get("horizon_hours", 168)) // 24) if has_results else 9
    max_t = max_day * 24 - 1

    st.session_state.setdefault("ui_day", 1)
    st.session_state.setdefault("ui_hour", 12)
    st.session_state.setdefault("playing", False)

    # 자동 재생 중이면 위젯 생성 전에 시각을 1시간 전진
    if st.session_state["playing"] and has_results:
        cur = (st.session_state["ui_day"] - 1) * 24 + st.session_state["ui_hour"]
        nxt = cur + 1
        if nxt >= max_t:
            nxt = max_t
            st.session_state["playing"] = False  # 끝에 도달하면 정지
        st.session_state["ui_day"] = nxt // 24 + 1
        st.session_state["ui_hour"] = nxt % 24

    st.subheader("시점 선택 (UTC)")
    base_date = sim_base().date()
    min_date = base_date
    max_date = (sim_base() + pd.Timedelta(hours=max_t)).date()

    picked = st.date_input("날짜 (UTC)", value=base_date + pd.Timedelta(
        days=int(st.session_state["ui_day"]) - 1).to_pytimedelta(),
        min_value=min_date, max_value=max_date, disabled=not has_results)
    # 날짜 위젯 값 -> ui_day 로 되돌려 자동재생과 상태를 공유
    st.session_state["ui_day"] = (pd.Timestamp(picked).date() - base_date).days + 1

    hour = st.number_input("시각 (시, UTC)", min_value=0, max_value=23, step=1,
                           key="ui_hour", disabled=not has_results)
    day = st.session_state["ui_day"]
    t_now = (int(day) - 1) * 24 + int(hour)
    st.metric("선택 시각 (UTC)", to_utc(t_now).strftime("%Y-%m-%d %H:%M"))

    play_col, stop_col = st.columns(2)
    if play_col.button("▶ 자동 재생", width="stretch", disabled=not has_results):
        st.session_state["playing"] = True
        st.rerun()
    if stop_col.button("■ 정지", width="stretch", disabled=not has_results):
        st.session_state["playing"] = False
    if st.session_state["playing"]:
        st.caption(f"재생 중… {PLAY_INTERVAL_SEC}초마다 1시간씩 전진")


# ─────────────────────── 메인 ───────────────────────
st.title("탄소 인식 time-shift 스케줄러")

results_by_mode = st.session_state.get("results_by_mode")
if results_by_mode is None:
    st.error("시뮬레이션 결과를 만들지 못했습니다. 왼쪽의 '다시 실행'을 눌러주세요.")
    st.stop()

immediate = results_by_mode["carbon_lb_immediate"]
shifted = results_by_mode["carbon_lb_timeshift"]

comparison = metrics.compare_modes(results_by_mode)
total_imm = comparison["carbon_lb_immediate"]["total_carbon"]
total_shift = comparison["carbon_lb_timeshift"]["total_carbon"]
overall_pct = (1 - total_shift / total_imm) * 100 if total_imm else 0.0
avg_delay = comparison["carbon_lb_timeshift"]["avg_delay"]

imm_running = running_jobs(immediate, t_now)
shift_running = running_jobs(shifted, t_now)
imm_counts = country_job_counts(imm_running)
shift_counts = country_job_counts(shift_running)

cum_imm = carbon_up_to(immediate, t_now)
cum_shift = carbon_up_to(shifted, t_now)
saved_now = cum_imm - cum_shift
saved_now_pct = (saved_now / cum_imm * 100) if cum_imm else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("이 시점까지 누적 절감", f"{saved_now:,.0f} gCO₂", f"{saved_now_pct:.1f}%")
m2.metric("실행 중 job (즉시)", f"{len(imm_running)}개", f"{len(imm_counts)}개국")
m3.metric("실행 중 job (ours)", f"{len(shift_running)}개", f"{len(shift_counts)}개국")
_period = "1년" if st.session_state.get("data_scope") == "year" else "7일"
m4.metric(f"전체 절감 ({_period})", f"{overall_pct:.1f}%", f"평균 지연 {avg_delay:.1f}h")

# 지도 2개 (나라 위 숫자 = 그 나라에서 실행 중인 job 수)
c_left, c_right = st.columns(2)
with c_left:
    st.markdown(f"**즉시 실행** — job {len(imm_running)}개 / {len(imm_counts)}개국")
    st.plotly_chart(draw_map(imm_counts, "#9aa0a6"),
                    width="stretch", config={"displayModeBar": False}, key="map_immediate")
with c_right:
    st.markdown(f"**time-shift (ours)** — job {len(shift_running)}개 / {len(shift_counts)}개국")
    st.plotly_chart(draw_map(shift_counts, COLOR_SHIFT),
                    width="stretch", config={"displayModeBar": False}, key="map_shift")

# 현재 실행 중 job (ours 기준) — 표(전체 폭) + 요청→실행 타임라인 (항상 고정 렌더)
imm_by_id = {r["job_id"]: r["carbon_emitted"] for r in immediate}
tbl = jobs_table(shift_running, t_now, imm_by_id)
bar_max = float(tbl["절감(gCO₂)"].max()) if not tbl.empty and tbl["절감(gCO₂)"].max() > 0 else 1.0
day_start = (int(day) - 1) * 24

st.markdown(f"#### {to_utc(t_now).strftime('%Y-%m-%d %H:%M')} UTC 실행 중 job — "
            f"{len(tbl)}개 · 실제 배출 {tbl['time-shift(gCO₂)'].sum():,.0f} · "
            f"time-shift 절감 {tbl['절감(gCO₂)'].sum():,.0f} gCO₂")
st.dataframe(
    tbl, width="stretch", height=250, hide_index=True,
    column_config={
        "즉시실행(gCO₂)": st.column_config.NumberColumn("즉시실행(gCO₂)", format="%.0f",
                                                     help="안 미루고 즉시 실행했을 때의 배출량"),
        "time-shift(gCO₂)": st.column_config.NumberColumn("time-shift(gCO₂)", format="%.0f",
                                                          help="실제로 미뤄서 실행한 배출량"),
        "절감(gCO₂)": st.column_config.ProgressColumn(
            "절감(gCO₂)", help="즉시실행 대비 줄인 탄소량 (막대)",
            format="%.0f", min_value=0, max_value=bar_max),
    },
)

st.caption(f"아래 타임라인 — ◆ 요청 · · · 대기 ── 실행(초록) · 빨강선 = 현재 시각 "
           f"({fmt_date(t_now)} 00~24시 UTC)")
st.plotly_chart(draw_timeline(shift_running, t_now, day_start),
                width="stretch", config={"displayModeBar": False}, key="timeline")

# ── 세부 정보 (접힘) ──
with st.expander("로드밸런서 배정 결과 · LSTM 예측 · 상세 수치"):
    st.markdown("**로드밸런서 리전 배정 (7일 누적)**")
    counts = pd.Series([r["region"] for r in shifted]).value_counts()
    counts.index = [f"{r} ({ZONE_LABELS.get(r, r)})" for r in counts.index]
    st.bar_chart(counts)

    st.markdown("**LSTM 예측 데이터 (get_carbon_forecast)**")
    if "lstm_forecast" not in st.session_state:
        st.session_state["lstm_forecast"] = carbon_forecast.get_carbon_forecast(horizon=24)
    fc = st.session_state["lstm_forecast"]
    fdf = pd.DataFrame(fc["forecast"]).T
    fdf.index = [f"{r} ({ZONE_LABELS.get(r, r)})" for r in fdf.index]
    fdf.columns = [f"+{h}h" for h in fdf.columns]
    st.dataframe(fdf.style.format("{:.0f}"), width="stretch")

    st.markdown("**비교군별 집계 수치**")
    cdf = pd.DataFrame(comparison).T
    cdf.index = [MODES[m] for m in cdf.index]
    cdf = cdf[["n_jobs", "total_carbon", "avg_delay", "slo_violation_rate"]]
    st.dataframe(cdf.style.format({
        "n_jobs": "{:.0f}", "total_carbon": "{:,.1f}",
        "avg_delay": "{:.3f}", "slo_violation_rate": "{:.4f}",
    }), width="stretch")

    detail_df = pd.DataFrame(shifted)
    st.download_button(
        "결과 CSV 다운로드", detail_df.to_csv(index=False).encode("utf-8"),
        file_name="carbon_scheduler_results.csv", mime="text/csv",
    )

# ── 자동 재생 루프 ──
if st.session_state.get("playing"):
    time.sleep(PLAY_INTERVAL_SEC)
    st.rerun()
