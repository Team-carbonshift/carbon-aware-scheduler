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

from config import (BASE_DIR, DATA_DIR, RESULTS_DIR, JOBS_CSV, REGIONS,
                    L_NET_MAX_MS, load_latency_matrix)

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
    carbon = pd.read_csv(DATA_DIR / "carbon_intensity.csv")
    jobs = pd.read_csv(JOBS_CSV)
    latency = load_latency_matrix()
    slots = {name: pd.read_csv(RESULTS_DIR / f"slots_{name}.csv")
             for name in summary}
    return summary, carbon, jobs, latency, slots


def hours(series_s):
    return series_s / 3600.0


# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 탄소 인지 LB")
    st.caption("LSTM(예정) + ILP 라우팅 시뮬레이터 · jobs.csv 2,800개 · Azure 8리전 · 7일")
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

summary, carbon, jobs, latency, slots = load_all(
    (RESULTS_DIR / "summary.json").stat().st_mtime)
BASE_M = summary["baseline"]["metrics"]

# 이동 거리 정책 (논문의 데이터 이동 제한 규칙) — 모든 탭에 공통 적용
POLICIES = {"제한 없음": "", "2500km (대륙 내)": "_d2500", "1200km (인접국만, 논문 규칙)": "_d1200"}
pol_name = st.radio("🚚 이동 거리 정책", list(POLICIES), horizontal=True,
                    help="job을 출발 리전에서 얼마나 먼 리전까지 보낼 수 있는지. "
                         "1200km이면 프랑스↔독일, 한국↔일본만 허용 (선행 연구의 이동 제한 규칙).")
SUFFIX = POLICIES[pol_name]
ALPHA_RUNS = sorted([k for k in summary if k.startswith("alpha_")
                     and (k.endswith(SUFFIX) if SUFFIX else "_d" not in k)],
                    key=lambda k: float(k.split("_")[1]))
if not ALPHA_RUNS:
    st.warning(f"'{pol_name}' 정책의 결과가 없습니다. 결과 파일이 예전 버전인 것 같아요 — "
               "사이드바의 **🔄 실험 다시 실행**을 누르거나 터미널에서 "
               "`python run_experiments.py`를 실행해 주세요.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["① 입력 데이터", "② 전 / 후 비교", "③ α 스윕 · 모드 비교"])

# ═════════════════════ ① 입력 데이터 ═════════════════════
with tab1:
    st.subheader("리전별 탄소강도 (합성 데이터, 15분 해상도)")
    st.caption("실데이터를 받으면 data/carbon_intensity.csv만 같은 스키마로 교체하면 됩니다.")
    fig = go.Figure()
    for r in REGIONS:
        fig.add_trace(go.Scatter(
            x=hours(carbon.time_s), y=carbon[r], name=r, mode="lines",
            line=dict(color=REGION_COLORS[r], width=2),
            hovertemplate=f"{r}: %{{y:.0f}} g/kWh<br>t=%{{x:.1f}}h<extra></extra>"))
    fig.update_layout(**LAYOUT, height=420, legend=dict(orientation="h", y=1.12),
                      xaxis_title="경과 시간 (h, UTC)", yaxis_title="gCO₂/kWh")
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
        st.subheader("job 제출 분포 (리전 × 현지 시각)")
        pivot = (jobs.assign(h=jobs.submit_local_hour.astype(int))
                 .groupby(["region", "h"]).size().unstack(fill_value=0)
                 .reindex(REGIONS))
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(range(24)), y=REGIONS, colorscale=BLUE_SEQ,
            hovertemplate="%{y} · %{x}시: %{z}개<extra></extra>",
            colorbar=dict(title="개")))
        fig.update_layout(**LAYOUT, height=420, yaxis_autorange="reversed",
                          xaxis_title="현지 시각")
        st.plotly_chart(fig, width="stretch")

    with st.expander("jobs.csv 미리보기 / 통계"):
        st.dataframe(jobs.head(50), width="stretch")
        st.write(f"총 {len(jobs):,}개 · 리전별 {len(jobs)//8}개 균등 · "
                 f"deferrable(L_max≥1h) {(jobs.L_max >= 3600).mean()*100:.1f}%")

# ═════════════════════ ② 전 / 후 비교 ═════════════════════
with tab2:
    alpha_pick = st.radio("α 선택 (0 = 레이턴시 중심 ←→ 1 = 탄소 중심)",
                          ALPHA_RUNS, index=2, horizontal=True,
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
        fig.add_trace(go.Scatter(x=hours(sb.time_s), y=sb.emission_g_per_h / 1000,
                                 name="baseline (전)", mode="lines",
                                 line=dict(color=BASELINE_GRAY, width=2)))
        fig.add_trace(go.Scatter(x=hours(sa.time_s), y=sa.emission_g_per_h / 1000,
                                 name=f"탄소 인지 LB (후)", mode="lines",
                                 line=dict(color=ACCENT, width=2)))
        fig.update_layout(**LAYOUT, height=380, legend=dict(orientation="h", y=1.12),
                          xaxis_title="경과 시간 (h)", yaxis_title="배출률 (kg CO₂/h)")
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

    st.subheader("리전별 처리 job 수 — 전 vs 후")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=REGIONS, y=[BASE_M["region_load"][r] for r in REGIONS],
                         name="baseline (전)", marker_color=BASELINE_GRAY))
    fig.add_trace(go.Bar(x=REGIONS, y=[M["region_load"][r] for r in REGIONS],
                         name="탄소 인지 LB (후)", marker_color=ACCENT))
    fig.update_layout(**LAYOUT, height=340, barmode="group", bargap=0.25,
                      legend=dict(orientation="h", y=1.15), yaxis_title="처리 job 수")
    st.plotly_chart(fig, width="stretch")
    st.caption("탄소강도가 낮은 리전(France 등)으로 부하가 이동하는 정도가 α에 따라 달라집니다. "
               "용량 headroom(0.8)이 쏠림을 제한합니다.")

# ═════════════════════ ③ α 스윕 · 모드 비교 ═════════════════════
with tab3:
    st.subheader("파레토 곡선 — 탄소 vs 레이턴시 트레이드오프")
    st.caption("ILP의 비용함수(α로 가중)는 run **안**에서 최적 x\\*를 찾는 운전대이고, "
               "아래 **평가점수**는 모든 run을 같은 잣대로 재는 심판입니다. 둘은 별개.")

    w_c = st.slider("평가 가중치 — 탄소를 얼마나 중요하게 볼지 (심판의 잣대)",
                    0.0, 1.0, 0.5, 0.05)

    # 종합 절감률 (α와 무관한 전역 고정 앵커, 높을수록 좋음):
    #   탄소절감% = 1 − 탄소/최대탄소     (최대 = α=0/baseline, 691kg)
    #   지연절감% = 1 − 지연/최대지연     (최대 = 전체 run 중 최악 평균지연, α=1 무제한)
    # 앵커가 "전체에서 도달 가능한 최대치"로 고정돼 있어 거리 정책이 달라도 같은
    # 잣대 — 정책 내 최악값(범위) 정규화는 지연 범위가 좁은 정책(1200km 등)에서
    # 몇 ms 차이를 과대평가해 α=0 쪽으로 왜곡되므로 쓰지 않음.
    MAX_C = BASE_M["total_carbon_kg"]
    MAX_L = max(summary[k]["metrics"]["avg_latency_ms"] for k in summary)

    def carbon_saving(name):
        return 1 - summary[name]["metrics"]["total_carbon_kg"] / MAX_C

    def latency_saving(name):
        return 1 - summary[name]["metrics"]["avg_latency_ms"] / MAX_L

    def eval_score(name):  # 종합 절감률
        return w_c * carbon_saving(name) + (1 - w_c) * latency_saving(name)

    best = max(ALPHA_RUNS, key=eval_score)
    BM = summary[best]["metrics"]
    st.success(f"⭐ **추천 α = {best.split('_')[1]}** (종합 절감률 {eval_score(best)*100:.1f}%) — "
               f"최대 탄소({MAX_C:,.0f} kg) 대비 **{carbon_saving(best)*100:.1f}% 절감**, "
               f"최대 지연({MAX_L:.0f} ms) 대비 **{latency_saving(best)*100:.1f}% 억제** "
               f"(실측: {BM['total_carbon_kg']:,.0f} kg · {BM['avg_latency_ms']:.1f} ms)")

    xs = [summary[k]["metrics"]["avg_latency_ms"] for k in ALPHA_RUNS]
    ys = [summary[k]["metrics"]["total_carbon_kg"] for k in ALPHA_RUNS]
    labels = [f"α={k.split('_')[1]}" for k in ALPHA_RUNS]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers+text", text=labels, textposition="top right",
        textfont=dict(color=INK), name="탄소 인지 LB",
        line=dict(color=ACCENT, width=2), marker=dict(size=10, color=ACCENT)))
    fig.add_trace(go.Scatter(
        x=[BASE_M["avg_latency_ms"]], y=[BASE_M["total_carbon_kg"]],
        mode="markers+text", text=["baseline"], textposition="top right",
        textfont=dict(color=MUTED), name="baseline",
        marker=dict(size=12, color=BASELINE_GRAY, symbol="diamond")))
    fig.add_trace(go.Scatter(
        x=[BM["avg_latency_ms"]], y=[BM["total_carbon_kg"]],
        mode="markers", name="⭐ 추천 α",
        marker=dict(size=22, color="rgba(0,0,0,0)",
                    line=dict(color=INK, width=2), symbol="circle")))
    fig.update_layout(**LAYOUT, height=440, legend=dict(orientation="h", y=1.1),
                      xaxis_title="평균 네트워크 지연 (ms)",
                      yaxis_title="총 탄소 배출 (kg CO₂)")
    st.plotly_chart(fig, width="stretch")
    st.caption("좌하단이 이상적. 검은 링 = 현재 평가 가중치 기준 균형점. "
               "슬라이더를 움직이면 '심판의 잣대'가 바뀌어 추천 α도 달라집니다.")

    # α 미세 탐색 결과 (fine_alpha_search.py 실행 시 생성)
    fine_path = RESULTS_DIR / "fine_alpha.csv"
    if fine_path.exists():
        fine = pd.read_csv(fine_path)
        pol_key = {"": "none", "_d2500": "d2500", "_d1200": "d1200"}[SUFFIX]
        f = fine[fine.policy == pol_key].sort_values("alpha")
        if len(f):
            st.subheader("α 미세 탐색 (0.01 간격, 피크 주변)")
            sc = (w_c * (1 - f.total_carbon_kg / MAX_C)
                  + (1 - w_c) * (1 - f.avg_latency_ms / MAX_L)) * 100
            peak_i = int(sc.values.argmax())
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=f.alpha, y=sc, mode="lines+markers",
                line=dict(color=ACCENT, width=2), marker=dict(size=7, color=ACCENT),
                hovertemplate="α=%{x:.2f}: %{y:.2f}%<extra></extra>", name="종합 절감률"))
            fig.add_trace(go.Scatter(
                x=[f.alpha.iloc[peak_i]], y=[sc.iloc[peak_i]], mode="markers",
                marker=dict(size=20, color="rgba(0,0,0,0)",
                            line=dict(color=INK, width=2)), name="⭐ 피크"))
            fig.update_layout(**LAYOUT, height=340, legend=dict(orientation="h", y=1.15),
                              xaxis_title="α", yaxis_title="종합 절감률 (%)")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"현재 가중치(탄소 {w_c:g}) 기준 피크: "
                       f"**α = {f.alpha.iloc[peak_i]:.2f}** "
                       f"(절감률 {sc.iloc[peak_i]:.2f}%). 곡선이 평평한 구간(플라토)이면 "
                       f"그 안의 α는 사실상 동급입니다.")

            with st.expander(f"α 세부 표 — 탐색한 {len(f)}개 지점 전체 (0.01 간격 포함)"):
                tbl = pd.DataFrame({
                    "α": f.alpha.round(2),
                    "총탄소_kg": f.total_carbon_kg,
                    "평균지연_ms": f.avg_latency_ms.round(2),
                    "탄소절감(vs최대)": ((1 - f.total_carbon_kg / MAX_C) * 100).round(1)
                                        .astype(str) + "%",
                    "지연절감(vs최대)": ((1 - f.avg_latency_ms / MAX_L) * 100).round(1)
                                        .astype(str) + "%",
                    "종합절감률": sc.round(2).astype(str) + "%",
                    "드롭": f.dropped,
                    "추천": ["⭐" if i == peak_i else "" for i in range(len(f))],
                })
                st.dataframe(tbl, width="stretch", hide_index=True)

    st.divider()
    st.subheader("세 가지 운영 모드")
    modes = [(f"alpha_0{SUFFIX}", "🏠 지역(레이턴시) 중심", "α=0 — 항상 가장 가까운 리전"),
             (f"alpha_0.5{SUFFIX}", "⚖️ 균형", "α=0.5 — 탄소·지연 절충"),
             (f"alpha_1{SUFFIX}", "🌱 탄소 중심", "α=1 — 항상 가장 깨끗한 리전")]
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
            run=name, 모드=m["mode"], α=f"{m['alpha']:g}" if m["mode"] == "ilp" else "—",
            총탄소_kg=m["total_carbon_kg"],
            평균지연_ms=m["avg_latency_ms"], p95지연_ms=m["p95_latency_ms"],
            홈리전=f"{m['home_ratio']*100:.1f}%", 드롭=m["dropped"],
            **{"내부비용(참고용)": m.get("ilp_score", None),
               "탄소절감(vs최대)": f"{carbon_saving(name)*100:.1f}%",
               "지연절감(vs최대)": f"{latency_saving(name)*100:.1f}%",
               "종합절감률": f"{eval_score(name)*100:.1f}%"},
            추천="⭐" if name == best else ""))
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "**내부비용(참고용)** = ILP 목적함수 합 Σ(α·M̃+(1−α)·l̃). α라는 색안경을 낀 채 계산된 "
        "값이라 α가 다른 run끼리는 비교 불가 — α=0에서 탄소가 최악인데도 ≈0이 나오는 것이 그 "
        "증거입니다. run **선정에는 사용하지 않습니다.**  \n"
        "**종합절감률** = w·탄소절감% + (1−w)·지연절감%. 앵커는 전역 최대치 고정 — "
        "최대 탄소는 α=0(baseline), 최대 지연은 α=1·거리 무제한의 평균 지연. "
        "거리 정책이 달라도 같은 잣대이며, ⭐ 추천 α는 이 값이 최대인 run입니다.")
