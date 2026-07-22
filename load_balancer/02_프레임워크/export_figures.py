"""results/의 데이터를 정적 이미지(PNG)로 내보내기 → results/figures/

대시보드 없이도 결과를 볼 수 있는 발표·문서용 그림 5장:
  1. pareto_curve.png       고정 α 곡선 + ★auto (정사각형)
  2. cumulative.png         누적 배출 — baseline vs auto
  3. alpha_timeline.png     슬롯별 무릎점 α (1년)
  4. daily_savings.png      일별 절감량 (막대)
  5. region_load.png        리전별 처리 job 수 — 전/후

사용법: .venv/bin/python export_figures.py   (run_experiments.py 끝에서 자동 호출)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR, REGIONS

FIG_DIR = RESULTS_DIR / "figures"
BLUE, GRAY, INK = "#2a78d6", "#898781", "#0b0b0b"

# 한글 폰트 (macOS AppleGothic → 없으면 기본)
for f in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  저장: figures/{name}")


def export_all():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    base_m = summary["baseline"]["metrics"]
    auto_m = summary["alpha_auto"]["metrics"]

    # 1. 파레토 곡선 (정사각형)
    fixed = sorted([(k, v["metrics"]) for k, v in summary.items()
                    if k.startswith("alpha_") and v["metrics"].get("alpha_mode") != "auto"],
                   key=lambda kv: kv[1]["alpha"])
    sav = lambda m: (1 - m["total_carbon_kg"] / base_m["total_carbon_kg"]) * 100
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = [m["avg_latency_ms"] for _, m in fixed]
    ys = [sav(m) for _, m in fixed]
    ax.plot(xs, ys, "-o", color=BLUE, label="고정 α 스윕")
    for (_, m), x, y in zip(fixed, xs, ys):
        ax.annotate(f"α={m['alpha']:g}", (x, y), textcoords="offset points",
                    xytext=(8, -12), fontsize=9)
    ax.scatter([base_m["avg_latency_ms"]], [0], marker="D", color=GRAY, zorder=5,
               label="baseline (절감 0%)")
    ax.scatter([auto_m["avg_latency_ms"]], [sav(auto_m)], marker="*", s=350,
               color=INK, zorder=6, label="α = auto (무릎점)")
    ax.set_xlabel("평균 네트워크 지연 (ms)")
    ax.set_ylabel("탄소 절감률 (% vs baseline)")
    ax.set_title("파레토 곡선 — 고정 α vs 슬롯별 무릎점(auto), 1년")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "pareto_curve.png")

    # 2. 누적 배출
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, label, color in [("baseline", "baseline", GRAY),
                               ("alpha_auto", "탄소 인지 LB (auto)", BLUE)]:
        a = pd.read_csv(RESULTS_DIR / f"assign_{name}.csv").sort_values("submit_time")
        ax.plot(a.submit_time / 86400, a.carbon_g.cumsum() / 1000, color=color, label=label)
    ax.set_xlabel("경과 일수")
    ax.set_ylabel("누적 배출 (kg CO₂)")
    ax.set_title(f"누적 탄소 배출 — 연간 {base_m['total_carbon_kg'] - auto_m['total_carbon_kg']:,.0f} kg 절감"
                 f" ({sav(auto_m):.1f}%)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "cumulative.png")

    # 3. 슬롯별 α 타임라인
    s = pd.read_csv(RESULTS_DIR / "slots_alpha_auto.csv").dropna(subset=["alpha"])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(s.time_s / 86400, s.alpha, color=BLUE, lw=0.4, alpha=0.55)
    ax.plot(s.time_s / 86400, s.alpha.rolling(24 * 7, center=True).mean(),
            color=INK, lw=1.5, label="7일 이동평균")
    ax.set_xlabel("경과 일수")
    ax.set_ylabel("α")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"슬롯별 무릎점 α (1시간 단위, 평균 {auto_m['alpha']:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "alpha_timeline.png")

    # 4. 일별 절감량
    hs = pd.read_csv(RESULTS_DIR / "hourly_savings.csv")
    daily = hs.assign(d=(hs.time_s // 86400).astype(int)).groupby("d").saved_g.sum() / 1000
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(daily.index, daily.values, color=BLUE, width=1.0)
    ax.set_xlabel("경과 일수")
    ax.set_ylabel("절감량 (kg CO₂/일)")
    ax.set_title("일별 탄소 절감량 — baseline 대비")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "daily_savings.png")

    # 5. 리전별 처리 job 수 전/후
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(REGIONS))
    ax.bar([i - 0.2 for i in x], [base_m["region_load"][r] for r in REGIONS],
           width=0.4, color=GRAY, label="baseline (전)")
    ax.bar([i + 0.2 for i in x], [auto_m["region_load"][r] for r in REGIONS],
           width=0.4, color=BLUE, label="auto (후)")
    ax.set_xticks(list(x), REGIONS, rotation=20)
    ax.set_ylabel("처리 job 수")
    ax.set_title("리전별 처리 job 수 — 탄소가 깨끗한 리전으로의 이동")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "region_load.png")


if __name__ == "__main__":
    export_all()
