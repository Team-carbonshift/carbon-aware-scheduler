"""
실험 일괄 실행: baseline + (거리 정책 3종 × α 스윕 {0, 0.25, 0.5, 0.75, 1.0})

거리 정책 (논문의 이동 거리 제한 규칙):
  제한 없음 / 2500km (대륙 내) / 1200km (인접국만: 프랑스↔독일, 한국↔일본)

결과 → results/summary.json + results/assign_<run>.csv
사용법: .venv/bin/python run_experiments.py
"""
import json
import time

import pandas as pd

from config import JOBS_CSV, RESULTS_DIR, DATA_DIR, load_latency_matrix
from gen_carbon import generate as gen_carbon_df
from simulator import CarbonSeries, SimConfig, run_sim

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DIST_POLICIES = [(None, ""), (2500.0, "_d2500"), (1200.0, "_d1200")]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not (DATA_DIR / "carbon_intensity.csv").exists():
        DATA_DIR.mkdir(exist_ok=True)
        gen_carbon_df().to_csv(DATA_DIR / "carbon_intensity.csv", index=False)
        print("탄소강도 데이터 생성 완료")

    jobs = pd.read_csv(JOBS_CSV)
    carbon = CarbonSeries()
    latency = load_latency_matrix()

    runs = [("baseline", SimConfig(alpha=0.0, label="baseline"), "baseline")]
    for dist, suffix in DIST_POLICIES:
        runs += [(f"alpha_{a:g}{suffix}",
                  SimConfig(alpha=a, dist_max_km=dist, label=f"α={a:g}{suffix}"), "ilp")
                 for a in ALPHAS]

    summary = {}

    def run_one(name, cfg, mode):
        t0 = time.time()
        out = run_sim(jobs, carbon, latency, cfg, mode=mode)
        out["assignments"].to_csv(RESULTS_DIR / f"assign_{name}.csv", index=False)
        out["slot_series"].to_csv(RESULTS_DIR / f"slots_{name}.csv", index=False)
        summary[name] = dict(metrics=out["metrics"], routing_matrix=out["routing_matrix"])
        m = out["metrics"]
        print(f"{name:>10}: 탄소 {m['total_carbon_kg']:>8.1f} kg | "
              f"평균지연 {m['avg_latency_ms']:>6.1f} ms | 홈리전 {m['home_ratio']*100:5.1f}% | "
              f"드롭 {m['dropped']} | {time.time()-t0:.1f}s")

    for name, cfg, mode in runs:
        run_one(name, cfg, mode)

    # α 미세 탐색 결과(fine_alpha.csv)가 있으면 정책별 최적 α run도 정식 포함
    # → 대시보드의 α 선택지·표·파레토에 등장 (w=0.5, 전역 고정 앵커 기준)
    fine_path = RESULTS_DIR / "fine_alpha.csv"
    if fine_path.exists():
        fine = pd.read_csv(fine_path)
        max_c = summary["baseline"]["metrics"]["total_carbon_kg"]
        max_l = max(v["metrics"]["avg_latency_ms"] for v in summary.values())
        for pol_key, (dist, suffix) in zip(["none", "d2500", "d1200"], DIST_POLICIES):
            f = fine[fine.policy == pol_key]
            if f.empty:
                continue
            sc = 0.5 * (1 - f.total_carbon_kg / max_c) + 0.5 * (1 - f.avg_latency_ms / max_l)
            a = float(f.alpha.iloc[sc.values.argmax()])
            name = f"alpha_{a:g}{suffix}"
            if name not in summary:
                print(f"[미세 탐색 최적 α 추가] {name}")
                run_one(name, SimConfig(alpha=a, dist_max_km=dist, label=f"α={a:g}{suffix}"), "ilp")

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n저장: {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
