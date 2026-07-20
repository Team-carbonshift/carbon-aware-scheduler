"""
α 미세 탐색 (소수점 둘째 자리) — 정책별 2단계 탐색:
  1단계: α ∈ [0.25, 0.75] 0.05 간격
  2단계: 1단계 피크 ±0.04를 0.01 간격

평가 = 종합 절감률 (전역 고정 앵커: 최대 탄소 = baseline, 최대 지연 = α=1 무제한
평균 지연, 가중치 w=0.5). 앵커·w가 바뀌어도 재활용할 수 있게 지표 원본을 저장.

결과: results/fine_alpha.csv (policy, alpha, total_carbon_kg, avg_latency_ms, dropped)
사용법: .venv/bin/python fine_alpha_search.py
"""
import json
import time

import numpy as np
import pandas as pd

from config import JOBS_CSV, RESULTS_DIR, load_latency_matrix
from simulator import CarbonSeries, SimConfig, run_sim

W = 0.5
POLICIES = [("none", None), ("d2500", 2500.0), ("d1200", 1200.0)]


def main():
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    max_c = summary["baseline"]["metrics"]["total_carbon_kg"]
    max_l = max(v["metrics"]["avg_latency_ms"] for v in summary.values())

    jobs = pd.read_csv(JOBS_CSV)
    carbon = CarbonSeries()
    latency = load_latency_matrix()

    cache: dict[tuple, dict] = {}

    def metrics_for(alpha: float, dist: float | None) -> dict:
        key = (round(alpha, 2), dist)
        if key not in cache:
            t0 = time.time()
            cfg = SimConfig(alpha=key[0], dist_max_km=dist)
            cache[key] = run_sim(jobs, carbon, latency, cfg)["metrics"]
            print(f"  α={key[0]:.2f} dist={dist}: "
                  f"{cache[key]['total_carbon_kg']:.1f}kg "
                  f"{cache[key]['avg_latency_ms']:.2f}ms ({time.time()-t0:.0f}s)", flush=True)
        return cache[key]

    def score(m: dict) -> float:
        return W * (1 - m["total_carbon_kg"] / max_c) + (1 - W) * (1 - m["avg_latency_ms"] / max_l)

    rows = []
    for pol_name, dist in POLICIES:
        print(f"[{pol_name}] 1단계: 0.25~0.75 (0.05 간격)", flush=True)
        stage1 = [round(a, 2) for a in np.arange(0.25, 0.751, 0.05)]
        best1 = max(stage1, key=lambda a: score(metrics_for(a, dist)))

        print(f"[{pol_name}] 2단계: {best1}±0.04 (0.01 간격)", flush=True)
        stage2 = [round(a, 2) for a in np.arange(best1 - 0.04, best1 + 0.041, 0.01)
                  if 0.0 <= round(a, 2) <= 1.0]
        for a in stage2:
            metrics_for(a, dist)

        tested = sorted(a for (a, d) in cache if d == dist)
        best = max(tested, key=lambda a: score(metrics_for(a, dist)))
        bm = metrics_for(best, dist)
        print(f"[{pol_name}] ⭐ 최적 α = {best:.2f} (절감률 {score(bm)*100:.2f}%) "
              f"— {bm['total_carbon_kg']:.1f}kg, {bm['avg_latency_ms']:.2f}ms\n", flush=True)
        rows += [dict(policy=pol_name, alpha=a,
                      total_carbon_kg=metrics_for(a, dist)["total_carbon_kg"],
                      avg_latency_ms=metrics_for(a, dist)["avg_latency_ms"],
                      dropped=metrics_for(a, dist)["dropped"]) for a in tested]

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "fine_alpha.csv", index=False)
    print(f"저장: {RESULTS_DIR}/fine_alpha.csv")


if __name__ == "__main__":
    main()
