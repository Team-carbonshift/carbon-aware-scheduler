"""
실험 일괄 실행: baseline + α 스윕 (0.1 간격 + 0.25/0.75)
+ α=auto (매 슬롯 파레토 무릎점으로 α 자동 선택)

결과 → results/summary.json + results/assign_<run>.csv
     + ../06_라우팅결과/jobs_routed_<α>.csv (스케줄러 인계용)
사용법: .venv/bin/python run_experiments.py
"""
import json
import time

import pandas as pd

from config import BASE_DIR, JOBS_CSV, RESULTS_DIR, DATA_DIR, load_latency_matrix
from gen_carbon import generate as gen_carbon_df
from simulator import CarbonSeries, SimConfig, run_sim

ALPHAS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]

# 스케줄러 인계용 최종 산출물: jobs.csv 원본 8열 + alpha + assigned_region
ROUTED_DIR = BASE_DIR.parent / "06_라우팅결과"


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
    runs += [(f"alpha_{a:g}", SimConfig(alpha=a, label=f"α={a:g}"), "ilp")
             for a in ALPHAS]
    # 슬롯별 파레토 무릎점 α 자동 선택 (평가 가중치 w 불사용)
    runs += [("alpha_auto", SimConfig(adaptive_alpha=True, label="α=auto"), "ilp")]

    summary = {}

    def export_routed(name, cfg, out):
        """jobs_routed_<α>.csv — jobs.csv 스키마 그대로 + alpha, assigned_region."""
        a, ss = out["assignments"], out["slot_series"]
        alpha_by_slot = {int(t // cfg.slot_s): al
                         for t, al in zip(ss.time_s, ss.alpha) if pd.notna(al)}
        routed = jobs.set_index("job_name").loc[a.job_name].reset_index()
        routed["alpha"] = (a.submit_time // cfg.slot_s).astype(int).map(alpha_by_slot).values
        routed["assigned_region"] = a.assigned.values
        label = name.replace("alpha_", "")
        routed.to_csv(ROUTED_DIR / f"jobs_routed_{label}.csv", index=False)

    def run_one(name, cfg, mode):
        t0 = time.time()
        out = run_sim(jobs, carbon, latency, cfg, mode=mode)
        out["assignments"].to_csv(RESULTS_DIR / f"assign_{name}.csv", index=False)
        out["slot_series"].to_csv(RESULTS_DIR / f"slots_{name}.csv", index=False)
        if mode == "ilp":
            ROUTED_DIR.mkdir(exist_ok=True)
            export_routed(name, cfg, out)
        summary[name] = dict(metrics=out["metrics"], routing_matrix=out["routing_matrix"])
        m = out["metrics"]
        print(f"{name:>10}: 탄소 {m['total_carbon_kg']:>8.1f} kg | "
              f"평균지연 {m['avg_latency_ms']:>6.1f} ms | 홈리전 {m['home_ratio']*100:5.1f}% | "
              f"드롭 {m['dropped']} | {time.time()-t0:.1f}s")

    for name, cfg, mode in runs:
        run_one(name, cfg, mode)

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n저장: {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
