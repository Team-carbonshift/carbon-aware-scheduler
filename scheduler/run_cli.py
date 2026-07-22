"""터미널에서 빠르게 3개 비교군을 돌려보는 스크립트.

data/job/jobs_routed_alpha_auto.csv(로드밸런서가 리전까지 배정한 버전)가 있으면
그걸 쓰고, 없으면 data/job/jobs.csv(원본)를 쓴다. 총 2,800개.

사용법:
    python run_cli.py
"""

import os

from scheduler import carbon_forecast, data_loader, metrics, simulator

JOB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "job")
ROUTED_CSV = os.path.join(JOB_DIR, "jobs_routed_alpha_auto.csv")
JOBS_CSV = os.path.join(JOB_DIR, "jobs.csv")


def main():
    if os.path.exists(ROUTED_CSV):
        jobs = data_loader.load_routed_jobs_csv(ROUTED_CSV)
    else:
        jobs = data_loader.load_jobs_csv(JOBS_CSV)
    horizon_hours = max(j["deadline"] for j in jobs) + 24
    total_hours = int(horizon_hours) + 48

    carbon_series, is_real = carbon_forecast.load_actual_series(total_hours)
    print(f"탄소 회계: {'실데이터' if is_real else '더미 데이터 (실데이터 CSV 없음)'}")

    results_by_mode = simulator.run_all_modes(jobs, carbon_series)
    metrics.print_comparison(results_by_mode)


if __name__ == "__main__":
    main()
