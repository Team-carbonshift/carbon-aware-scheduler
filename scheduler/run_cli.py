"""터미널에서 빠르게 비교군을 돌려보는 스크립트.

기본은 **2025년 1년치**(로드밸런서와 동일한 job 목록 + 동일한 배정 결과).
그 데이터가 없으면 저장소 안의 7일치로 폴백한다.

사용법:
    python run_cli.py
"""

import os

from scheduler import carbon_forecast, data_loader, metrics, simulator

_SCHED_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCHED_ROOT)

JOB_DIR = os.path.join(_SCHED_ROOT, "data", "job")
ROUTED_CSV = os.path.join(JOB_DIR, "jobs_routed_alpha_auto.csv")
JOBS_CSV = os.path.join(JOB_DIR, "jobs.csv")

# 2025년 1년치 (로드밸런서와 같은 소스)
YEAR_JOBS_CSV = os.path.join(_REPO_ROOT, "load_balancer", "01_데이터", "jobs.csv")
YEAR_ASSIGN_CSV = os.path.join(_REPO_ROOT, "load_balancer", "02_프레임워크",
                               "results", "assign_alpha_auto.csv")


def load_jobs():
    if os.path.exists(YEAR_JOBS_CSV) and os.path.exists(YEAR_ASSIGN_CSV):
        return data_loader.load_jobs_with_assignment(YEAR_JOBS_CSV, YEAR_ASSIGN_CSV), "2025년 1년치"
    if os.path.exists(ROUTED_CSV):
        return data_loader.load_routed_jobs_csv(ROUTED_CSV), "7일치"
    return data_loader.load_jobs_csv(JOBS_CSV), "7일치"


def main():
    jobs, scope = load_jobs()
    horizon_hours = max(j["deadline"] for j in jobs) + 24
    total_hours = int(horizon_hours) + 48

    carbon_series, is_real = carbon_forecast.load_actual_series(total_hours)
    print(f"데이터: {scope} · job {len(jobs):,}개")
    print(f"탄소 회계: {'실측' if is_real else '더미'} · 예측: {carbon_forecast.backend_info()}")

    results_by_mode = simulator.run_all_modes(jobs, carbon_series)
    metrics.print_comparison(results_by_mode)


if __name__ == "__main__":
    main()
