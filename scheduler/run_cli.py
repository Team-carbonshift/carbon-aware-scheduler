"""터미널에서 빠르게 3개 비교군을 돌려보는 스크립트.

data/job/jobs_routed_alpha_auto.csv(로드밸런서가 리전까지 배정한 버전)가 있으면
그걸 쓰고, 없으면 data/job/jobs.csv(원본)를 쓴다. 총 2,800개.

사용법:
    python run_cli.py
"""

import os

import pandas as pd

from scheduler import carbon_forecast, data_loader, metrics, simulator

JOB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "job")
ROUTED_CSV = os.path.join(JOB_DIR, "jobs_routed_alpha_auto.csv")
JOBS_CSV = os.path.join(JOB_DIR, "jobs.csv")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CARBON_CSV = os.path.join(_REPO_ROOT, "carbon-forecast-LSTM", "data", "carbon_intensity_demo.csv")
BASE_TIME = pd.Timestamp("2026-01-01 00:00:00")


def load_real_carbon_series(total_hours):
    """실제 탄소강도 CSV -> {region: [값 by 시간인덱스]} (시뮬레이션 탄소 회계용)."""
    df = pd.read_csv(REAL_CARBON_CSV, parse_dates=["timestamp"])
    end = BASE_TIME + pd.Timedelta(hours=total_hours)
    df = df[(df["timestamp"] >= BASE_TIME) & (df["timestamp"] < end)].copy()

    series = {}
    for region, grp in df.groupby("region"):
        grp = grp.sort_values("timestamp")
        arr = [None] * total_hours
        for row in grp.itertuples(index=False):
            h = int(round((row.timestamp - BASE_TIME).total_seconds() / 3600))
            if 0 <= h < total_hours:
                arr[h] = float(row.carbon_intensity)
        # 빈 슬롯 forward-fill
        last = next((v for v in arr if v is not None), 400.0)
        for i in range(total_hours):
            if arr[i] is None:
                arr[i] = last
            else:
                last = arr[i]
        series[region] = arr
    return series


def main():
    if os.path.exists(ROUTED_CSV):
        jobs = data_loader.load_routed_jobs_csv(ROUTED_CSV)
    else:
        jobs = data_loader.load_jobs_csv(JOBS_CSV)
    horizon_hours = max(j["deadline"] for j in jobs) + 24
    total_hours = int(horizon_hours) + 48

    if os.path.exists(REAL_CARBON_CSV):
        carbon_series = load_real_carbon_series(total_hours)
        print(f"탄소 회계: 실데이터 ({REAL_CARBON_CSV})")
    else:
        carbon_series = carbon_forecast.generate_master_series(total_hours)
        print("탄소 회계: 더미 데이터 (실데이터 CSV 없음)")

    results_by_mode = simulator.run_all_modes(jobs, carbon_series)
    metrics.print_comparison(results_by_mode)


if __name__ == "__main__":
    main()
