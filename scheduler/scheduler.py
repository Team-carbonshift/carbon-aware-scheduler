"""핵심 스케줄러 알고리즘 (time-shift).

이 모듈의 책임은 "어느 리전에서 실행할지"가 아니라 "언제 실행할지"뿐이다.
리전 배정(공간 이동)은 로드밸런서 담당이며, 여기서는 로드밸런서가 데이터로 넘겨준
리전 값(job["region"], job["carbon_region"])을 그대로 사용한다. LB 알고리즘은 구현하지 않는다.

- time-shift 알고리즘 (Step 1~5)
- 비교군 3종을 하나의 진입점(schedule_job)에서 mode로 분기
"""

import math

from . import carbon_forecast
from .config import FORECAST_HORIZON


def compute_alpha(k):
    """k(중요도 1~5) -> alpha. k가 작을수록(여유 있을수록) 탄소 가중치가 커진다."""
    return (6 - k) / 5


def get_forecast_window(carbon_series, t_now, horizon=FORECAST_HORIZON):
    """t_now 시점에 carbon_forecast.get_carbon_forecast(horizon)이 내놓을 예측값.

    실서비스에서는 이 함수를 안 거치고 스케줄러가 직접
    `carbon_forecast.get_carbon_forecast(24)`를 호출하면 된다. 시뮬레이션에서는
    "미래의 여러 t_now 시점"을 재현해야 해서, 정답 시계열(carbon_series)을
    master_series로 넘겨 그 시점의 예측을 재구성한다.
    """
    result = carbon_forecast.get_carbon_forecast(
        horizon=horizon, master_series=carbon_series, now_hour=t_now
    )
    return result["forecast"]


def _mean_carbon(series, start_hour, duration, series_len):
    start_idx = max(0, int(round(start_hour)))
    duration_int = int(math.ceil(duration))
    end_idx = min(start_idx + duration_int, series_len)
    window = series[start_idx:end_idx]
    if not window:
        idx = min(start_idx, series_len - 1)
        return series[idx]
    return sum(window) / len(window)


def compute_time_shift(job, forecast_window, t_now):
    """Step 1~5: 실행 가능 윈도우 안에서 탄소/지연 가중 score가 최소인 슬롯을 찾는다."""
    duration = job["duration"]
    region = job["region"]
    series = forecast_window[region]

    # Step 1. 실행 가능 윈도우
    t_earliest = job["submit_time"]
    t_latest = job["deadline"] - duration

    if t_latest <= t_earliest:
        return t_earliest  # 즉시 실행 fallback

    # Step 5. LSTM 예측 범위 초과 처리
    if t_earliest > t_now + FORECAST_HORIZON:
        return t_earliest  # 즉시 실행 fallback

    horizon_limit = t_now + len(series) - duration
    t_search_limit = min(t_latest, t_now + FORECAST_HORIZON, horizon_limit)
    if t_search_limit <= t_earliest:
        return t_earliest

    rel_start = max(0, math.ceil(t_earliest - t_now))
    rel_end = math.floor(t_search_limit - t_now)
    candidates = list(range(rel_start, rel_end + 1))
    if not candidates:
        return t_earliest

    # Step 2. 탄소비용 계산
    duration_int = int(math.ceil(duration))
    carbon_cost = {}
    for i in candidates:
        window = series[i:i + duration_int]
        carbon_cost[i] = sum(window) / len(window) if window else series[min(i, len(series) - 1)]

    c_min, c_max = min(carbon_cost.values()), max(carbon_cost.values())
    denom_d = t_latest - t_earliest
    alpha = compute_alpha(job["k"])

    # Step 3, 4. 지연비용 + score, argmin
    best_i, best_score = candidates[0], float("inf")
    for i in candidates:
        t = t_now + i
        c_hat = 0.0 if c_max == c_min else (carbon_cost[i] - c_min) / (c_max - c_min)
        d_hat = (t - t_earliest) / denom_d if denom_d > 0 else 0.0
        score = alpha * c_hat + (1 - alpha) * d_hat
        if score < best_score:
            best_score, best_i = score, i

    return t_now + best_i


def schedule_job(job, carbon_series, mode):
    """job 하나를 mode(비교군)에 따라 스케줄링하고 결과 dict를 반환한다.

    리전은 로드밸런서가 준 데이터에서만 읽는다:
      - job["region"]        : 원본 배정 (단순 LB baseline)
      - job["carbon_region"] : 탄소 인식 배정 (로드밸런서의 spatial-shift 결과)
    스케줄러가 결정하는 것은 scheduled_start(실행 시각)뿐이다.
    """
    duration = job["duration"]
    t_now = job["submit_time"]

    if mode == "simple_lb_immediate":
        region = job["region"]
        scheduled_start = job["submit_time"]
    elif mode == "carbon_lb_immediate":
        region = job.get("carbon_region") or job["region"]
        scheduled_start = job["submit_time"]
    elif mode == "carbon_lb_timeshift":
        region = job.get("carbon_region") or job["region"]
        forecast_window = get_forecast_window(carbon_series, t_now)
        job_with_region = dict(job, region=region)
        scheduled_start = compute_time_shift(job_with_region, forecast_window, t_now)
    else:
        raise ValueError(f"unknown mode: {mode}")

    actual_series = carbon_series[region]
    carbon_rate = _mean_carbon(actual_series, scheduled_start, duration, len(actual_series))
    carbon_emitted = carbon_rate * duration
    delay = scheduled_start - job["submit_time"]
    slo_satisfied = (scheduled_start + duration) <= job["deadline"]

    return {
        "job_id": job["id"],
        "k": job["k"],
        "region": region,
        "submit_time": job["submit_time"],
        "duration": duration,
        "scheduled_start": scheduled_start,
        "delay": delay,
        "carbon_emitted": carbon_emitted,
        "slo_satisfied": slo_satisfied,
    }
