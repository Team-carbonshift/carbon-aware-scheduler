"""LSTM 탄소강도 예측 인터페이스.

팀원이 만드는 실제 모듈과 함수 시그니처/반환 형식을 맞춰둔 자리다.
지금은 학습된 모델이 없으므로 사인파 + 노이즈로 만든 더미 값을 반환하고,
나중에는 이 파일을 팀원의 `carbon_forecast.py`로 그대로 교체하면 된다
(스케줄러/시뮬레이터는 이 파일의 함수만 호출하므로 다른 코드는 안 건드려도 됨).

실제 LSTM 쪽 계약 (설계 문서 기준):
    get_carbon_forecast(horizon=24) -> {
        "generated_at": "2025-07-06T14:00:00",   # 예측 생성 시각 (ISO)
        "forecast": {
            "KR": [24개 값], "US-CAL-CISO": [...], "US-TEX-ERCO": [...],
            "US-MIDA-PJM": [...], "FR": [...], "DE": [...],
            "IN-NO": [...], "JP-TK": [...],
        }
    }
    # index 0 = 현재 시각(포함), index 23 = 23시간 후. 단위 gCO2/kWh. 매 시간 갱신.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

from .config import CARBON_PROFILE, FORECAST_HORIZON, REGIONS

_DUMMY_SEED = 42  # 더미 생성용 내부 고정값. 사용자가 조절할 값이 아니라 실LSTM 대체용 placeholder.


def _dummy_value(region, hour_offset_from_epoch):
    """region의 (에폭 기준) 특정 시각 탄소강도 더미값. 일간 사인패턴 + 결정적 노이즈."""
    prof = CARBON_PROFILE[region]
    daily = prof["amplitude"] * np.sin(2 * np.pi * (hour_offset_from_epoch - prof["phase"]) / 24)
    rng = np.random.default_rng(_DUMMY_SEED + hash(region) % 1000)
    noise = rng.normal(0, prof["noise"])
    return float(max(prof["base"] + daily + noise, 5))


def generate_master_series(total_hours, profile=None):
    """시뮬레이션 전체 구간(0~total_hours)에 대한 결정적 더미 탄소강도 시계열.

    실서비스에서는 필요 없다 (매 시간 get_carbon_forecast만 호출하면 됨).
    시뮬레이터가 "미래 시점 t_now마다 그때의 24h 예측"을 재현하려면
    전체 구간의 정답 시계열이 있어야 해서 여기서 한 번에 만들어둔다.
    """
    profile = profile or CARBON_PROFILE
    rng = np.random.default_rng(_DUMMY_SEED)
    hours = np.arange(total_hours)
    series = {}
    for region in REGIONS:
        prof = profile[region]
        daily = prof["amplitude"] * np.sin(2 * np.pi * (hours - prof["phase"]) / 24)
        noise = rng.normal(0, prof["noise"], size=total_hours)
        values = np.clip(prof["base"] + daily + noise, 5, None)
        series[region] = values.tolist()
    return series


def get_carbon_forecast(horizon=FORECAST_HORIZON, master_series=None, now_hour=None):
    """팀원의 실제 LSTM 모듈과 동일한 시그니처의 예측 함수.

    운영 환경에서는 인자 없이 `get_carbon_forecast(horizon=24)`만 호출하면 되고,
    이 더미 구현은 시뮬레이션에서 특정 시점(now_hour)의 예측을 재현하기 위해
    master_series/now_hour를 추가로 받는다 (실제 LSTM 모듈에는 이 두 인자가 없다).
    """
    now = datetime.now(timezone.utc)
    if master_series is not None and now_hour is not None:
        start = int(now_hour)
        forecast = {}
        for region in REGIONS:
            series = master_series[region]
            sliced = series[start:start + horizon]
            if len(sliced) < horizon:
                pad = sliced[-1] if sliced else 0.0
                sliced = sliced + [pad] * (horizon - len(sliced))
            forecast[region] = sliced
        generated_at = (now.replace(microsecond=0) + timedelta(hours=0)).isoformat()
    else:
        base_hour = now.hour
        forecast = {
            region: [_dummy_value(region, base_hour + i) for i in range(horizon)]
            for region in REGIONS
        }
        generated_at = now.replace(microsecond=0).isoformat()

    return {"generated_at": generated_at, "forecast": forecast}
