"""탄소강도 예측 접근 (얇은 위임 계층).

실제 예측 로직은 이 저장소(스케줄러)에 없다. LSTM은 다른 담당의 모듈이며,
저장소 루트의 interface/carbon_forecast_api.py 가 그 경계를 맡는다.
여기서는 스케줄러가 쓰기 편한 형태로 감싸기만 한다.

    load_actual_series(total_hours)     : 탄소 회계용 **실측** 시계열 (없으면 더미)
    generate_master_series(total_hours) : 더미 시계열
    get_carbon_forecast(...)            : t 시점의 24시간 예측
"""

from interface import carbon_2025, carbon_forecast_api
from interface.carbon_history import load_actual_series as _load_actual

FORECAST_HORIZON = carbon_forecast_api.FORECAST_HORIZON

# 2025년 사전계산 데이터(eval_records) 캐시. 로드밸런서와 동일한 소스·시간축.
_c25 = None


def use_2025():
    """2025년 1년치 데이터를 로드해 예측/회계의 기본 소스로 삼는다.

    로드밸런서와 같은 eval_records를 쓰므로 두 모듈의 시간축·탄소값이 일치한다.
    실패하면 None을 돌려주고, 기존 경로(LSTM 라이브 / 더미)로 폴백한다.
    """
    global _c25
    if _c25 is None:
        try:
            _c25 = carbon_2025.load_2025()
        except Exception:
            _c25 = False
    return _c25 or None


def generate_master_series(total_hours):
    """시뮬레이션 전 구간의 탄소강도 시계열 (더미 백엔드용)."""
    return carbon_forecast_api.generate_master_series(total_hours)


def load_actual_series(total_hours):
    """탄소 회계(실제 배출량 계산)에 쓸 시계열.

    우선순위: 2025 실측(eval_records y_true) > 실측 CSV > 더미.
    반환: (series, is_real)
    """
    data = use_2025()
    if data is not None:
        return carbon_2025.actual_series(data, total_hours), True

    series = _load_actual(total_hours)
    if series is not None:
        return series, True
    return generate_master_series(total_hours), False


def get_carbon_forecast(horizon=FORECAST_HORIZON, master_series=None, now_hour=0,
                        prefer_lstm=True):
    """now_hour 시점의 향후 horizon시간 예측.

    2025 사전계산 예측(y_pred)이 있으면 그것을 쓰고, 없으면 LSTM 라이브/더미로 폴백한다.
    반환 형식은 LSTM 계약과 동일하게 {"generated_at", "forecast"} 를 유지한다.
    """
    data = use_2025() if prefer_lstm else None
    if data is not None:
        return {"generated_at": f"t+{now_hour}h",
                "forecast": carbon_2025.forecast_at(data, now_hour, horizon)}

    forecast = carbon_forecast_api.get_forecast(
        t_hour=now_hour, horizon=horizon,
        master_series=master_series, prefer_lstm=prefer_lstm)
    return {"generated_at": f"t+{now_hour}h", "forecast": forecast}


def backend_info():
    """현재 예측 백엔드 설명 (화면 표시용)."""
    if use_2025() is not None:
        return ("LSTM 사전계산 예측 (2025년 eval_records · y_pred) — "
                "로드밸런서와 동일 소스")
    return carbon_forecast_api.backend_info()


def last_backend():
    """마지막 예측이 실제로 쓴 백엔드."""
    if use_2025() is not None:
        return "lstm-2025"
    return carbon_forecast_api.last_backend()
