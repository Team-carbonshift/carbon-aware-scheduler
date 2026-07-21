"""탄소강도 예측 접근 (얇은 위임 계층).

실제 예측 로직은 이 저장소(스케줄러)에 없다. LSTM은 다른 담당의 모듈이며,
저장소 루트의 interface/carbon_forecast_api.py 가 그 경계를 맡는다.
여기서는 스케줄러가 쓰기 편한 형태로 감싸기만 한다.

    generate_master_series(total_hours) : (시뮬레이션용) 정답 시계열
    get_carbon_forecast(...)            : t 시점의 24시간 예측
"""

from interface import carbon_forecast_api

FORECAST_HORIZON = carbon_forecast_api.FORECAST_HORIZON


def generate_master_series(total_hours):
    """시뮬레이션 전 구간의 탄소강도 시계열 (더미 백엔드용)."""
    return carbon_forecast_api.generate_master_series(total_hours)


def get_carbon_forecast(horizon=FORECAST_HORIZON, master_series=None, now_hour=0,
                        prefer_lstm=True):
    """now_hour 시점의 향후 horizon시간 예측.

    실제 LSTM을 쓸 수 있는 시점이면 LSTM이, 아니면 더미가 응답한다(자동).
    반환 형식은 LSTM 계약과 동일하게 {"generated_at", "forecast"} 를 유지한다.
    """
    forecast = carbon_forecast_api.get_forecast(
        t_hour=now_hour, horizon=horizon,
        master_series=master_series, prefer_lstm=prefer_lstm)
    return {"generated_at": f"t+{now_hour}h", "forecast": forecast}


def backend_info():
    """현재 예측 백엔드 설명 (화면 표시용)."""
    return carbon_forecast_api.backend_info()


def last_backend():
    """마지막 예측이 실제로 쓴 백엔드 ('lstm' | 'dummy')."""
    return carbon_forecast_api.last_backend()
