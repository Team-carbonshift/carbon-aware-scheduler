"""LSTM ↔ 스케줄러/로드밸런서 경계.

스케줄러는 LSTM의 내부 구현(torch 모델, scaler, 168시간 입력 등)을 몰라야 한다.
이 모듈이 그 사이를 막아주며, 아래 한 가지 형태만 노출한다:

    get_forecast(t_hour, horizon=24) -> {표준리전코드: [24개 예측값(gCO2/kWh)]}

동작 방식 (2단계 폴백):
    1) 실제 LSTM 사용 가능 (torch 설치 + carbon-forecast-LSTM/models/*.pt 존재
       + 과거 탄소강도 데이터 존재) → 진짜 예측값
    2) 아니면 → 더미 예측값 (사인파 + 노이즈). 개발/데모용.

어느 쪽을 쓰고 있는지는 backend_info()로 확인할 수 있다.

LSTM 쪽 실제 시그니처:
    load_all_models(model_dir) -> (models, scalers)
    get_forecast_at(t: pd.Timestamp, models, scalers, all_df) -> {"generated_at", "forecast"}
"""

import os
import sys

import numpy as np

from .regions import REGIONS, to_region

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
LSTM_DIR = os.path.join(_REPO_ROOT, "carbon-forecast-LSTM")
LSTM_MODEL_DIR = os.path.join(LSTM_DIR, "models")

FORECAST_HORIZON = 24  # LSTM이 내놓는 예측 길이 (시간)

# 더미 생성용 리전 프로필: baseline, 일간 진폭, 위상(시), 노이즈 표준편차 (gCO2/kWh)
_DUMMY_PROFILE = {
    "US-CAL-CISO": {"base": 220, "amplitude": 60, "phase": 14, "noise": 8},
    "US-TEX-ERCO": {"base": 380, "amplitude": 40, "phase": 15, "noise": 10},
    "US-NY-NYIS":  {"base": 340, "amplitude": 50, "phase": 13, "noise": 9},
    "FR":          {"base": 60,  "amplitude": 15, "phase": 12, "noise": 4},
    "DE":          {"base": 350, "amplitude": 90, "phase": 12, "noise": 12},
    "KR":          {"base": 430, "amplitude": 30, "phase": 16, "noise": 10},
    "IN":          {"base": 620, "amplitude": 70, "phase": 13, "noise": 15},
    "JP":          {"base": 470, "amplitude": 45, "phase": 14, "noise": 11},
}
_DUMMY_SEED = 42

_state = {"backend": None, "models": None, "scalers": None, "all_df": None}


# ── 더미 백엔드 ──────────────────────────────────────────────
def generate_master_series(total_hours):
    """시뮬레이션 전체 구간(0~total_hours)의 결정적 더미 탄소강도 시계열.

    실서비스에는 필요 없다. 시뮬레이터가 '과거의 여러 t 시점에서의 예측'을
    재현하려면 정답 시계열이 통째로 있어야 해서 여기서 한 번에 만든다.
    """
    rng = np.random.default_rng(_DUMMY_SEED)
    hours = np.arange(total_hours)
    series = {}
    for region in REGIONS:
        p = _DUMMY_PROFILE[region]
        daily = p["amplitude"] * np.sin(2 * np.pi * (hours - p["phase"]) / 24)
        noise = rng.normal(0, p["noise"], size=total_hours)
        series[region] = np.clip(p["base"] + daily + noise, 5, None).tolist()
    return series


def _slice_master(master_series, t_hour, horizon):
    start = int(t_hour)
    out = {}
    for region in REGIONS:
        s = master_series[region]
        w = s[start:start + horizon]
        if len(w) < horizon:
            w = w + [w[-1] if w else 0.0] * (horizon - len(w))
        out[region] = w
    return out


# ── 실제 LSTM 백엔드 ─────────────────────────────────────────
def try_init_lstm(carbon_csv=None):
    """실제 LSTM을 쓸 수 있으면 초기화한다. 성공하면 True.

    carbon_csv: timestamp, region, carbon_intensity, cfe_pct, re_pct 컬럼을 가진
                과거 데이터 CSV 경로 (LSTM 입력 168시간용).
    """
    if not os.path.isdir(LSTM_MODEL_DIR):
        return False
    try:
        import pandas as pd  # noqa: F401
        import torch  # noqa: F401
        if LSTM_DIR not in sys.path:
            sys.path.insert(0, LSTM_DIR)
        import carbon_forecast as lstm_mod
    except Exception:
        return False

    if carbon_csv is None or not os.path.exists(carbon_csv):
        return False
    try:
        import pandas as pd
        all_df = pd.read_csv(carbon_csv, parse_dates=["timestamp"])
        models, scalers = lstm_mod.load_all_models(LSTM_MODEL_DIR)
    except Exception:
        return False

    _state.update(backend="lstm", models=models, scalers=scalers,
                  all_df=all_df, module=lstm_mod)
    return True


# ── 공개 인터페이스 ──────────────────────────────────────────
def get_forecast(t_hour, horizon=FORECAST_HORIZON, master_series=None, base_time=None):
    """t_hour 시점 기준 향후 horizon시간 예측 -> {표준리전코드: [값 …]}.

    t_hour       : 시뮬레이션 절대 시각(시간 단위 float/int)
    master_series: 더미 백엔드에서 쓸 정답 시계열 (generate_master_series 결과)
    base_time    : 실제 LSTM 백엔드에서 t_hour를 실제 시각으로 바꿀 기준 pd.Timestamp
    """
    if _state["backend"] == "lstm" and base_time is not None:
        import pandas as pd
        t = base_time + pd.Timedelta(hours=float(t_hour))
        result = _state["module"].get_forecast_at(
            t=t, models=_state["models"], scalers=_state["scalers"],
            all_df=_state["all_df"])
        return {to_region(k): v for k, v in result["forecast"].items()}

    if master_series is None:
        master_series = generate_master_series(int(t_hour) + horizon + 1)
    return _slice_master(master_series, t_hour, horizon)


def backend_info():
    """현재 어떤 백엔드로 예측하고 있는지 (화면 표시용)."""
    if _state["backend"] == "lstm":
        return "실제 LSTM 모델 (carbon-forecast-LSTM/models)"
    return "더미 예측 (사인파 + 노이즈) — LSTM 미연결"
