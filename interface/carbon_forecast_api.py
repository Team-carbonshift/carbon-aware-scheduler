"""LSTM ↔ 스케줄러/로드밸런서 경계.

스케줄러는 LSTM의 내부(torch 모델, scaler, 168시간 입력 등)를 몰라야 한다.
이 모듈이 그 사이를 막아주며, 아래 한 가지 형태만 노출한다:

    get_forecast(t_hour, horizon=24) -> {표준리전코드: [24개 예측값(gCO2/kWh)]}

백엔드 2가지 (자동 선택):
    lstm  : 실제 학습된 LSTM 모델. 요청 시점 t 이전 168시간 이력이 있어야 사용 가능.
    dummy : 사인파 + 노이즈. 이력이 부족하거나 torch가 없을 때의 대체.

어떤 백엔드로 응답했는지는 backend_info() / last_backend() 로 확인한다.
"""

import os
import sys

import numpy as np

from .regions import REGIONS, to_region

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
LSTM_DIR = os.path.join(_REPO_ROOT, "carbon-forecast-LSTM")
LSTM_MODEL_DIR = os.path.join(LSTM_DIR, "models")
LSTM_DATA_CSV = os.path.join(LSTM_DIR, "data", "carbon_intensity_demo.csv")

FORECAST_HORIZON = 24  # LSTM이 내놓는 예측 길이 (시간)
SEQ_LEN = 168          # LSTM이 요구하는 입력 이력 길이 (시간)

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

_state = {
    "ready": False,          # LSTM 사용 준비됨?
    "models": None, "scalers": None, "module": None,
    "history": None,         # LSTM 입력 이력 DataFrame
    "placeholder": True,     # cfe/re가 임시 추정값인가
    "base_time": None,
    "forecastable_from": None,
    "last_backend": None,    # 마지막 get_forecast가 실제로 쓴 백엔드
    "error": None,
}


# ── 더미 백엔드 ──────────────────────────────────────────────
def generate_master_series(total_hours):
    """시뮬레이션 전체 구간(0~total_hours)의 결정적 더미 탄소강도 시계열."""
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
        s = master_series.get(region) or master_series.get(to_region(region))
        w = s[start:start + horizon]
        if len(w) < horizon:
            w = list(w) + [w[-1] if len(w) else 0.0] * (horizon - len(w))
        out[region] = list(w)
    return out


# ── 실제 LSTM 백엔드 ─────────────────────────────────────────
def init_lstm(carbon_csv=None, master_series=None, force=False):
    """실제 LSTM을 초기화한다. 성공하면 True.

    carbon_csv    : 실측 이력 CSV (timestamp, region, carbon_intensity, cfe_pct, re_pct)
    master_series : 이력이 없을 때 더미 시계열로 이력을 구성 (시뮬레이션 전 구간 커버)
    """
    if _state["ready"] and not force:
        return True
    _state["error"] = None

    if not os.path.isdir(LSTM_MODEL_DIR):
        _state["error"] = "LSTM 모델 폴더 없음"
        return False
    try:
        import pandas as pd  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:
        _state["error"] = f"torch 미설치 ({e})"
        return False

    if LSTM_DIR not in sys.path:
        sys.path.insert(0, LSTM_DIR)
    try:
        import carbon_forecast as lstm_mod
        from .carbon_history import BASE_TIME, coverage, load_history

        history, placeholder = load_history(
            carbon_csv=carbon_csv, master_series=master_series)
        models, scalers = lstm_mod.load_all_models(LSTM_MODEL_DIR)
        _, _, forecastable_from = coverage(history)
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
        return False

    _state.update(ready=True, models=models, scalers=scalers, module=lstm_mod,
                  history=history, placeholder=placeholder,
                  base_time=BASE_TIME, forecastable_from=forecastable_from)
    return True


def _lstm_forecast(t_hour, horizon):
    """실제 LSTM으로 예측. 이력이 모자라면 None."""
    import pandas as pd

    t = _state["base_time"] + pd.Timedelta(hours=float(t_hour))
    if t < _state["forecastable_from"]:
        return None
    if t > _state["history"]["timestamp"].max():
        return None
    try:
        result = _state["module"].get_forecast_at(
            t=t, models=_state["models"], scalers=_state["scalers"],
            all_df=_state["history"])
    except Exception:
        return None
    out = {to_region(k): v[:horizon] for k, v in result["forecast"].items()}
    return out if len(out) == len(REGIONS) else None


# ── 공개 인터페이스 ──────────────────────────────────────────
def get_forecast(t_hour, horizon=FORECAST_HORIZON, master_series=None,
                 prefer_lstm=True):
    """t_hour 시점 기준 향후 horizon시간 예측 -> {표준리전코드: [값 …]}.

    실제 LSTM을 쓸 수 있으면 그걸 쓰고, 아니면 더미로 자동 폴백한다.
    """
    if prefer_lstm and _state["ready"]:
        pred = _lstm_forecast(t_hour, horizon)
        if pred is not None:
            _state["last_backend"] = "lstm"
            return pred

    _state["last_backend"] = "dummy"
    if master_series is None:
        master_series = generate_master_series(int(t_hour) + horizon + 1)
    return _slice_master(master_series, t_hour, horizon)


def last_backend():
    """마지막 get_forecast 호출이 실제로 사용한 백엔드 ('lstm' | 'dummy' | None)."""
    return _state["last_backend"]


def status():
    """현재 LSTM 연결 상태 상세 (UI 표시용)."""
    return {
        "ready": _state["ready"],
        "error": _state["error"],
        "placeholder_cfe_re": _state["placeholder"],
        "forecastable_from": _state["forecastable_from"],
        "history_end": (None if _state["history"] is None
                        else _state["history"]["timestamp"].max()),
        "last_backend": _state["last_backend"],
    }


def backend_info():
    """현재 예측 백엔드 한 줄 설명."""
    if not _state["ready"]:
        return f"더미 예측 (사인파+노이즈) — LSTM 미연결: {_state['error'] or '미초기화'}"
    note = " · cfe/re는 임시 추정값" if _state["placeholder"] else ""
    return f"실제 LSTM 모델 연결됨 (carbon-forecast-LSTM/models){note}"


# import 시점에 한 번 자동 시도 (실패해도 더미로 계속 동작)
init_lstm(carbon_csv=LSTM_DATA_CSV)
