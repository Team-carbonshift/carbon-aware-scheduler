"""2025년 1년치 탄소강도 — 사전 계산된 LSTM 평가기록(eval_records) 기반.

로드밸런서와 **완전히 같은 데이터·규약**을 쓴다. 그래야 LB의 리전 배정과
스케줄러의 시간 이동이 같은 시간축·같은 탄소값 위에서 맞물린다.

규약 (load_balancer/02_프레임워크/simulator.py 의 CarbonSeries와 동일):
    시간축 : t=0h ↔ 2025-01-01 00:00 UTC, 1시간 해상도
    y_true : 실측 탄소강도 → **탄소 회계**(실제 배출량 계산)에 사용
    y_pred : LSTM 예측    → **스케줄링 판단**에 사용
    1월 1~7일(0~167h): 데이터가 01-08부터 시작하므로 01-08 프로파일을 반복 (합의 규약)

eval_records 한 행 = (timestamp=예측 대상 시각, horizon=몇 시간 앞, y_true, y_pred).
따라서 **발행 시각 = timestamp - horizon** 이며, 이를 이용해
"t 시점에 알 수 있었던 향후 24시간 예측"을 그대로 복원한다.
"""

import os

import numpy as np
import pandas as pd

from .regions import REGIONS, to_region

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

# eval_records 위치 (LSTM 쪽 원본 / 로드밸런서 사본 중 있는 것)
_EVAL_DIRS = [
    os.path.join(_REPO_ROOT, "carbon-forecast-LSTM", "logs"),
    os.path.join(_REPO_ROOT, "load_balancer", "01_데이터", "lstm_eval"),
]

BASE_TIME = pd.Timestamp("2025-01-01 00:00:00")  # 시뮬레이션 t=0
HORIZON = 24
WARMUP_H = 168  # 1월 1~7일 (데이터 시작 전 구간)


def _eval_dir():
    for d in _EVAL_DIRS:
        if os.path.isdir(d) and any(f.endswith("_eval_records.csv") for f in os.listdir(d)):
            return d
    raise FileNotFoundError(f"eval_records를 찾을 수 없습니다: {_EVAL_DIRS}")


def _fill_warmup(arr):
    """앞쪽 결측(1월 1~7일)을 첫 유효일 프로파일 반복으로 채운다 (LB와 동일 규약)."""
    first = int(np.argmax(~np.isnan(arr)))
    for h in range(first):
        arr[h] = arr[first + h % 24]
    return arr


def _ffill(arr):
    """내부 결측을 직전 값으로 보간."""
    mask = np.isnan(arr)
    if mask.any():
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr[mask] = arr[idx[mask]]
    return arr


def load_2025():
    """2025년 1년치 탄소 데이터를 읽는다.

    반환: dict
        actual : {리전: np.array[시간]}          실측 (탄소 회계용)
        pred24 : {리전: np.array[발행시각, 24]}  t 시점에 본 향후 24시간 예측
        n_hours: 공통 커버 시간 수
    """
    d = _eval_dir()
    actual, pred24 = {}, {}
    n_common = None

    for fname in sorted(os.listdir(d)):
        if not fname.endswith("_eval_records.csv"):
            continue
        code = fname.replace("_eval_records.csv", "")
        region = to_region(code)
        if region not in REGIONS:
            continue

        df = pd.read_csv(os.path.join(d, fname), parse_dates=["timestamp"])
        target_h = ((df["timestamp"] - BASE_TIME).dt.total_seconds() / 3600).round().astype(int)
        n = int(target_h.max()) + 1

        # (1) 실측 시계열 — horizon=1 행의 y_true 사용 (대상 시각의 실제값)
        h1 = df[df["horizon"] == 1]
        h1_h = ((h1["timestamp"] - BASE_TIME).dt.total_seconds() / 3600).round().astype(int)
        a = np.full(n, np.nan)
        a[h1_h.values] = h1["y_true"].values
        actual[region] = _ffill(_fill_warmup(a))

        # (2) 발행시각별 24시간 예측 행렬
        issue_h = target_h.values - df["horizon"].values
        p = np.full((n, HORIZON), np.nan)
        ok = issue_h >= 0
        p[issue_h[ok], df["horizon"].values[ok] - 1] = df["y_pred"].values[ok]
        # 발행시각별 결측은 실측 프로파일로 보강 (워밍업 구간 등)
        for h in range(n):
            row = p[h]
            if np.isnan(row).all():
                base_idx = np.arange(h + 1, h + 1 + HORIZON).clip(0, n - 1)
                p[h] = actual[region][base_idx]
            elif np.isnan(row).any():
                p[h] = _ffill(row)
        pred24[region] = p

        n_common = n if n_common is None else min(n_common, n)

    missing = set(REGIONS) - set(actual)
    if missing:
        raise ValueError(f"eval_records에 없는 리전: {sorted(missing)}")

    return {"actual": actual, "pred24": pred24, "n_hours": n_common}


def actual_series(data, total_hours):
    """탄소 회계용 실측 시계열 -> {리전: [시간별 값]} (스케줄러 시뮬레이터 입력 형식)."""
    out = {}
    for region, arr in data["actual"].items():
        if total_hours <= len(arr):
            out[region] = arr[:total_hours].tolist()
        else:  # 모자라면 마지막 값으로 연장
            out[region] = arr.tolist() + [float(arr[-1])] * (total_hours - len(arr))
    return out


def forecast_at(data, t_hour, horizon=HORIZON):
    """t 시점에 알 수 있었던 향후 horizon시간 예측 -> {리전: [값 …]}.

    index 0 = t 시각(현재, 실측으로 확정) · index i = t+i 시각의 LSTM 예측.
    """
    h = max(0, int(t_hour))
    out = {}
    for region in REGIONS:
        pred = data["pred24"][region]
        act = data["actual"][region]
        hh = min(h, len(pred) - 1)
        vals = [float(act[min(hh, len(act) - 1)])]           # index 0 = 현재 실측
        vals += [float(v) for v in pred[hh][: horizon - 1]]  # index 1.. = 예측
        out[region] = vals[:horizon]
    return out
