"""LSTM 입력용 과거 탄소강도 이력 구성.

LSTM은 예측 시점 t 이전 **168시간(1주일)** 의 이력을 요구하며, 필요한 컬럼은
    timestamp, region, carbon_intensity, cfe_pct, re_pct
이다. 이 중 cfe_pct(무탄소 전력 비중)·re_pct(재생에너지 비중)는 실측값이라
원래는 데이터 파이프라인 담당이 공급해야 한다.

현재 저장소에 있는 것:
    load_balancer/05_프레임워크/data/carbon_intensity.csv
        → 8리전 × 192시간, 15분 간격. carbon_intensity만 있고 cfe/re는 없음.

따라서 이 모듈은
    carbon_intensity : 위 파일(또는 더미 시계열)에서 가져오고
    cfe_pct / re_pct : **임시 추정값**으로 채운다 (탄소강도가 낮을수록 청정 비중이 높다는 단순 가정)
는 방식으로 이력을 구성한다. 임시값이라는 사실은 is_placeholder 플래그로 노출한다.

실측 cfe/re 데이터가 확보되면 load_history(...)에 그 CSV를 넘기면 그대로 대체된다.
"""

import os

import numpy as np
import pandas as pd

from .regions import REGIONS, to_region

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

# 로드밸런서가 쓰는 탄소강도 시계열 (wide 포맷: time_s + 리전 컬럼들)
LB_CARBON_CSV = os.path.join(
    _REPO_ROOT, "load_balancer", "05_프레임워크", "data", "carbon_intensity.csv")

# 시뮬레이션 t=0 에 대응하는 실제 시각 (jobs 데이터 규약: 2026-01-01 00:00 UTC)
BASE_TIME = pd.Timestamp("2026-01-01 00:00:00")

# 임시 cfe/re 추정에 쓰는 기준 탄소강도 상한 (gCO2/kWh)
_CI_CEILING = 800.0


def _estimate_clean_pct(ci):
    """탄소강도 -> (cfe_pct, re_pct) 임시 추정.

    실측값이 아니다. 탄소강도가 낮을수록 무탄소/재생 비중이 높다는 단조 가정만 반영한다.
    """
    cfe = np.clip((1.0 - np.asarray(ci) / _CI_CEILING) * 100.0, 0.0, 100.0)
    re = cfe * 0.6  # 무탄소 중 재생에너지가 대략 6할이라는 임시 가정 (원자력 등 제외분)
    return cfe, re


def load_history(carbon_csv=None, master_series=None, base_time=BASE_TIME):
    """LSTM 입력용 long-format 이력 DataFrame을 만든다.

    우선순위:
      1) carbon_csv 가 주어지고 실측 컬럼(cfe_pct, re_pct)까지 있으면 그대로 사용
      2) master_series(더미 시계열)가 주어지면 그것으로 구성 (시뮬레이션 전 구간 커버)
      3) 아무것도 없으면 로드밸런서의 carbon_intensity.csv 사용 (192시간만 커버)

    반환: (df, is_placeholder)
        df            : timestamp, region, carbon_intensity, cfe_pct, re_pct
        is_placeholder: cfe/re가 임시 추정값이면 True
    """
    if carbon_csv and os.path.exists(carbon_csv):
        df = pd.read_csv(carbon_csv, parse_dates=["timestamp"])
        if {"cfe_pct", "re_pct"}.issubset(df.columns):
            df["region"] = df["region"].map(to_region)
            return df, False
        # 실측 cfe/re가 없으면 아래 임시 추정 경로로 넘어간다
        frames = df
    elif master_series is not None:
        rows = []
        for region, series in master_series.items():
            r = to_region(region)
            ts = [base_time + pd.Timedelta(hours=h) for h in range(len(series))]
            rows.append(pd.DataFrame(
                {"timestamp": ts, "region": r, "carbon_intensity": series}))
        frames = pd.concat(rows, ignore_index=True)
    else:
        frames = _load_lb_carbon(base_time)

    cfe, re = _estimate_clean_pct(frames["carbon_intensity"].values)
    frames = frames.assign(cfe_pct=cfe, re_pct=re)
    return frames, True


def _load_lb_carbon(base_time):
    """로드밸런서의 wide 포맷 탄소강도 CSV -> long 포맷(1시간 간격)."""
    if not os.path.exists(LB_CARBON_CSV):
        raise FileNotFoundError(f"탄소강도 데이터를 찾을 수 없습니다: {LB_CARBON_CSV}")
    wide = pd.read_csv(LB_CARBON_CSV)
    wide["timestamp"] = base_time + pd.to_timedelta(wide["time_s"], unit="s")
    wide = wide.set_index("timestamp").drop(columns=["time_s"])
    hourly = wide.resample("1h").mean()  # 15분 간격 -> 1시간

    rows = []
    for col in hourly.columns:
        region = to_region(col)
        if region not in REGIONS:
            continue
        rows.append(pd.DataFrame({
            "timestamp": hourly.index,
            "region": region,
            "carbon_intensity": hourly[col].values,
        }))
    return pd.concat(rows, ignore_index=True)


def coverage(df):
    """이력이 커버하는 (시작, 끝, 예측 가능 시작) 시각을 돌려준다.

    LSTM은 168시간 이력이 필요하므로 예측은 시작 + 168h 이후부터 가능하다.
    """
    start, end = df["timestamp"].min(), df["timestamp"].max()
    return start, end, start + pd.Timedelta(hours=168)


# 실제 탄소강도(실측) CSV — 배출량 회계의 정답값
REAL_CARBON_CSV = os.path.join(
    _REPO_ROOT, "carbon-forecast-LSTM", "data", "carbon_intensity_demo.csv")


def load_actual_series(total_hours, carbon_csv=REAL_CARBON_CSV, base_time=BASE_TIME):
    """시뮬레이션 **탄소 회계용** 실측 시계열 -> {표준리전코드: [시간별 값]}.

    예측(get_forecast)이 아니라 "실제로 그 시각에 얼마나 배출됐는가"의 정답값이다.
    스케줄링 판단은 예측으로 하되 채점은 이 실측값으로 해야 앞뒤가 맞는다.

    실측 CSV가 없으면 None을 돌려준다(호출 측에서 더미로 폴백).
    """
    if not (carbon_csv and os.path.exists(carbon_csv)):
        return None

    df = pd.read_csv(carbon_csv, parse_dates=["timestamp"])
    end = base_time + pd.Timedelta(hours=total_hours)
    df = df[(df["timestamp"] >= base_time) & (df["timestamp"] < end)].copy()
    if df.empty:
        return None
    df["region"] = df["region"].map(to_region)

    series = {}
    for region, grp in df.groupby("region"):
        if region not in REGIONS:
            continue
        grp = grp.sort_values("timestamp")
        hours = ((grp["timestamp"] - base_time).dt.total_seconds() / 3600).round().astype(int)
        arr = [None] * total_hours
        for h, v in zip(hours.values, grp["carbon_intensity"].values):
            if 0 <= h < total_hours:
                arr[h] = float(v)
        # 빈 슬롯 forward-fill (자료 간격이 1시간보다 성길 때 대비)
        last = next((v for v in arr if v is not None), 400.0)
        for i in range(total_hours):
            if arr[i] is None:
                arr[i] = last
            else:
                last = arr[i]
        series[region] = arr

    return series if len(series) == len(REGIONS) else None
