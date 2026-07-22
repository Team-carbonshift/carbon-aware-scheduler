# =====================================================
# carbon_forecast.py
# Carbon-Aware Scheduler — LSTM 탄소강도 예측 인터페이스
# =====================================================
# 로드밸런서/스케줄러가 import해서 사용하는 독립 모듈
#
#
# 사용법:
#   from carbon_forecast import load_all_models, get_carbon_forecast, get_forecast_at
#   models, scalers = load_all_models(MODEL_DIR)
#   result = get_forecast_at(t, models, scalers, all_df)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import holidays
from datetime import datetime


# ── 상수 정의 ─────────────────────────────────────
# 코랩 STEP 1 하이퍼파라미터와 반드시 일치해야 함

REGIONS = {
    'KR':          'Korea',
    'US-CAL-CISO': 'US West (California)',
    'US-TEX-ERCO': 'US Central (Texas)',
    'US-NY-NYIS':  'US East (New York)',
    'FR':          'France',
    'DE':          'Germany',
    'IN':          'India',
    'JP':          'Japan',
}

HOLIDAY_CODES = {
    'KR':          'KR',
    'US-CAL-CISO': 'US',
    'US-TEX-ERCO': 'US',
    'US-NY-NYIS':  'US',
    'FR':          'FR',
    'DE':          'DE',
    'IN':          'IN',
    'JP':          'JP',
}

# 날씨 피처가 있는 리전 (시간별 관측치: 풍속, 일사량)
# → 나머지 5개 리전은 날씨 데이터가 없어 기존 10개 피처만 사용
WEATHER_REGIONS = ['US-TEX-ERCO', 'US-CAL-CISO', 'DE']
WEATHER_RAW_COLS     = ['wind_speed_10m', 'shortwave_radiation']
WEATHER_FEATURE_COLS = ['wind_speed_10m_norm', 'shortwave_radiation_norm']

BASE_INPUT_SIZE = 10  # carbon_intensity + cfe_pct_norm + re_pct_norm + sin/cos×3쌍 + is_holiday
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
OUTPUT_SIZE = 24   # 향후 24시간 예측
SEQ_LEN     = 168  # 입력 168시간 (1주일)

# STEP 5와 동일한 순서로 유지 (모델 가중치 shape과 순서가 대응돼야 함)
BASE_FEATURE_COLS = [
    'carbon_intensity',
    'cfe_pct_norm', 're_pct_norm',
    'sin_hour',  'cos_hour',
    'sin_dow',   'cos_dow',
    'sin_month', 'cos_month',
    'is_holiday'
]


def get_feature_cols(region: str) -> list:
    """리전별 LSTM 입력 피처 목록. WEATHER_REGIONS는 날씨 피처 2개가 추가됨."""
    if region in WEATHER_REGIONS:
        return BASE_FEATURE_COLS + WEATHER_FEATURE_COLS
    return BASE_FEATURE_COLS


def get_input_size(region: str) -> int:
    return len(get_feature_cols(region))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── CarbonLSTM 모델 정의 ──────────────────────────
# 코랩 STEP 6과 완전히 동일한 구조여야 load_state_dict 가능

class CarbonLSTM(nn.Module):
    """
    탄소강도 24시간 예측 LSTM 모델
    입력: (batch, 168, N)  ← N은 리전별로 다름 (기본 10, 날씨 리전은 12)
    출력: (batch, 24)  ← 정규화된 값, 역변환 후 gCO₂/kWh
    """

    def __init__(
        self,
        input_size:  int = BASE_INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers:  int = NUM_LAYERS,
        output_size: int = OUTPUT_SIZE,
        dropout:     float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        last_hidden  = lstm_out[:, -1, :]
        return self.fc(last_hidden)


# ── 피처 생성 헬퍼 ────────────────────────────────

def _add_time_features(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    timestamp로부터 sin/cos 시간 피처 + is_holiday 생성
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    hour      = df['timestamp'].dt.hour
    dayofweek = df['timestamp'].dt.dayofweek
    month     = df['timestamp'].dt.month

    df['sin_hour']  = np.sin(2 * np.pi * hour      / 24)
    df['cos_hour']  = np.cos(2 * np.pi * hour      / 24)
    df['sin_dow']   = np.sin(2 * np.pi * dayofweek / 7)
    df['cos_dow']   = np.cos(2 * np.pi * dayofweek / 7)
    df['sin_month'] = np.sin(2 * np.pi * month     / 12)
    df['cos_month'] = np.cos(2 * np.pi * month     / 12)

    country = HOLIDAY_CODES[region]
    years   = df['timestamp'].dt.year.unique().tolist()
    hols    = set(holidays.country_holidays(country, years=years).keys())
    df['is_holiday'] = df['timestamp'].dt.date.apply(lambda d: int(d in hols))

    if 'cfe_pct_norm' not in df.columns and 'cfe_pct' in df.columns:
        df['cfe_pct_norm'] = df['cfe_pct'] / 100.0
    if 're_pct_norm' not in df.columns and 're_pct' in df.columns:
        df['re_pct_norm'] = df['re_pct'] / 100.0

    return df


# ── 모델 + Scaler 로드 ────────────────────────────

def load_all_models(model_dir: str) -> tuple:
    """
    8개 리전 모델 + Scaler 로드

    Args:
        model_dir: 모델/Scaler 저장 경로

    Returns:
        models:          {region: CarbonLSTM}
        scalers:         {region: MinMaxScaler}   (carbon_intensity 전용)
        weather_scalers: {region: MinMaxScaler}   (날씨 피처 전용, WEATHER_REGIONS만 존재)
    """
    models  = {}
    scalers = {}
    weather_scalers = {}

    for region in REGIONS.keys():
        model_path  = os.path.join(model_dir, f'{region}_lstm.pt')
        scaler_path = os.path.join(model_dir, f'{region}_scaler.pkl')

        model = CarbonLSTM(input_size=get_input_size(region)).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        model.eval()

        scalers[region] = joblib.load(scaler_path)
        models[region]  = model

        if region in WEATHER_REGIONS:
            weather_scaler_path = os.path.join(model_dir, f'{region}_weather_scaler.pkl')
            weather_scalers[region] = joblib.load(weather_scaler_path)

    print(f"✅ 8개 리전 모델 + Scaler 로드 완료 (device={device})")
    return models, scalers, weather_scalers


# ── 단일 리전 예측 ────────────────────────────────

def predict_region(
    region:   str,
    model,
    scaler,
    input_df: pd.DataFrame,
    weather_scaler=None
) -> list:
    """
    단일 리전 향후 24시간 탄소강도 예측

    Args:
        region:   리전 키
        model:    해당 리전 CarbonLSTM
        scaler:   해당 리전 MinMaxScaler (carbon_intensity 전용)
        input_df: 최근 168시간 DataFrame
                  필수 컬럼: carbon_intensity
                  cfe_pct/re_pct (또는 _norm 버전) — 반드시 필요, 자동 생성 불가
                  region이 WEATHER_REGIONS(US-TEX-ERCO, US-CAL-CISO, DE)이면
                  wind_speed_10m, shortwave_radiation 원본값도 반드시 포함되어야 함
                  timestamp가 있으면 시간 피처는 자동 생성
        weather_scaler: 해당 리전 MinMaxScaler (날씨 피처 전용). WEATHER_REGIONS만 필요

    Returns:
        24개 예측값 리스트 (gCO₂/kWh, 역변환된 실제값)
    """
    df = input_df.copy()

    if 'sin_hour' not in df.columns:
        df = _add_time_features(df, region)

    missing = [c for c in ['cfe_pct_norm', 're_pct_norm'] if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{region}] {missing} 컬럼이 없습니다. "
            f"cfe_pct/re_pct(0~100) 또는 _norm 버전이 input_df에 반드시 포함되어야 합니다."
        )

    # carbon_intensity scaler로 정규화
    df['carbon_intensity'] = scaler.transform(
        df[['carbon_intensity']]
    )

    if region in WEATHER_REGIONS:
        missing_weather = [c for c in WEATHER_RAW_COLS if c not in df.columns]
        if missing_weather:
            raise ValueError(
                f"[{region}] {missing_weather} 컬럼이 없습니다. "
                f"이 리전은 날씨 피처로 학습되어 wind_speed_10m, shortwave_radiation "
                f"원본값이 input_df에 반드시 포함되어야 합니다."
            )
        if weather_scaler is None:
            raise ValueError(
                f"[{region}] weather_scaler가 필요합니다 (load_all_models 반환값 확인)."
            )
        df[WEATHER_FEATURE_COLS] = weather_scaler.transform(df[WEATHER_RAW_COLS])

    feature_array = df[get_feature_cols(region)].values.astype(np.float32)

    if len(feature_array) != SEQ_LEN:
        raise ValueError(
            f"[{region}] 입력 길이 불일치: {len(feature_array)} != {SEQ_LEN}"
        )

    x = torch.tensor(feature_array).unsqueeze(0).to(device)  # (1, 168, N)
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy()                  # (1, 24)

    pred_inv = scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten().tolist()

    return pred_inv


# ── 핵심 인터페이스 함수 ──────────────────────────

def get_carbon_forecast(
    models:          dict,
    scalers:         dict,
    region_data:     dict,
    weather_scalers: dict = None,
    generated_at:    str = None
) -> dict:
    """
    8개 리전 × 향후 24시간 탄소강도 예측값 반환

    Args:
        models:          {region: CarbonLSTM}
        scalers:         {region: MinMaxScaler}
        region_data:     {region: DataFrame}  ← C로부터 받은 168h 데이터
        weather_scalers: {region: MinMaxScaler}  날씨 피처 전용, WEATHER_REGIONS만 필요
        generated_at:    예측 생성 시각 문자열 (없으면 현재 시각)

    Returns:
        {
          "generated_at": "2025-07-06T14:00:00",
          "forecast": {
            "KR":          [380.2, 360.1, ..., 290.3],  # 24개
            "US-CAL-CISO": [210.3, 198.4, ..., 188.2],
            ...
          }
        }
        index 0  = generated_at 시각(t) 자체의 예측값 (현재, 요청시각)
        index 23 = t로부터 23시간 후 예측값
        단위: gCO₂/kWh
    """
    if generated_at is None:
        generated_at = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    weather_scalers = weather_scalers or {}

    forecast = {}
    for region in REGIONS.keys():
        pred = predict_region(
            region=region,
            model=models[region],
            scaler=scalers[region],
            input_df=region_data[region],
            weather_scaler=weather_scalers.get(region)
        )
        forecast[region] = [round(v, 2) for v in pred]

    return {
        "generated_at": generated_at,
        "forecast":     forecast
    }


# ── 시뮬레이터용 래퍼 함수 ────────────────────────

def get_forecast_at(
    t:               pd.Timestamp,
    models:          dict,
    scalers:         dict,
    all_df:          pd.DataFrame,
    weather_scalers: dict = None
) -> dict:
    """
    시뮬레이터용: 특정 시점 t 기준 24시간 예측
    t 이전 168시간(t 미포함)을 자동으로 슬라이싱해서 예측
    → 반환된 forecast의 index 0가 정확히 t 시각의 예측값이 됨

    Args:
        t:       예측 기준 시각 (pd.Timestamp)
        models:  {region: CarbonLSTM}
        scalers: {region: MinMaxScaler}
        all_df:  전체 탄소강도 데이터
                 컬럼: timestamp, region, carbon_intensity, cfe_pct, re_pct
                 (cfe_pct/re_pct 원본 필수 — 예측에 반드시 필요)
                 WEATHER_REGIONS(US-TEX-ERCO, US-CAL-CISO, DE)는
                 wind_speed_10m, shortwave_radiation 원본값도 필요
        weather_scalers: {region: MinMaxScaler}  날씨 피처 전용, WEATHER_REGIONS만 필요

    Returns:
        get_carbon_forecast()와 동일한 형태
    """
    region_data = {}

    for region in REGIONS.keys():
        region_df = all_df[all_df['region'] == region].copy()
        region_df = region_df.sort_values('timestamp').reset_index(drop=True)

        # t 이전 168시간 슬라이싱 (t 미포함)
        window = region_df[region_df['timestamp'] < t].tail(SEQ_LEN)

        if len(window) < SEQ_LEN:
            raise ValueError(
                f"[{region}] 입력 데이터 부족: {len(window)} < {SEQ_LEN}"
            )

        region_data[region] = window

    return get_carbon_forecast(
        models=models,
        scalers=scalers,
        region_data=region_data,
        weather_scalers=weather_scalers,
        generated_at=t.strftime('%Y-%m-%dT%H:%M:%S')
    )
