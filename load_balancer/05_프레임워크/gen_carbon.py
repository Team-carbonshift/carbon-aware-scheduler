"""
임의 탄소강도 데이터 생성기 (실데이터 도착 전 placeholder).

- 해상도 15분, 기간 8일 (jobs.csv 7일 + 최장 실행 여유), t=0 = 2026-01-01 00:00 UTC.
- 리전별 현실적 프로파일: 발전 믹스 기반 base 수준 + 현지시간 일주기(태양광 낮 하락,
  저녁 수요 피크) + 일 단위 랜덤워크(바람 좋은 날/나쁜 날) + AR(1) 노이즈.
- SEED 고정 → 재현 가능.

⚠️ 실제 탄소강도 데이터(ElectricityMaps / WattTime 등)를 받으면 이 파일 대신
   같은 스키마(time_s, 리전 8열, 단위 gCO2/kWh)의 CSV로 교체하면 끝.

출력: data/carbon_intensity.csv
"""
import numpy as np
import pandas as pd

from config import DATA_DIR, REGIONS, UTC_OFFSET

SEED = 42
STEP_S = 900          # 15분
N_DAYS = 8
OUT_PATH = DATA_DIR / "carbon_intensity.csv"

# 리전별 프로파일 (gCO2/kWh) — 대략적 현실 수준
#   base: 평균 / solar: 낮(현지 12~15시) 태양광 하락폭 / evening: 저녁(19시) 수요 피크
#   day_walk: 일 단위 변동폭(풍력 비중 클수록 큼) / noise: 15분 노이즈
PROFILES = {
    #            base  solar evening day_walk noise
    "US_West":    (240,  100,   60,     25,    8),   # 캘리포니아: 태양광 duck curve
    "US_Central": (400,   40,   45,     60,   14),   # 텍사스: 풍력 변동 큼
    "US_East":    (360,   25,   50,     30,   10),   # 버지니아: 가스 중심
    "France":     ( 55,    8,   10,      8,    3),   # 원전: 낮고 평탄
    "Germany":    (340,   90,   50,     70,   15),   # 태양광+풍력 변동 큼
    "Korea":      (430,   35,   55,     25,   10),   # 석탄/가스
    "India":      (680,   60,   45,     35,   12),   # 석탄 중심, 최고 수준
    "Japan":      (470,   70,   55,     30,   10),   # 가스/석탄 + 태양광 증가
}


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    t = np.arange(0, N_DAYS * 86400, STEP_S, dtype=float)
    out = {"time_s": t}

    for region in REGIONS:
        base, solar, evening, day_walk, noise_sd = PROFILES[region]
        local_h = ((t / 3600.0) + UTC_OFFSET[region]) % 24.0

        # 태양광: 현지 6~18시 반원 곡선(13시 최대) 만큼 하락
        sun = np.clip(np.cos((local_h - 13.0) / 7.0 * (np.pi / 2)), 0, None)
        # 저녁 피크: 19시 중심 가우시안
        eve = np.exp(-0.5 * (((local_h - 19.0 + 12) % 24 - 12) / 2.0) ** 2)

        # 일 단위 랜덤워크 (하루 안에서는 선형 보간)
        walk_daily = np.cumsum(rng.normal(0, day_walk, N_DAYS + 1))
        walk = np.interp(t / 86400.0, np.arange(N_DAYS + 1), walk_daily)

        # AR(1) 노이즈
        eps = rng.normal(0, noise_sd, len(t))
        ar = np.zeros(len(t))
        for i in range(1, len(t)):
            ar[i] = 0.85 * ar[i - 1] + eps[i]

        ci = base - solar * sun + evening * eve + walk + ar
        out[region] = np.clip(ci, 15.0, None).round(1)

    return pd.DataFrame(out)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}  ({len(df)}행 = {N_DAYS}일 × 15분)")
    print(df[REGIONS].describe().loc[["mean", "min", "max"]].round(0))
