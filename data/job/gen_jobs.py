"""
탄소 인지 스케줄러 — job 데이터 생성기 (간결판)

열 8개
─────
job_name          식별자
submit_time       제출 시각 [초, UTC 절대축]  (t=0 = START_DATE 00:00 UTC)
duration          실행 길이 [초]              (k별 로그정규 + 클램프)
region            LB 배정 리전 (Azure 8개)
k                 중요도 5(급함)~1(여유), 전역 비율 3:2:1:1:3
L_max             지연 예산 [초] — k별 범위 내 log-uniform 랜덤
submit_local_hour 현지시간 제출 시각 [0~24h]  (검증용)
band              생성 시 시간대 "day"/"night" (검증용)

유도 규칙 (저장 안 함, 코드에서 계산):
  t_earliest = submit_time
  t_latest   = submit_time + L_max
  deadline   = submit_time + L_max + duration
  alpha      = {5:0.2, 4:0.4, 3:0.6, 2:0.8, 1:1.0}[k]

샘플링: k-first, 낮/밤 = 현지 08~20시 이분, 주력 70%(봉우리 13h/02h, σ=3h) / 반대 30%(균등)
"""

import numpy as np
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────
SEED = 42
JOBS_PER_REGION_PER_DAY = 50
N_DAYS = 7                  # 개발 7일 / 실험 365일
START_DATE = "2026-01-01"   # t=0 (UTC), 기록용
OUT_PATH = "jobs.csv"

DAY_START_H, DAY_END_H = 8, 20
MAIN_BAND_PROB = 0.7
DAY_PEAK_H, NIGHT_PEAK_H = 13.0, 2.0
PEAK_SIGMA_H = 3.0

L_MAX_RANDOM = True   # False면 범위 대신 고정값(fixed) 사용

# k → (가중치, L_max범위[s], L_max고정값[s], dur중앙[s], dur시그마, dur클램프[s])
K_TABLE = {
    5: dict(w=3, l_rng=(1, 5),          l_fix=1,     dur_med=5,    dur_sig=0.6, clamp=(1, 60)),      # 웹페이지 로딩, API 응답, 결제
    4: dict(w=2, l_rng=(15, 60),        l_fix=30,    dur_med=30,   dur_sig=0.7, clamp=(1, 300)),     # 검색 결과, 업로드 확인
    3: dict(w=1, l_rng=(120, 600),      l_fix=300,   dur_med=180,  dur_sig=0.8, clamp=(5, 1800)),    # 주문상태 업데이트, 배송알림
    2: dict(w=1, l_rng=(7200, 28800),   l_fix=21600, dur_med=1800, dur_sig=0.9, clamp=(60, 14400)),  # 정산배치, DB백업, 재고동기화
    1: dict(w=3, l_rng=(43200, 86400),  l_fix=86400, dur_med=5400, dur_sig=0.9, clamp=(300, 21600)), # 로그정리, 통계리포트, 모델재학습
}
DIURNAL_K = {5, 4, 3}

REGIONS = {  # LB 표기 : UTC offset[h]
    "US_West": -8.0, "US_Central": -6.0, "US_East": -5.0, "France": 1.0,
    "Germany": 1.0, "Korea": 9.0, "India": 5.5, "Japan": 9.0,
}

rng = np.random.default_rng(SEED)

# ── 제출시각 샘플링 ──────────────────────────────────────
def sample_local_hour(k):
    diurnal = k in DIURNAL_K
    main_is_day = diurnal
    if rng.random() < MAIN_BAND_PROB:   # 주력: 봉우리
        center = DAY_PEAK_H if diurnal else NIGHT_PEAK_H
        for _ in range(200):
            h = rng.normal(center, PEAK_SIGMA_H) % 24.0
            in_day = DAY_START_H <= h < DAY_END_H
            if in_day == main_is_day:
                band = "day" if in_day else "night"
                return h, band
        return center, ("day" if main_is_day else "night")
    # 반대: 균등
    if main_is_day:   # 반대 = 밤
        return rng.uniform(DAY_END_H, DAY_END_H + 12.0) % 24.0, "night"
    return rng.uniform(DAY_START_H, DAY_END_H), "day"

# ── 생성 ────────────────────────────────────────────────
def generate():
    ks = np.array([5, 4, 3, 2, 1])
    ws = np.array([K_TABLE[k]["w"] for k in ks], float); ws /= ws.sum()
    rows, jid = [], 0
    for region, off in REGIONS.items():
        for day in range(N_DAYS):
            for _ in range(JOBS_PER_REGION_PER_DAY):
                k = int(rng.choice(ks, p=ws)); cfg = K_TABLE[k]
                local_h, band = sample_local_hour(k)
                submit = round(day * 86400.0 + ((local_h - off) % 24.0) * 3600.0, 1)
                dur = float(np.exp(rng.normal(np.log(cfg["dur_med"]), cfg["dur_sig"])))
                dur = round(float(np.clip(dur, *cfg["clamp"])), 1)
                if L_MAX_RANDOM:
                    lo, hi = cfg["l_rng"]
                    l_max = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 1)
                else:
                    l_max = float(cfg["l_fix"])
                rows.append(dict(job_name=f"j_{jid:06d}", submit_time=submit,
                                 duration=dur, region=region, k=k, L_max=l_max,
                                 submit_local_hour=round(local_h, 2), band=band))
                jid += 1
    df = pd.DataFrame(rows).sort_values("submit_time").reset_index(drop=True)
    # 검증
    for k, cfg in K_TABLE.items():
        sub = df[df.k == k]
        assert sub.duration.between(*cfg["clamp"]).all()
        if L_MAX_RANDOM:
            assert sub.L_max.between(*cfg["l_rng"]).all()
    return df

def report(df):
    n = len(df)
    print(f"총 {n}개 | {len(REGIONS)}리전 × {N_DAYS}일 × {JOBS_PER_REGION_PER_DAY}/일")
    print(f"L_max 모드: {'k별 범위 내 랜덤(log-uniform)' if L_MAX_RANDOM else 'k별 고정값'}\n")
    print(f"{'k':>2} {'비율%':>6} {'L_max범위(실측)':>24} {'dur중앙':>9}")
    for k in [5, 4, 3, 2, 1]:
        s = df[df.k == k]
        print(f"{k:>2} {len(s)/n*100:>6.1f} {s.L_max.min():>10.1f}~{s.L_max.max():<12.1f} {s.duration.median():>8.1f}s")
    for grp, name in [(DIURNAL_K, "주간형"), ({1, 2}, "야간형")]:
        s = df[df.k.isin(grp)]
        d = (s.submit_local_hour >= DAY_START_H) & (s.submit_local_hour < DAY_END_H)
        main = d.mean() if grp == DIURNAL_K else (~d).mean()
        print(f"{name} 주력비중 {main*100:.1f}%", end="  ")
    print(f"\ndeferrable(L_max≥1h): {(df.L_max >= 3600).sum()}개 ({(df.L_max >= 3600).mean()*100:.1f}%)")

if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}\n"); report(df)
