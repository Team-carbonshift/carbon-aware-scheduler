"""
탄소 인지 로드밸런서 시뮬레이터 (정리 노트 §1~§8의 프레임워크 구현).

매 슬롯(기본 15분)마다:
  1. 슬롯 내 제출된 job 수집 (출발지 = jobs.csv의 region)
  2. 탄소강도 M̂ 확보 — 지금은 perfect forecast placeholder (실측값 사용).
     ★ LSTM이 준비되면 CarbonSeries.forecast()만 교체하면 됨.
  3. 정규화: M̃ = M̂ / max_r M̂ (매 슬롯), l̃ = l / 244 (고정)
  4. ILP (PuLP/CBC): min Σ (α·M̃_r + (1-α)·l̃_or)·x  s.t. 용량(headroom)·전량처리(slack)
  5. 배정 확정 → job은 즉시 시작, duration 동안 해당 리전 용량 점유

baseline 모드 = jobs.csv의 원래 region 그대로 (전/후 비교의 "전").

탄소 배출 = 실행 리전의 탄소강도를 실행 구간 [start, start+duration]에 대해
적분 × 1kW 가정 (job당 전력 1kW, 상대 비교용).
"""
import heapq
import json
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import pulp

from config import REGIONS, L_NET_MAX_MS, DATA_DIR, distance_matrix


# ────────────────────────── 탄소강도 시계열 ──────────────────────────
class CarbonSeries:
    def __init__(self, csv_path=DATA_DIR / "carbon_intensity.csv"):
        df = pd.read_csv(csv_path)
        self.t = df["time_s"].to_numpy()
        self.step = float(self.t[1] - self.t[0])
        self.ci = {r: df[r].to_numpy() for r in REGIONS}

    def at(self, region: str, t: float) -> float:
        """시각 t의 탄소강도 (step 함수)."""
        i = min(int(t // self.step), len(self.t) - 1)
        return float(self.ci[region][i])

    def forecast(self, t: float) -> np.ndarray:
        """슬롯 t 시점의 리전별 예측 탄소강도 (8,).
        ★ LSTM 연결 지점: 지금은 perfect forecast (실측 반환)."""
        return np.array([self.at(r, t) for r in REGIONS])

    def integrate_gco2(self, region: str, t0: float, t1: float, power_kw: float = 1.0) -> float:
        """[t0, t1] 실행 시 배출량 (g). ∫ CI dt × P / 3600."""
        if t1 <= t0:
            return 0.0
        i0, i1 = int(t0 // self.step), int(min(t1, self.t[-1] + self.step) // self.step)
        i1 = min(i1, len(self.t) - 1)
        total = 0.0
        for i in range(i0, i1 + 1):
            seg0 = max(t0, self.t[i])
            seg1 = min(t1, self.t[i] + self.step)
            if seg1 > seg0:
                total += self.ci[region][i] * (seg1 - seg0)
        return total / 3600.0 * power_kw


# ────────────────────────── 설정 ──────────────────────────
@dataclass
class SimConfig:
    alpha: float = 0.5          # 탄소(1) ↔ 레이턴시(0) 가중치
    slot_s: float = 900.0       # 재최적화 주기
    headroom: float = 0.8       # 용량 여유 (쏠림 완화, 노트 §8B)
    slack_penalty: float = 1000.0  # 미배정 페널티 (infeasibility 방어)
    capacity: int | None = None    # 리전당 동시 실행 한도. None = 자동 산정
    cap_factor: float = 1.2        # 자동 산정: baseline 최대 동시실행 × factor
    l_net_max: float | None = None  # 네트워크 SLO 상한(ms). None = 제한 없음
    dist_max_km: float | None = None  # 이동 거리 상한(km, 논문 정책). None = 제한 없음
    label: str = ""


def auto_capacity(jobs: pd.DataFrame, cap_factor: float) -> int:
    """baseline(원래 배정)의 리전별 최대 동시 실행 수 × factor → 균일 용량."""
    peak = 0
    for r in REGIONS:
        sub = jobs[jobs.region == r]
        events = sorted(
            [(t, 1) for t in sub.submit_time] +
            [(t + d, -1) for t, d in zip(sub.submit_time, sub.duration)]
        )
        cur = best = 0
        for _, delta in events:
            cur += delta
            best = max(best, cur)
        peak = max(peak, best)
    return max(int(np.ceil(peak * cap_factor)), 4)


# ────────────────────────── 슬롯 배정 (ILP) ──────────────────────────
def assign_slot(batch: list[dict], m_tilde: np.ndarray, l_tilde: np.ndarray,
                avail: np.ndarray, cfg: SimConfig,
                blocked: np.ndarray) -> list[int | None]:
    """슬롯 내 job 배치 배정. 반환: job별 리전 인덱스 (None = 슬랙/드롭).
    blocked[o][j] = True 면 o발 job은 j로 못 감 (SLO/거리 정책).

    용량이 어떤 리전에서도 묶일 수 없으면(배치 크기 ≤ 모든 가용량) greedy가
    ILP 최적해와 동일하므로 지름길 사용. 그 외엔 PuLP/CBC.
    """
    n = len(batch)
    cost = np.empty((n, 8))
    for i, job in enumerate(batch):
        o = job["origin_idx"]
        cost[i] = cfg.alpha * m_tilde + (1 - cfg.alpha) * l_tilde[o]
        cost[i][blocked[o]] = np.inf

    feasible_any = ~np.isinf(cost)
    # greedy 지름길: 전 리전 가용량이 배치 크기 이상이면 argmin이 곧 최적
    if n <= avail.min():
        return [int(np.argmin(c)) if f.any() else None for c, f in zip(cost, feasible_any)]

    prob = pulp.LpProblem("slot_assign", pulp.LpMinimize)
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") if np.isfinite(cost[i][j]) else None
          for j in range(8)] for i in range(n)]
    s = [pulp.LpVariable(f"s_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(
        [cost[i][j] * x[i][j] for i in range(n) for j in range(8) if x[i][j] is not None]
        + [cfg.slack_penalty * s[i] for i in range(n)]
    )
    for i in range(n):  # 전량 처리 (슬랙 허용)
        prob += pulp.lpSum([v for v in x[i] if v is not None]) + s[i] == 1
    for j in range(8):  # 용량
        prob += pulp.lpSum([x[i][j] for i in range(n) if x[i][j] is not None]) <= int(avail[j])

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    out = []
    for i in range(n):
        pick = next((j for j in range(8) if x[i][j] is not None and x[i][j].value() > 0.5), None)
        out.append(pick)
    return out


# ────────────────────────── 시뮬레이션 본체 ──────────────────────────
def run_sim(jobs: pd.DataFrame, carbon: CarbonSeries, latency: np.ndarray,
            cfg: SimConfig, mode: str = "ilp") -> dict:
    """mode: 'ilp' = 탄소 인지 LB / 'baseline' = 원래 배정 그대로."""
    ridx = {r: i for i, r in enumerate(REGIONS)}
    l_tilde = latency / L_NET_MAX_MS
    # 이동 불가 경로 마스크 (SLO + 거리 정책). 홈 리전은 항상 허용.
    blocked = np.zeros((8, 8), dtype=bool)
    if cfg.l_net_max is not None:
        blocked |= latency > cfg.l_net_max
    if cfg.dist_max_km is not None:
        blocked |= distance_matrix() > cfg.dist_max_km
    np.fill_diagonal(blocked, False)
    cap = cfg.capacity or auto_capacity(jobs, cfg.cap_factor)
    cap_eff = int(np.floor(cfg.headroom * cap))  # headroom 적용 용량

    jobs = jobs.sort_values("submit_time").reset_index(drop=True)
    horizon = float(jobs.submit_time.max() + jobs.duration.max() + cfg.slot_s)
    n_slots = int(np.ceil(horizon / cfg.slot_s))

    running: list[tuple[float, int]] = []  # (end_time, region_idx) min-heap
    run_count = np.zeros(8, dtype=int)
    records = []
    slot_series = []  # 슬롯별 배출률/실행수 (시각화용)
    ilp_score = 0.0   # 목적함수 누적값 (α가 다르면 run 간 비교 불가 — 표시용)

    by_slot = jobs.groupby((jobs.submit_time // cfg.slot_s).astype(int))

    for slot in range(n_slots):
        t0 = slot * cfg.slot_s
        while running and running[0][0] <= t0:
            _, j = heapq.heappop(running)
            run_count[j] -= 1

        batch_df = by_slot.get_group(slot) if slot in by_slot.groups else None

        if batch_df is not None:
            m_hat = carbon.forecast(t0)           # ★ LSTM 연결 지점
            m_tilde = m_hat / m_hat.max()          # 매 슬롯 max 정규화
            batch = [dict(origin_idx=ridx[r]) for r in batch_df.region]

            if mode == "baseline":
                picks = [ridx[r] for r in batch_df.region]
            else:
                avail = np.maximum(cap_eff - run_count, 0)
                picks = assign_slot(batch, m_tilde, l_tilde, avail, cfg, blocked)

            for (_, job), pick in zip(batch_df.iterrows(), picks):
                o = ridx[job.region]
                if pick is None:
                    ilp_score += cfg.slack_penalty
                    records.append(dict(job_name=job.job_name, origin=job.region,
                                        assigned=None, k=int(job.k), duration=job.duration,
                                        latency_ms=np.nan, carbon_g=0.0, dropped=True))
                    continue
                ilp_score += cfg.alpha * m_tilde[pick] + (1 - cfg.alpha) * l_tilde[o][pick]
                start, end = float(job.submit_time), float(job.submit_time + job.duration)
                heapq.heappush(running, (end, pick))
                run_count[pick] += 1
                records.append(dict(
                    job_name=job.job_name, origin=job.region, assigned=REGIONS[pick],
                    k=int(job.k), duration=float(job.duration),
                    latency_ms=float(latency[o][pick]),
                    carbon_g=carbon.integrate_gco2(REGIONS[pick], start, end),
                    dropped=False,
                ))

        # 슬롯 스냅샷: 리전별 실행 수 × 탄소강도 → 배출률 (시각화용 근사)
        ci_now = np.array([carbon.at(r, t0) for r in REGIONS])
        slot_series.append(dict(time_s=t0,
                                emission_g_per_h=float((run_count * ci_now).sum()),
                                **{f"run_{r}": int(c) for r, c in zip(REGIONS, run_count)}))

    res = pd.DataFrame(records)
    ok = res[~res.dropped]
    routing = pd.crosstab(ok.origin, ok.assigned).reindex(
        index=REGIONS, columns=REGIONS, fill_value=0)

    metrics = dict(
        mode=mode, alpha=cfg.alpha, capacity=cap, headroom=cfg.headroom,
        ilp_score=round(ilp_score, 1),
        total_carbon_kg=round(float(ok.carbon_g.sum()) / 1000.0, 2),
        avg_latency_ms=round(float(ok.latency_ms.mean()), 2),
        p95_latency_ms=round(float(ok.latency_ms.quantile(0.95)), 1),
        home_ratio=round(float((ok.origin == ok.assigned).mean()), 4),
        dropped=int(res.dropped.sum()),
        n_jobs=len(res),
        energy_kwh=round(float(ok.duration.sum()) / 3600.0, 1),
        region_load={r: int((ok.assigned == r).sum()) for r in REGIONS},
    )
    return dict(metrics=metrics, assignments=res,
                routing_matrix=routing.values.tolist(),
                slot_series=pd.DataFrame(slot_series))
