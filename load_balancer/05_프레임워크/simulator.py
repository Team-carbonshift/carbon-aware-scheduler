"""
탄소 인지 로드밸런서 시뮬레이터 (정리 노트 §1~§8의 프레임워크 구현).

매 슬롯(기본 1시간)마다:
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
import os
import sys
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import pulp

from config import REGIONS, L_NET_MAX_MS, DATA_DIR

# interface 모듈 경로 등록 (repo 루트 기준)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from interface import carbon_forecast_api as _cfa
    from interface.regions import LB_TO_REGION as _LB_TO_STD, REGION_TO_LB as _STD_TO_LB
    _INTERFACE_OK = True
except ImportError:
    _INTERFACE_OK = False

_REAL_CSV = os.path.join(_REPO_ROOT, "carbon-forecast-LSTM", "data", "carbon_intensity_demo.csv")
_BASE_TIME = pd.Timestamp("2026-01-01 00:00:00")


# ────────────────────────── 탄소강도 시계열 ──────────────────────────
class CarbonSeries:
    def __init__(self, csv_path=DATA_DIR / "carbon_intensity.csv"):
        # 합성 데이터 (폴백 / 호환성 유지)
        df = pd.read_csv(csv_path)
        self.t = df["time_s"].to_numpy()
        self.step = float(self.t[1] - self.t[0])
        self.ci = {r: df[r].to_numpy() for r in REGIONS}

        # 실데이터 (탄소 회계용, 시간 인덱스 배열)
        self._real: dict | None = None
        if _INTERFACE_OK and os.path.exists(_REAL_CSV):
            self._load_real()

    def _load_real(self):
        df = pd.read_csv(_REAL_CSV, parse_dates=["timestamp"])
        df = df[df["timestamp"] >= _BASE_TIME].copy()
        real = {}
        for std_code, grp in df.groupby("region"):
            lb = _STD_TO_LB.get(std_code)
            if lb not in REGIONS:
                continue
            grp = grp.sort_values("timestamp")
            hours = ((grp["timestamp"] - _BASE_TIME).dt.total_seconds() / 3600).round().astype(int)
            arr = np.zeros(int(hours.max()) + 1)
            for h, v in zip(hours.values, grp["carbon_intensity"].values):
                arr[h] = float(v)
            real[lb] = arr
        if len(real) == len(REGIONS):
            self._real = real

    def at(self, region: str, t: float) -> float:
        """시각 t의 탄소강도. 실데이터 우선, 없으면 합성 데이터."""
        if self._real is not None:
            h = min(int(t / 3600), len(self._real[region]) - 1)
            return float(self._real[region][h])
        i = min(int(t // self.step), len(self.t) - 1)
        return float(self.ci[region][i])

    def forecast(self, t: float) -> np.ndarray:
        """슬롯 t 시점의 리전별 예측 탄소강도 (8,). LSTM 우선, 없으면 실측 반환."""
        if _INTERFACE_OK and _cfa._state["ready"]:
            try:
                pred = _cfa.get_forecast(t_hour=t / 3600.0, horizon=1)
                return np.array([pred[_LB_TO_STD[r]][0] for r in REGIONS])
            except Exception:
                pass
        return np.array([self.at(r, t) for r in REGIONS])

    def integrate_gco2(self, region: str, t0: float, t1: float, power_kw: float = 1.0) -> float:
        """[t0, t1] 실행 시 배출량 (g). ∫ CI dt × P / 3600."""
        if t1 <= t0:
            return 0.0
        if self._real is not None:
            # 실데이터: 시간 단위 step으로 적분
            h0, h1 = int(t0 / 3600), int(t1 / 3600)
            arr = self._real[region]
            total = 0.0
            for h in range(h0, min(h1 + 1, len(arr))):
                seg0 = max(t0, h * 3600.0)
                seg1 = min(t1, (h + 1) * 3600.0)
                if seg1 > seg0:
                    total += arr[min(h, len(arr) - 1)] * (seg1 - seg0)
            return total / 3600.0 * power_kw
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
    slot_s: float = 3600.0      # 재최적화 주기 (라우팅 행렬 1시간마다 갱신)
    headroom: float = 0.8       # 용량 여유 (쏠림 완화, 노트 §8B)
    slack_penalty: float = 1000.0  # 미배정 페널티 (infeasibility 방어)
    capacity: int | None = None    # 리전당 동시 실행 한도. None = 자동 산정
    cap_factor: float = 1.2        # 자동 산정: baseline 최대 동시실행 × factor
    l_net_max: float | None = None  # 네트워크 SLO 상한(ms). None = 제한 없음
    adaptive_alpha: bool = False    # True면 매 슬롯 파레토 무릎점으로 α 자동 선택
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
                blocked: np.ndarray, alpha: float | None = None) -> list[int | None]:
    """슬롯 내 job 배치 배정. 반환: job별 리전 인덱스 (None = 슬랙/드롭).
    blocked[o][j] = True 면 o발 job은 j로 못 감 (레이턴시 SLO 정책).
    alpha 미지정 시 cfg.alpha 사용 (무릎점 탐색이 후보 α를 넘겨줌).

    용량이 어떤 리전에서도 묶일 수 없으면(배치 크기 ≤ 모든 가용량) greedy가
    ILP 최적해와 동일하므로 지름길 사용. 그 외엔 PuLP/CBC.
    """
    a = cfg.alpha if alpha is None else alpha
    n = len(batch)
    cost = np.empty((n, 8))
    for i, job in enumerate(batch):
        o = job["origin_idx"]
        cost[i] = a * m_tilde + (1 - a) * l_tilde[o]
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


# ────────────────────────── 슬롯별 α 자동 선택 (파레토 무릎점) ──────────────────────────
ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def knee_slot_alpha(batch: list[dict], durations: np.ndarray, m_hat: np.ndarray,
                    m_tilde: np.ndarray, l_tilde: np.ndarray, latency: np.ndarray,
                    avail: np.ndarray, cfg: SimConfig,
                    blocked: np.ndarray) -> tuple[float, list[int | None]]:
    """이 슬롯의 파레토 무릎점 α와 그 배정을 반환.

    α 후보(0~1, 0.05 간격)마다 배정을 구해 (평균 지연, 예상 배출) 점을 찍고,
    두 축을 0~1 정규화한 뒤 이상점 (0,0)에 가장 가까운 점을 채택 — 평가 가중치
    w 없이 곡선의 기하학만으로 "급격히 꺾이는 지점"을 고른다.
    배출 추정 = 예측 강도 × duration × 1kW (실제 정산은 integrate_gco2가 담당).
    드롭 최소인 후보들 안에서만 고르고, 동률이면 작은 α(지연 우선)."""
    cand = []
    for a in ALPHA_GRID:
        picks = assign_slot(batch, m_tilde, l_tilde, avail, cfg, blocked, alpha=float(a))
        lats = [latency[j["origin_idx"]][p] for j, p in zip(batch, picks) if p is not None]
        carb = sum(m_hat[p] * d / 3600.0
                   for p, d in zip(picks, durations) if p is not None)
        drops = sum(p is None for p in picks)
        cand.append((float(a), picks, float(np.mean(lats)) if lats else 0.0, carb, drops))

    min_drop = min(c[4] for c in cand)
    cand = [c for c in cand if c[4] == min_drop]

    def norm(v: np.ndarray) -> np.ndarray:
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 1e-9 else np.zeros_like(v)

    lat_n = norm(np.array([c[2] for c in cand]))
    carb_n = norm(np.array([c[3] for c in cand]))
    i = int(np.hypot(lat_n, carb_n).argmin())
    return cand[i][0], cand[i][1]


# ────────────────────────── 시뮬레이션 본체 ──────────────────────────
def run_sim(jobs: pd.DataFrame, carbon: CarbonSeries, latency: np.ndarray,
            cfg: SimConfig, mode: str = "ilp") -> dict:
    """mode: 'ilp' = 탄소 인지 LB / 'baseline' = 원래 배정 그대로."""
    ridx = {r: i for i, r in enumerate(REGIONS)}
    l_tilde = latency / L_NET_MAX_MS
    # 이동 불가 경로 마스크 (레이턴시 SLO 정책). 홈 리전은 항상 허용.
    blocked = np.zeros((8, 8), dtype=bool)
    if cfg.l_net_max is not None:
        blocked |= latency > cfg.l_net_max
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
        a_slot = float("nan")  # 이 슬롯에 적용된 α (배치 없으면 NaN)

        if batch_df is not None:
            m_hat = carbon.forecast(t0)           # ★ LSTM 연결 지점
            m_tilde = m_hat / m_hat.max()          # 매 슬롯 max 정규화
            batch = [dict(origin_idx=ridx[r]) for r in batch_df.region]

            if mode == "baseline":
                picks = [ridx[r] for r in batch_df.region]
                a_slot = cfg.alpha
            else:
                avail = np.maximum(cap_eff - run_count, 0)
                if cfg.adaptive_alpha:
                    a_slot, picks = knee_slot_alpha(
                        batch, batch_df.duration.to_numpy(), m_hat, m_tilde,
                        l_tilde, latency, avail, cfg, blocked)
                else:
                    a_slot = cfg.alpha
                    picks = assign_slot(batch, m_tilde, l_tilde, avail, cfg, blocked)

            for (_, job), pick in zip(batch_df.iterrows(), picks):
                o = ridx[job.region]
                if pick is None:
                    ilp_score += cfg.slack_penalty
                    records.append(dict(job_name=job.job_name,
                                        submit_time=float(job.submit_time),
                                        origin=job.region,
                                        assigned=None, k=int(job.k), duration=job.duration,
                                        latency_ms=np.nan, carbon_g=0.0, dropped=True))
                    continue
                ilp_score += a_slot * m_tilde[pick] + (1 - a_slot) * l_tilde[o][pick]
                start, end = float(job.submit_time), float(job.submit_time + job.duration)
                heapq.heappush(running, (end, pick))
                run_count[pick] += 1
                records.append(dict(
                    job_name=job.job_name, submit_time=start,
                    origin=job.region, assigned=REGIONS[pick],
                    k=int(job.k), duration=float(job.duration),
                    latency_ms=float(latency[o][pick]),
                    carbon_g=carbon.integrate_gco2(REGIONS[pick], start, end),
                    dropped=False,
                ))

        # 슬롯 스냅샷: 리전별 실행 수 × 탄소강도 → 배출률 (시각화용 근사)
        ci_now = np.array([carbon.at(r, t0) for r in REGIONS])
        slot_series.append(dict(time_s=t0, alpha=a_slot,
                                emission_g_per_h=float((run_count * ci_now).sum()),
                                **{f"run_{r}": int(c) for r, c in zip(REGIONS, run_count)}))

    res = pd.DataFrame(records)
    ok = res[~res.dropped]
    routing = pd.crosstab(ok.origin, ok.assigned).reindex(
        index=REGIONS, columns=REGIONS, fill_value=0)

    slot_alpha = [s["alpha"] for s in slot_series]
    metrics = dict(
        mode=mode,
        alpha=(round(float(np.nanmean(slot_alpha)), 3) if cfg.adaptive_alpha
               else cfg.alpha),  # auto 모드는 슬롯 α 평균 (참고용)
        alpha_mode="auto" if cfg.adaptive_alpha else "fixed",
        capacity=cap, headroom=cfg.headroom,
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
