"""로드밸런서 ↔ 스케줄러 경계.

로드밸런서는 "job을 어느 리전에서 돌릴지"(공간 이동)를 결정해 CSV로 넘겨준다.
스케줄러는 그 결과를 **읽기만** 하고, 리전 선택 로직은 구현하지 않는다.

지원하는 CSV 형식 2가지 (컬럼 이름만 다르고 의미는 같음):

  A) load_balancer/05_프레임워크/results/assign_*.csv   ← 로드밸런서 공식 산출물
     job_name, submit_time, origin, assigned, k, duration, latency_ms, carbon_g, dropped

  B) jobs_routed_alpha_auto.csv                        ← 초기에 공유받은 형식
     job_name, submit_time, duration, region, k, L_max, ..., α, 배정

공통 의미:
    origin / region : 원래(단순 LB) 배정 리전
    assigned / 배정 : 탄소 인식 LB가 최종 배정한 리전
"""

import pandas as pd

from .regions import to_region

# (원본 리전 컬럼 후보, 탄소인식 배정 컬럼 후보)
_ORIGIN_COLS = ("origin", "region")
_ASSIGNED_COLS = ("assigned", "배정")


def _pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_assignments(csv_path):
    """로드밸런서 결과 CSV -> {job_name: {"origin": 표준코드, "assigned": 표준코드}}.

    두 형식(A/B) 모두 자동 인식한다. 리전 표기는 표준 코드로 변환된다.
    """
    df = pd.read_csv(csv_path)
    origin_col = _pick(df, _ORIGIN_COLS)
    assigned_col = _pick(df, _ASSIGNED_COLS)
    if origin_col is None and assigned_col is None:
        raise ValueError(
            f"리전 컬럼을 찾을 수 없습니다: {csv_path}\n"
            f"기대 컬럼 {_ORIGIN_COLS} 또는 {_ASSIGNED_COLS}, 실제 {list(df.columns)}")

    out = {}
    for row in df.itertuples(index=False):
        name = str(getattr(row, "job_name"))
        origin = to_region(str(getattr(row, origin_col))) if origin_col else None
        assigned = to_region(str(getattr(row, assigned_col))) if assigned_col else origin
        out[name] = {"origin": origin or assigned, "assigned": assigned}
    return out


def attach_to_jobs(jobs, assignments):
    """job 리스트에 로드밸런서 배정 결과를 붙인다.

    job["region"]        <- origin   (단순 LB baseline용)
    job["carbon_region"] <- assigned (탄소 인식 LB 결과)
    매칭되는 배정이 없는 job은 기존 값을 유지한다.
    """
    for job in jobs:
        a = assignments.get(job["id"])
        if not a:
            continue
        if a.get("origin"):
            job["region"] = a["origin"]
        job["carbon_region"] = a.get("assigned")
    return jobs
