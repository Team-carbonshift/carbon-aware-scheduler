"""job 데이터 로딩/전처리.

data/job/jobs.csv (gen_jobs.py로 생성)가 기준 job 목록이고, 리전 배정은
로드밸런서 담당 산출물이다. 로드밸런서 결과를 붙이는 일은
interface/lb_assignment.py 가 맡는다 (형식 A/B 자동 인식).

탄소강도 예측은 interface/carbon_forecast_api.py 쪽 책임이다 (LSTM 경계).
"""

import pandas as pd

from interface.lb_assignment import attach_to_jobs, load_assignments
from interface.regions import to_region


def load_jobs_csv(csv_path):
    """jobs.csv를 읽는다 (시간 값은 전부 초 단위 → 시간으로 변환).

    L_max는 job마다 랜덤이므로 그대로 사용하고, region 표기는 표준 코드로 변환한다.
    deadline = submit_time + L_max + duration (README_jobs.md 유도 규칙).

    이 파일에는 로드밸런서의 탄소 인식 배정이 없으므로 carbon_region은 None이며,
    이 경우 비교군2·3도 원본 region을 그대로 쓴다(스케줄러는 LB를 계산하지 않는다).
    """
    df = pd.read_csv(csv_path)
    jobs = []
    for row in df.itertuples(index=False):
        submit_h = row.submit_time / 3600.0
        duration_h = row.duration / 3600.0
        l_max_h = row.L_max / 3600.0
        jobs.append({
            "id": str(row.job_name),
            "submit_time": submit_h,
            "duration": duration_h,
            "k": int(row.k),
            "L_max": l_max_h,
            "region": to_region(str(row.region)),
            "carbon_region": None,
            "deadline": submit_h + l_max_h + duration_h,
        })
    return jobs


def load_jobs_with_assignment(jobs_csv, assignment_csv):
    """jobs.csv + 로드밸런서 배정 CSV를 합쳐서 읽는다.

    assignment_csv는 로드밸런서 공식 산출물(assign_*.csv)이든
    초기에 공유받은 jobs_routed_alpha_auto.csv든 상관없이 자동 인식된다.
    """
    jobs = load_jobs_csv(jobs_csv)
    return attach_to_jobs(jobs, load_assignments(assignment_csv))


def load_routed_jobs_csv(csv_path):
    """job 정보와 로드밸런서 배정이 한 파일에 같이 든 CSV(jobs_routed_alpha_auto.csv).

    - region : 원본 배정        -> 비교군1(단순 LB) baseline
    - 배정   : 탄소 인식 배정   -> 비교군2·3에서 그대로 사용 (재계산 안 함)
    """
    jobs = load_jobs_csv(csv_path)
    return attach_to_jobs(jobs, load_assignments(csv_path))
