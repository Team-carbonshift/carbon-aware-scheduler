"""기준 job 데이터(jobs.csv / jobs_routed_alpha_auto.csv) 로딩/전처리.

data/job/jobs.csv (gen_jobs.py로 생성)를 기준 데이터셋으로 사용한다.
로드밸런서 팀이 실제 리전 배정 결과를 붙여 넘겨준 jobs_routed_alpha_auto.csv가
있으면 그걸 우선 사용한다 (비교군2·3에서 우리가 직접 리전을 고르지 않고
로드밸런서의 실제 결정을 그대로 받아쓰기 위함).
탄소강도 예측은 scheduler/carbon_forecast.py 쪽 책임이다 (LSTM 연동 지점).
"""

import pandas as pd

from .config import LB_TO_ZONE


def load_jobs_csv(csv_path):
    """gen_jobs.py가 만든 jobs.csv를 읽는다 (시간 값은 전부 초 단위 → 시간으로 변환).

    L_max는 job마다 랜덤이므로 그대로 사용하고, region은 LB 표기(예: "Korea")를
    LSTM 인터페이스가 쓰는 zone 코드("KR")로 변환해둔다 (config.LB_TO_ZONE).
    deadline = submit_time + L_max + duration (README_jobs.md 유도 규칙).

    이 파일에는 로드밸런서의 탄소 인식 리전 배정이 없으므로 carbon_region은 None이며,
    이 경우 비교군2·3도 원본 region을 그대로 쓴다(스케줄러는 LB를 계산하지 않는다).
    탄소 인식 배정을 쓰려면 jobs_routed_alpha_auto.csv(load_routed_jobs_csv)를 사용한다.
    """
    df = pd.read_csv(csv_path)
    jobs = []
    for row in df.itertuples(index=False):
        submit_h = row.submit_time / 3600.0
        duration_h = row.duration / 3600.0
        l_max_h = row.L_max / 3600.0
        region = LB_TO_ZONE.get(str(row.region), str(row.region))
        jobs.append({
            "id": str(row.job_name),
            "submit_time": submit_h,
            "duration": duration_h,
            "k": int(row.k),
            "L_max": l_max_h,
            "region": region,
            "carbon_region": None,
            "lb_alpha": None,
            "deadline": submit_h + l_max_h + duration_h,
        })
    return jobs


def load_routed_jobs_csv(csv_path):
    """로드밸런서가 실제 리전 배정을 붙여 넘겨준 jobs_routed_alpha_auto.csv를 읽는다.

    - region: 원본 LB 표기(탄소 인식 이전) -> 비교군1(단순 LB) baseline에 사용
    - 배정: 로드밸런서가 슬롯별 파레토 무릎점 alpha로 계산한 최종 배정 리전
            -> 비교군2·3(탄소 인식)에서 그대로 사용, 우리가 다시 계산하지 않음
    - α: 로드밸런서 쪽 슬롯별(공간 이동) alpha. 우리 스케줄러의 job별(k 기반,
         시간 이동) alpha와는 다른 값이라 스케줄링 로직엔 쓰지 않고 참고용으로만 보관.
    """
    df = pd.read_csv(csv_path)
    jobs = []
    for row in df.itertuples(index=False):
        submit_h = row.submit_time / 3600.0
        duration_h = row.duration / 3600.0
        l_max_h = row.L_max / 3600.0
        region = LB_TO_ZONE.get(str(row.region), str(row.region))
        carbon_region = LB_TO_ZONE.get(str(row.배정), str(row.배정))
        jobs.append({
            "id": str(row.job_name),
            "submit_time": submit_h,
            "duration": duration_h,
            "k": int(row.k),
            "L_max": l_max_h,
            "region": region,
            "carbon_region": carbon_region,
            "lb_alpha": float(row.α),
            "deadline": submit_h + l_max_h + duration_h,
        })
    return jobs
