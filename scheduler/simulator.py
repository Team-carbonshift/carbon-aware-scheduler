"""SimPy 기반 이벤트 시뮬레이션 루프.

job마다 프로세스를 하나씩 만들어 submit_time까지 대기 -> 스케줄링 결정
-> scheduled_start까지 대기 -> duration만큼 실행 순서로 진행한다.
무한 자원 가정이므로 리소스 경합은 없다.
"""

import simpy

from . import scheduler as sch
from .config import MODES


def _job_process(env, job, carbon_series, mode, results):
    wait_submit = job["submit_time"] - env.now
    if wait_submit > 0:
        yield env.timeout(wait_submit)

    decision = sch.schedule_job(job, carbon_series, mode)

    wait_start = decision["scheduled_start"] - env.now
    if wait_start > 0:
        yield env.timeout(wait_start)

    yield env.timeout(job["duration"])
    decision["finish_time"] = env.now
    results.append(decision)


def run_simulation(jobs, carbon_series, mode):
    """단일 mode에 대해 전체 job을 시뮬레이션하고 결과 리스트를 반환한다."""
    env = simpy.Environment()
    results = []
    for job in jobs:
        env.process(_job_process(env, job, carbon_series, mode, results))
    env.run()
    results.sort(key=lambda r: r["job_id"])
    return results


def run_all_modes(jobs, carbon_series, modes=None):
    """MODES에 정의된 비교군 전체(또는 지정된 modes)를 순서대로 시뮬레이션한다."""
    modes = modes or list(MODES.keys())
    return {mode: run_simulation(jobs, carbon_series, mode) for mode in modes}
