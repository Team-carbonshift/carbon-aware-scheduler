"""
공통 설정 — 리전 정의, 레이턴시 행렬 로더, 시뮬레이션 파라미터.

리전 순서는 프로젝트 전체에서 고정 (정리 노트 §0과 동일).
"""
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
JOBS_CSV = BASE_DIR.parent / "03_데이터" / "job" / "jobs.csv"
LATENCY_CSV = BASE_DIR.parent / "03_데이터" / "8x8레이턴시표.csv"

# 고정 순서 (i, j 모두 이 순서)
REGIONS = ["US_West", "US_Central", "US_East", "France", "Germany", "Korea", "India", "Japan"]

# 현지시간 변환용 UTC 오프셋 (gen_jobs.py와 동일)
UTC_OFFSET = {
    "US_West": -8.0, "US_Central": -6.0, "US_East": -5.0, "France": 1.0,
    "Germany": 1.0, "Korea": 9.0, "India": 5.5, "Japan": 9.0,
}

L_NET_MAX_MS = 244.0  # 레이턴시 정규화 분모 (행렬 최댓값)


def load_latency_matrix(path: Path = LATENCY_CSV) -> np.ndarray:
    """8x8 레이턴시 행렬(ms) 로드. 헤더 인코딩이 깨져 있어 위치 기반으로 파싱."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cells = line.split(",")
        # 데이터 행 = 첫 칸(행 이름) 뒤 8칸이 전부 숫자
        try:
            rows.append([float(c) for c in cells[1:9]])
        except ValueError:
            continue  # 헤더 행
    mat = np.array(rows)
    assert mat.shape == (8, 8), f"레이턴시 행렬 파싱 실패: {mat.shape}"
    assert np.allclose(mat, mat.T) and np.allclose(np.diag(mat), 0)
    return mat
