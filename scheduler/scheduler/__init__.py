"""탄소 인식 time-shift 스케줄러 패키지.

이 패키지는 저장소 루트의 interface/ 패키지(모듈 간 데이터 계약)를 사용한다.
어디서 실행하든 import가 되도록 여기서 저장소 루트를 sys.path에 넣어둔다.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
