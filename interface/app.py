"""통합 대시보드 진입점 — 3개 모듈 UI를 한 앱에서 본다.

실행:
    streamlit run interface/app.py

각 모듈의 기존 Streamlit 앱을 그대로 페이지로 실행한다(코드 복제 없음).
    로드밸런서 : load_balancer/05_프레임워크/app.py
    LSTM       : interface/views/lstm_view.py
    스케줄러   : scheduler/scheduler/gui.py
    최종       : interface/views/final.py   (통합 화면, 작성 중)
"""

import os
import runpy
import sys

import streamlit as st

st.set_page_config(page_title="Carbon-Aware Scheduler", layout="wide",
                   initial_sidebar_state="expanded")

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
LB_DIR = os.path.join(REPO_ROOT, "load_balancer", "02_프레임워크")
SCHED_DIR = os.path.join(REPO_ROOT, "scheduler")
LSTM_DIR = os.path.join(REPO_ROOT, "carbon-forecast-LSTM")

# 각 모듈이 자기 폴더 기준으로 import 하므로 경로를 모두 등록한다.
# SCHED_DIR을 앞에 둬야 scheduler 패키지가 정규 패키지로 잡힌다.
for _p in (SCHED_DIR, LB_DIR, LSTM_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 하위 페이지 스크립트도 st.set_page_config를 호출하는데,
# 한 앱에서는 진입점만 호출할 수 있으므로 무력화한다.
st.set_page_config = lambda *a, **k: None

PAGES = {
    "전체 개요": os.path.join(_HERE, "views", "overview.py"),
    "로드밸런서": os.path.join(LB_DIR, "app.py"),
    "LSTM": os.path.join(_HERE, "views", "lstm_view.py"),
    "스케줄러": os.path.join(SCHED_DIR, "scheduler", "gui.py"),
    "최종": os.path.join(_HERE, "views", "final.py"),
}

with st.sidebar:
    st.markdown("### Carbon-Aware Scheduler")
    choice = st.radio("화면 선택", list(PAGES), label_visibility="collapsed")
    st.divider()

script = PAGES[choice]
if not os.path.exists(script):
    st.error(f"화면 스크립트를 찾을 수 없습니다: {script}")
else:
    # 각 모듈 앱이 자기 폴더를 기준으로 상대경로를 쓰는 경우가 있어 cwd를 맞춰준다.
    _cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(script))
        runpy.run_path(script, run_name="__main__")
    finally:
        os.chdir(_cwd)
