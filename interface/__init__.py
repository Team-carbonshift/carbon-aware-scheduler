"""모듈 간 인터페이스 계층.

3개 모듈(LSTM · 로드밸런서 · 스케줄러)이 서로의 내부 구현을 몰라도 되도록,
데이터 계약과 표기 변환을 이 패키지 한 곳에 모아둔다.

    regions             : 리전 표기 통합 (LB 표기 ↔ LSTM/표준 코드 ↔ ISO-3)
    carbon_forecast_api : LSTM 예측 경계 (실모델 또는 더미)
    lb_assignment       : 로드밸런서 배정 결과 로딩
"""

from . import carbon_forecast_api, lb_assignment, regions  # noqa: F401
