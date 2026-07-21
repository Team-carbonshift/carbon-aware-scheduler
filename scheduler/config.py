"""상수 정의: L_max 테이블, 리전 목록, 시뮬레이션 기본 파라미터."""

# k(중요도) -> 최대 대기 가능 시간 (시간 단위)
L_MAX = {
    5: 1 / 3600,     # 1초
    4: 30 / 3600,    # 30초
    3: 300 / 3600,   # 5분
    2: 6,            # 6시간
    1: 24,           # 24시간
}

# LSTM 예측 인터페이스가 쓰는 electricitymaps 리전(zone) 코드.
# 스케줄러/시뮬레이터 내부에서는 이 코드를 리전 식별자로 그대로 사용한다.
REGIONS = [
    "US-CAL-CISO",
    "US-TEX-ERCO",
    "US-MIDA-PJM",
    "FR",
    "DE",
    "KR",
    "IN-NO",
    "JP-TK",
]

# jobs.csv에 적힌 LB 표기(친화적 이름) -> LSTM zone 코드.
# jobs.csv는 로드밸런서 쪽 표기를 쓰고, LSTM 인터페이스는 zone 코드를 쓰므로
# job 로딩 시점에 이 매핑으로 한 번 변환해 이후에는 zone 코드로 통일한다.
LB_TO_ZONE = {
    "US_West": "US-CAL-CISO",
    "US_Central": "US-TEX-ERCO",
    "US_East": "US-MIDA-PJM",
    "France": "FR",
    "Germany": "DE",
    "Korea": "KR",
    "India": "IN-NO",
    "Japan": "JP-TK",
}

# 화면 표시용 (zone 코드 -> 사람이 읽기 쉬운 이름)
ZONE_LABELS = {
    "US-CAL-CISO": "US_West (California)",
    "US-TEX-ERCO": "US_Central (Texas)",
    "US-MIDA-PJM": "US_East (New York)",
    "FR": "France",
    "DE": "Germany",
    "KR": "Korea",
    "IN-NO": "India",
    "JP-TK": "Japan",
}

SLOT_HOURS = 1
FORECAST_HORIZON = 24  # LSTM 예측 범위 (시간)

# 리전별 더미 탄소강도 프로필 (gCO2/kWh) - baseline, 일간 진폭, 위상(시), 노이즈 표준편차
# LSTM 연동 전까지 scheduler/carbon_forecast.py의 더미 생성기가 참고하는 값.
CARBON_PROFILE = {
    "US-CAL-CISO": {"base": 220, "amplitude": 60,  "phase": 14, "noise": 8},
    "US-TEX-ERCO": {"base": 380, "amplitude": 40,  "phase": 15, "noise": 10},
    "US-MIDA-PJM": {"base": 340, "amplitude": 50,  "phase": 13, "noise": 9},
    "FR":          {"base": 60,  "amplitude": 15,  "phase": 12, "noise": 4},
    "DE":          {"base": 350, "amplitude": 90,  "phase": 12, "noise": 12},
    "KR":          {"base": 430, "amplitude": 30,  "phase": 16, "noise": 10},
    "IN-NO":       {"base": 620, "amplitude": 70,  "phase": 13, "noise": 15},
    "JP-TK":       {"base": 470, "amplitude": 45,  "phase": 14, "noise": 11},
}

# 세계지도(choropleth)에서 국가 전체를 색칠하기 위한 ISO-3 국가 코드.
# 미국 3개 리전(California/Texas/New York)은 같은 나라(USA)로 합쳐진다.
ZONE_TO_ISO3 = {
    "US-CAL-CISO": "USA",
    "US-TEX-ERCO": "USA",
    "US-MIDA-PJM": "USA",
    "FR": "FRA",
    "DE": "DEU",
    "KR": "KOR",
    "IN-NO": "IND",
    "JP-TK": "JPN",
}

MODES = {
    "simple_lb_immediate": "비교군1: 단순 LB + 즉시 실행",
    "carbon_lb_immediate": "비교군2: 탄소 LB + 즉시 실행",
    "carbon_lb_timeshift": "비교군3(ours): 탄소 LB + time shift",
}
