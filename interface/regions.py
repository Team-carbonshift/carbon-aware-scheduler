"""리전 표기 통합 — 3개 모듈이 서로 다른 이름을 쓰므로 여기서 한 번에 변환한다.

세 모듈의 표기가 각각 다르다:
    로드밸런서 : US_West, US_Central, US_East, France, Germany, Korea, India, Japan
    LSTM       : US-CAL-CISO, US-TEX-ERCO, US-NY-NYIS, FR, DE, KR, IN, JP
    스케줄러   : (이 파일의 표준 코드를 그대로 사용)

표준(canonical) 코드는 **LSTM의 zone 코드**를 따른다.
LSTM이 실제 학습된 모델 파일을 그 코드로 저장해두었기 때문에(models/KR_lstm.pt 등),
그쪽에 맞추는 것이 가장 안전하다.
"""

# (표준코드, 로드밸런서 표기, ISO-3 국가코드, 사람이 읽는 이름)
_REGION_TABLE = [
    ("US-CAL-CISO", "US_West",    "USA", "US West (California)"),
    ("US-TEX-ERCO", "US_Central", "USA", "US Central (Texas)"),
    ("US-NY-NYIS",  "US_East",    "USA", "US East (New York)"),
    ("FR",          "France",     "FRA", "France"),
    ("DE",          "Germany",    "DEU", "Germany"),
    ("KR",          "Korea",      "KOR", "Korea"),
    ("IN",          "India",      "IND", "India"),
    ("JP",          "Japan",      "JPN", "Japan"),
]

# 표준 리전 코드 (순서 고정)
REGIONS = [row[0] for row in _REGION_TABLE]

LB_TO_REGION = {row[1]: row[0] for row in _REGION_TABLE}     # "Korea" -> "KR"
REGION_TO_LB = {row[0]: row[1] for row in _REGION_TABLE}     # "KR" -> "Korea"
REGION_TO_ISO3 = {row[0]: row[2] for row in _REGION_TABLE}   # "KR" -> "KOR"
REGION_LABELS = {row[0]: row[3] for row in _REGION_TABLE}    # "KR" -> "Korea"

# 예전 표기 호환 (스케줄러가 과거에 쓰던 코드 → 표준 코드)
_LEGACY_ALIASES = {
    "US-MIDA-PJM": "US-NY-NYIS",
    "IN-NO": "IN",
    "JP-TK": "JP",
}


def to_region(name):
    """어떤 표기로 들어오든 표준 리전 코드로 변환한다.

    로드밸런서 표기("Korea"), LSTM/표준 코드("KR"), 과거 코드("IN-NO") 모두 허용.
    알 수 없는 값은 그대로 돌려준다(호출 측에서 판단).
    """
    if name in REGION_TO_LB:      # 이미 표준 코드
        return name
    if name in LB_TO_REGION:      # 로드밸런서 표기
        return LB_TO_REGION[name]
    if name in _LEGACY_ALIASES:   # 과거 스케줄러 코드
        return _LEGACY_ALIASES[name]
    return name


def to_iso3(region):
    """표준 리전 코드 -> ISO-3 국가코드 (지도 시각화용). 미국 3개 리전은 USA로 합쳐진다."""
    return REGION_TO_ISO3.get(to_region(region), region)


def label(region):
    """표준 리전 코드 -> 사람이 읽는 이름."""
    r = to_region(region)
    return REGION_LABELS.get(r, r)
