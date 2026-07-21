# Carbon Forecast Module

8개 리전(KR, US-CAL-CISO, US-TEX-ERCO, US-NY-NYIS, FR, DE, IN, JP)의
향후 24시간 탄소강도(gCO₂/kWh)를 예측하는 LSTM 인터페이스 모듈입니다.

- Train: 2021~2023 / Val: 2024 / Test: 2025 (rolling-forecast evaluation)
- 입력 피처 10개: carbon_intensity + cfe_pct_norm + re_pct_norm + 시간 피처(sin/cos×3, is_holiday)

---

## 설치

```bash
pip install -r requirements.txt
```

---

## B (스케줄러) 사용법

```python
from carbon_forecast import load_all_models, get_forecast_at
import pandas as pd

models, scalers = load_all_models('./models')

result = get_forecast_at(
    t=pd.Timestamp('2025-03-15 14:00'),
    models=models,
    scalers=scalers,
    all_df=carbon_df
    # 필수 컬럼: timestamp, region, carbon_intensity, cfe_pct, re_pct
)
# result['forecast']['KR'] = [24개 예측값, gCO₂/kWh]
```

### 반환 형식

```json
{
  "generated_at": "2025-03-15T14:00:00",
  "forecast": {
    "KR": [352.1, 348.9, ..., 310.2],
    "FR": [78.3, 76.1, ..., 82.4],
    ...
  }
}
```

---

## 주의사항

- **index 0 = 요청 시각 t 시점 자체의 예측값** (t+1이 아님)
  `index 23` = t로부터 23시간 후
- `all_df`에는 **`cfe_pct`, `re_pct` 원본 컬럼이 반드시 포함**되어야 합니다.
  (timestamp만으로는 자동 생성 불가능한 실측값이라 C의 데이터 파이프라인에서 공급)
- 예측 시점 `t` 기준 **이전 168시간(1주일)** 데이터가 `all_df`에 있어야 합니다.
  데이터 시작일 기준으로는 `시작일 + 168시간` 이후부터 예측 가능합니다.
  (예: test가 2025-01-01부터 시작하면 2025-01-08 00:00 이후로 예측 요청)
- 리전 키는 `carbon_forecast.py`의 `REGIONS` 딕셔너리 기준입니다:
  `KR, US-CAL-CISO, US-TEX-ERCO, US-NY-NYIS, FR, DE, IN, JP`

---

## 배포 체크리스트

- [ ] Drive에서 `models/` 폴더 통째로 다운로드
- [ ] `carbon_forecast.py`의 `REGIONS` 키가 실제 모델 파일명과 일치하는지 확인
- [ ] `INPUT_SIZE = 10` 반영됐는지 확인 (cfe_pct_norm, re_pct_norm 포함)
- [ ] 로컬(VSCode 등)에서 `load_all_models()` 한 번 실행해 에러 없는지 확인 후 push
      — `carbon_forecast.py`는 코랩이 아닌 환경에서 처음 돌아가는 코드이므로
        push 전 로드 테스트 필수
