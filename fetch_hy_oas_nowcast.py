"""
최근 90일로 회귀 재학습 후, 마지막 확정 OAS 이후 오늘까지 누적 변화로 나우캐스트 계산
결과: data/hy_oas_nowcast.json
"""
import datetime, os, json, urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

FRED_KEY = os.environ['FRED_KEY']
TODAY = datetime.date.today()
START = TODAY - datetime.timedelta(days=120)

def fetch_fred(series_id, start, end):
    url = (f"https://api.stlouisfed.org/fred/series/observations?"
           f"series_id={series_id}&api_key={FRED_KEY}"
           f"&observation_start={start}&observation_end={end}&file_type=json")
    with urllib.request.urlopen(url, timeout=15) as r:
        d = json.load(r)
    obs = [(o['date'], float(o['value'])) for o in d['observations'] if o['value'] != '.']
    return pd.Series({pd.Timestamp(dd): v for dd, v in obs}).sort_index()

oas = fetch_fred("BAMLH0A0HYM2", START, TODAY)
vix = fetch_fred("VIXCLS", START, TODAY)
hyg = yf.Ticker("HYG").history(start=START, end=TODAY + datetime.timedelta(days=1))["Close"]
ief = yf.Ticker("IEF").history(start=START, end=TODAY + datetime.timedelta(days=1))["Close"]
hyg.index = hyg.index.tz_localize(None); ief.index = ief.index.tz_localize(None)

hyg_ret = hyg.pct_change() * 100
ief_ret = ief.pct_change() * 100
vix_diff = vix.diff()
oas_diff = oas.diff() * 100

df = pd.DataFrame({"hyg_ret": hyg_ret, "ief_ret": ief_ret, "vix_diff": vix_diff, "oas_diff": oas_diff}).dropna()

# 최근 90일로 재학습 (마지막 확정 OAS 시점까지)
train = df.tail(90)
X = np.column_stack([train["hyg_ret"], train["ief_ret"], train["vix_diff"], np.ones(len(train))])
y = train["oas_diff"].values
b_hyg, b_ief, b_vix, intercept = np.linalg.lstsq(X, y, rcond=None)[0]

# 마지막 확정 OAS 시점 이후, 오늘까지 누적 변화로 나우캐스트
confirmed_date = oas.index[-1]
confirmed_oas = oas.iloc[-1]
after = df[df.index > confirmed_date]

est_diff_sum = (b_hyg*after["hyg_ret"] + b_ief*after["ief_ret"] + b_vix*after["vix_diff"] + intercept).sum()
nowcast_oas = confirmed_oas + est_diff_sum / 100  # bp -> %p

result = {
    "as_of": TODAY.isoformat(),
    "confirmed_date": confirmed_date.strftime("%Y-%m-%d"),
    "confirmed_oas": round(float(confirmed_oas), 2),
    "nowcast_oas": round(float(nowcast_oas), 2),
    "gap_days": len(after),
}
os.makedirs("data", exist_ok=True)
with open("data/hy_oas_nowcast.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
