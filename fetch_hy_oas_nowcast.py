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
# VIX는 FRED(VIXCLS)도 하루 늦게 올라와서, 최신 날짜 추정에 쓰려면 yfinance에서 받는다
vix = yf.Ticker("^VIX").history(start=START, end=TODAY + datetime.timedelta(days=1))["Close"]
hyg = yf.Ticker("HYG").history(start=START, end=TODAY + datetime.timedelta(days=1))["Close"]
ief = yf.Ticker("IEF").history(start=START, end=TODAY + datetime.timedelta(days=1))["Close"]
hyg.index = hyg.index.tz_localize(None).normalize(); ief.index = ief.index.tz_localize(None).normalize()
vix.index = vix.index.tz_localize(None).normalize()

hyg_ret = hyg.pct_change() * 100
ief_ret = ief.pct_change() * 100
vix_diff = vix.diff()
oas_diff = oas.diff() * 100

# 기존 버그: 설명변수와 OAS를 한 표에 넣고 dropna()를 해서, OAS가 아직 발표 안 된 최근 날짜
# (=정작 추정이 필요한 구간)가 통째로 지워졌다. 그래서 after가 항상 비어 gap_days=0,
# 나우캐스트가 마지막 확정값을 그대로 복사만 하고 있었다.
# 설명변수 표(feat)와 학습용 표(train_df)를 분리해서, 추정 구간은 feat에서 가져온다.
feat = pd.DataFrame({"hyg_ret": hyg_ret, "ief_ret": ief_ret, "vix_diff": vix_diff}).dropna()
train_df = feat.join(oas_diff.rename("oas_diff"), how="inner").dropna()

# 최근 90일로 재학습 (마지막 확정 OAS 시점까지)
train = train_df.tail(90)
X = np.column_stack([train["hyg_ret"], train["ief_ret"], train["vix_diff"], np.ones(len(train))])
y = train["oas_diff"].values
b_hyg, b_ief, b_vix, intercept = np.linalg.lstsq(X, y, rcond=None)[0]

# 마지막 확정 OAS 시점 이후, 오늘까지 누적 변화로 나우캐스트
confirmed_date = oas.index[-1]
confirmed_oas = oas.iloc[-1]
after = feat[feat.index > confirmed_date]

est_diff_sum = (b_hyg*after["hyg_ret"] + b_ief*after["ief_ret"] + b_vix*after["vix_diff"] + intercept).sum()
nowcast_oas = confirmed_oas + est_diff_sum / 100  # bp -> %p

result = {
    "as_of": TODAY.isoformat(),
    "confirmed_date": confirmed_date.strftime("%Y-%m-%d"),
    "confirmed_oas": round(float(confirmed_oas), 2),
    "nowcast_oas": round(float(nowcast_oas), 2),
    "gap_days": len(after),
    "beta_hyg": round(float(b_hyg), 3),
    "beta_ief": round(float(b_ief), 3),
    "beta_vix": round(float(b_vix), 3),
    "intercept": round(float(intercept), 3),
    "train_window": len(train),
}
os.makedirs("data", exist_ok=True)
with open("data/hy_oas_nowcast.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# ── 베타 이력 누적 (드리프트 추적용) ──
history_path = "data/hy_oas_beta_history.json"
history = []
if os.path.exists(history_path):
    with open(history_path) as f:
        history = json.load(f)
history.append(result)
history = history[-90:]   # 최근 90회분만 유지
with open(history_path, "w") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False, indent=2))
