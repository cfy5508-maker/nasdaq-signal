"""
개선판: VIX 추가 + 노이즈구간 제외 + 3일 누적 방향성 평가
"""
import datetime, os
import numpy as np
import pandas as pd
import yfinance as yf
import urllib.request, json

FRED_KEY = os.environ['FRED_KEY']
TODAY = datetime.date.today()
TRAIN_START = TODAY - datetime.timedelta(days=180)
TRAIN_END   = TODAY - datetime.timedelta(days=60)
TEST_END    = TODAY

def fetch_fred(series_id, start, end):
    url = (f"https://api.stlouisfed.org/fred/series/observations?"
           f"series_id={series_id}&api_key={FRED_KEY}"
           f"&observation_start={start}&observation_end={end}&file_type=json")
    with urllib.request.urlopen(url, timeout=15) as r:
        d = json.load(r)
    obs = [(o['date'], float(o['value'])) for o in d['observations'] if o['value'] != '.']
    return pd.Series({pd.Timestamp(dd): v for dd, v in obs}).sort_index()

oas = fetch_fred("BAMLH0A0HYM2", TRAIN_START - datetime.timedelta(days=5), TEST_END)
vix = fetch_fred("VIXCLS", TRAIN_START - datetime.timedelta(days=5), TEST_END)

hyg = yf.Ticker("HYG").history(start=TRAIN_START, end=TEST_END + datetime.timedelta(days=1))["Close"]
ief = yf.Ticker("IEF").history(start=TRAIN_START, end=TEST_END + datetime.timedelta(days=1))["Close"]
hyg.index = hyg.index.tz_localize(None); ief.index = ief.index.tz_localize(None)

hyg_ret = hyg.pct_change() * 100
ief_ret = ief.pct_change() * 100
vix_diff = vix.diff()
oas_diff = oas.diff() * 100

df = pd.DataFrame({"hyg_ret": hyg_ret, "ief_ret": ief_ret, "vix_diff": vix_diff, "oas_diff": oas_diff}).dropna()

train = df[(df.index >= pd.Timestamp(TRAIN_START)) & (df.index <= pd.Timestamp(TRAIN_END))]
test  = df[(df.index >  pd.Timestamp(TRAIN_END))  & (df.index <= pd.Timestamp(TEST_END))]

X = np.column_stack([train["hyg_ret"], train["ief_ret"], train["vix_diff"], np.ones(len(train))])
y = train["oas_diff"].values
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
b_hyg, b_ief, b_vix, intercept = coef
print(f"베타(HYG)={b_hyg:.3f}, 베타(IEF)={b_ief:.3f}, 베타(VIX)={b_vix:.3f}, 절편={intercept:.3f}")

test = test.copy()
test["pred_diff"] = b_hyg*test["hyg_ret"] + b_ief*test["ief_ret"] + b_vix*test["vix_diff"] + intercept

# 전체(1일) 평가
mae = np.mean(np.abs(test["pred_diff"] - test["oas_diff"]))
hit_all = np.mean(np.sign(test["pred_diff"]) == np.sign(test["oas_diff"]))
print(f"[1일 전체] MAE={mae:.2f}bp, 방향적중률={hit_all*100:.1f}%")

# 노이즈 구간(±1bp 이내) 제외 평가
sig = test[test["oas_diff"].abs() > 1.0]
hit_sig = np.mean(np.sign(sig["pred_diff"]) == np.sign(sig["oas_diff"]))
print(f"[1일, ±1bp 이상만 ({len(sig)}일)] 방향적중률={hit_sig*100:.1f}%")

# 3일 누적 평가
test["pred_3d"] = test["pred_diff"].rolling(3).sum()
test["actual_3d"] = test["oas_diff"].rolling(3).sum()
t3 = test.dropna(subset=["pred_3d","actual_3d"])
hit_3d = np.mean(np.sign(t3["pred_3d"]) == np.sign(t3["actual_3d"]))
corr_3d = np.corrcoef(t3["pred_3d"], t3["actual_3d"])[0,1]
print(f"[3일 누적] 방향적중률={hit_3d*100:.1f}%, 상관계수={corr_3d:.2f}")

test.to_csv("backtest_result.csv")
