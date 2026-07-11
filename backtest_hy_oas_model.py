"""
120일 전 ~ 60일 전 데이터로 회귀모델(HYG+IEF -> HY OAS 변화) 학습,
이후 60일(-60일 ~ 오늘) 구간에 적용해서 정확도 검증
"""
import datetime, os
import numpy as np
import pandas as pd
import yfinance as yf
import urllib.request, json

FRED_KEY = os.environ['FRED_KEY']

TODAY = datetime.date.today()
TRAIN_START = TODAY - datetime.timedelta(days=180)   # -120일 지점 기준 넉넉히
TRAIN_END   = TODAY - datetime.timedelta(days=60)     # 학습 구간 끝
TEST_END    = TODAY                                    # 검증 구간 끝

# ── 1) HY OAS 실제값 (FRED) ──────────────────────
def fetch_oas(start, end):
    url = (f"https://api.stlouisfed.org/fred/series/observations?"
           f"series_id=BAMLH0A0HYM2&api_key={FRED_KEY}"
           f"&observation_start={start}&observation_end={end}&file_type=json")
    with urllib.request.urlopen(url, timeout=15) as r:
        d = json.load(r)
    obs = [(o['date'], float(o['value'])) for o in d['observations'] if o['value'] != '.']
    return pd.Series({pd.Timestamp(d): v for d, v in obs}).sort_index()

oas = fetch_oas(TRAIN_START - datetime.timedelta(days=5), TEST_END)

# ── 2) HYG, IEF 가격 (yfinance) ───────────────────
hyg = yf.Ticker("HYG").history(start=TRAIN_START, end=TEST_END + datetime.timedelta(days=1))["Close"]
ief = yf.Ticker("IEF").history(start=TRAIN_START, end=TEST_END + datetime.timedelta(days=1))["Close"]
hyg.index = hyg.index.tz_localize(None)
ief.index = ief.index.tz_localize(None)

hyg_ret = hyg.pct_change() * 100
ief_ret = ief.pct_change() * 100
oas_diff = oas.diff() * 100   # %p -> bp

# ── 3) 날짜 정합 (거래일 교집합만) ─────────────────
df = pd.DataFrame({"hyg_ret": hyg_ret, "ief_ret": ief_ret, "oas_diff": oas_diff}).dropna()

train = df[(df.index >= pd.Timestamp(TRAIN_START)) & (df.index <= pd.Timestamp(TRAIN_END))]
test  = df[(df.index >  pd.Timestamp(TRAIN_END))  & (df.index <= pd.Timestamp(TEST_END))]

print(f"학습 구간: {train.index.min().date()} ~ {train.index.max().date()} ({len(train)}일)")
print(f"검증 구간: {test.index.min().date()} ~ {test.index.max().date()} ({len(test)}일)")

# ── 4) 다중회귀: oas_diff ~ β1*hyg_ret + β2*ief_ret ──
X = np.column_stack([train["hyg_ret"], train["ief_ret"], np.ones(len(train))])
y = train["oas_diff"].values
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
beta_hyg, beta_ief, intercept = coef
print(f"베타(HYG)={beta_hyg:.3f}, 베타(IEF)={beta_ief:.3f}, 절편={intercept:.3f}")

# ── 5) 검증 구간에 적용 ────────────────────────────
test = test.copy()
test["pred_diff"] = beta_hyg*test["hyg_ret"] + beta_ief*test["ief_ret"] + intercept
test["actual_diff"] = test["oas_diff"]

mae  = np.mean(np.abs(test["pred_diff"] - test["actual_diff"]))
rmse = np.sqrt(np.mean((test["pred_diff"] - test["actual_diff"])**2))
hit_rate = np.mean(np.sign(test["pred_diff"]) == np.sign(test["actual_diff"]))
corr = np.corrcoef(test["pred_diff"], test["actual_diff"])[0,1]

print(f"MAE(bp)={mae:.2f}, RMSE(bp)={rmse:.2f}, 방향적중률={hit_rate*100:.1f}%, 상관계수={corr:.2f}")

test[["hyg_ret","ief_ret","actual_diff","pred_diff"]].to_csv("backtest_result.csv")
print("backtest_result.csv 저장 완료")
