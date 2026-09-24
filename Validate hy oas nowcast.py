"""
HY OAS 나우캐스트 검증 스크립트 (읽기 전용 - 운영 데이터/파일은 건드리지 않음)

목적: FRED에 OAS가 아직 발표되지 않은 1~2일 구간을 어떻게 채우는 게 가장 정확한지 실제 데이터로 비교.

비교 방식
  carry    : 직전 확정값을 그대로 사용 (지금 차트가 실제로 하는 방식)
  fred_vix : 버그 수정한 회귀 추정, VIX는 FRED(VIXCLS)
  yf_vix   : 버그 수정한 회귀 추정, VIX는 yfinance(^VIX)

방법
  과거 각 날짜 t마다 "최근 k일(1, 2) OAS를 모른다"고 가정하고,
  그 시점까지의 데이터로만 90일 회귀를 학습해 t의 OAS를 추정한 뒤, 실제 발표값과 비교한다.

주의
  fred_vix는 과거 시점에 FRED VIX가 이미 올라와 있었다고 가정한다.
  실제로 그 시점에 올라와 있는지는 맨 아래 [데이터 발표 지연 확인]으로 따로 본다.

실행: FRED_KEY=... python validate_hy_oas_nowcast.py
출력: 콘솔 요약 + validation_nowcast_rows.csv (날짜별 상세)
"""
import datetime, os, json, urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

FRED_KEY = os.environ["FRED_KEY"]
TODAY = datetime.date.today()
START = TODAY - datetime.timedelta(days=800)
TRAIN_WINDOW = 90          # 운영 스크립트와 동일
LAGS = [1, 2]              # OAS 미발표 영업일 수
THRESHOLDS = [3.0, 3.5, 4.5]  # 차트 신호 기준선
RECENT_DAYS = 120          # 최근 구간 별도 집계


def fetch_fred(series_id):
    url = (f"https://api.stlouisfed.org/fred/series/observations?"
           f"series_id={series_id}&api_key={FRED_KEY}"
           f"&observation_start={START}&observation_end={TODAY}&file_type=json")
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    obs = [(o["date"], float(o["value"])) for o in d["observations"] if o["value"] != "."]
    return pd.Series({pd.Timestamp(dd): v for dd, v in obs}).sort_index()


def fetch_yf(ticker):
    s = yf.Ticker(ticker).history(start=START, end=TODAY + datetime.timedelta(days=1))["Close"]
    s.index = s.index.tz_localize(None).normalize()
    return s


oas = fetch_fred("BAMLH0A0HYM2")
vix_fred = fetch_fred("VIXCLS")
vix_yf = fetch_yf("^VIX")
hyg = fetch_yf("HYG")
ief = fetch_yf("IEF")

oas_diff = oas.diff() * 100  # bp


def build(vix):
    feat = pd.DataFrame({
        "hyg_ret": hyg.pct_change() * 100,
        "ief_ret": ief.pct_change() * 100,
        "vix_diff": vix.diff(),
    }).dropna()
    train_df = feat.join(oas_diff.rename("oas_diff"), how="inner").dropna()
    return feat, train_df


METHODS = {"fred_vix": build(vix_fred), "yf_vix": build(vix_yf)}


def estimate(feat, train_df, known_date, target_date, known_oas):
    train = train_df[train_df.index <= known_date].tail(TRAIN_WINDOW)
    if len(train) < TRAIN_WINDOW:
        return None, 0
    X = np.column_stack([train["hyg_ret"], train["ief_ret"], train["vix_diff"], np.ones(len(train))])
    b = np.linalg.lstsq(X, train["oas_diff"].values, rcond=None)[0]
    after = feat[(feat.index > known_date) & (feat.index <= target_date)]
    pred = (b[0] * after["hyg_ret"] + b[1] * after["ief_ret"] + b[2] * after["vix_diff"] + b[3]).sum()
    return known_oas + pred / 100, len(after)


rows = []
dates = list(oas.index)
for i in range(len(dates)):
    t = dates[i]
    for k in LAGS:
        if i - k < 0:
            continue
        known = dates[i - k]
        row = {"date": t.strftime("%Y-%m-%d"), "lag": k, "actual": oas[t], "carry": oas[known]}
        ok = True
        for name, (feat, train_df) in METHODS.items():
            est, n_after = estimate(feat, train_df, known, t, oas[known])
            if est is None:
                ok = False
                break
            row[name] = est
            row[f"{name}_days"] = n_after
        if ok:
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("validation_nowcast_rows.csv", index=False)

cutoff = (pd.Timestamp(TODAY) - pd.Timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")


def summarize(sub, label):
    print(f"\n=== {label} ===")
    for k in LAGS:
        s = sub[sub["lag"] == k]
        if s.empty:
            continue
        print(f"[미발표 {k}일] 표본 {len(s)}건")
        print(f"  {'방식':<9} {'MAE(bp)':>8} {'RMSE(bp)':>9} {'최대(bp)':>9} {'편향(bp)':>9}  " +
              "  ".join(f"{th}%선 불일치" for th in THRESHOLDS))
        for m in ["carry", "fred_vix", "yf_vix"]:
            err = (s[m] - s["actual"]) * 100
            mism = [int(((s[m] < th) != (s["actual"] < th)).sum()) for th in THRESHOLDS]
            print(f"  {m:<9} {err.abs().mean():>8.2f} {np.sqrt((err ** 2).mean()):>9.2f} "
                  f"{err.abs().max():>9.2f} {err.mean():>9.2f}  " + "  ".join(f"{x:>10}" for x in mism))


print(f"실행 시각(UTC): {datetime.datetime.utcnow():%Y-%m-%d %H:%M}")
print(f"검증 기간: {df['date'].min()} ~ {df['date'].max()}")
summarize(df, "전체 기간")
summarize(df[df["date"] >= cutoff], f"최근 {RECENT_DAYS}일")

print("\n=== 데이터 발표 지연 확인 (지금 이 순간 각 소스의 최신 날짜) ===")
for name, s in [("OAS (FRED)", oas), ("VIX (FRED)", vix_fred), ("VIX (yfinance)", vix_yf),
                ("HYG (yfinance)", hyg), ("IEF (yfinance)", ief)]:
    print(f"  {name:<16} {s.index[-1]:%Y-%m-%d}")
print("\n상세 행 데이터: validation_nowcast_rows.csv")
