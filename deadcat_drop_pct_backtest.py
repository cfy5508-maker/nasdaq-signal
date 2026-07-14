"""
deadcat_drop_pct_backtest.py
사용법: python deadcat_drop_pct_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 정석 다이버전스(pass)가 확정된 시점의 "6개월 고점 대비 하락폭(%)"을
      4구간(-15%이내/-15~25%/-25~40%/-40%이상)으로 나눠서, 그 구간별
      10일 뒤 승률과 새고점(직전20일 저항선) 돌파율을 비교한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
]
SMALLCAP_TICKERS = [
    "RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
    "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ",
    "FUBO", "PATH", "ASAN", "BBAI", "SOUN", "RUN", "ENVX", "JOBY", "ACHR", "LAZR",
]
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MIN_GAP_DAYS = 5
MAX_GAP_DAYS = 30
HIGH_LOOKBACK = 126
RECENT_HIGH_LOOKBACK = 20


def find_realtime_lows(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n):
        past_window = close_values[i - order:i]
        if close_values[i] < past_window.min():
            positions.append(i)
    return positions


def calc_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 200:
        return None
    c = hist["Close"]
    close_values = c.values
    rsi = calc_rsi(c)
    n = len(hist)

    realtime_lows = find_realtime_lows(close_values, order=5)

    is_pass_divergence = {}
    for pos2 in realtime_lows:
        candidates = [p for p in realtime_lows if p != pos2 and MIN_GAP_DAYS <= (pos2 - p) <= MAX_GAP_DAYS]
        if not candidates:
            continue
        pos1 = min(candidates, key=lambda p: close_values[p])
        price1, price2 = close_values[pos1], close_values[pos2]
        rsi1, rsi2 = rsi.iloc[pos1], rsi.iloc[pos2]
        if pd.isna(rsi1) or pd.isna(rsi2):
            continue
        if price2 < price1 and rsi2 > rsi1:
            is_pass_divergence[pos2] = True

    rolling_high = c.rolling(HIGH_LOOKBACK, min_periods=20).max()

    rows = []
    for i in range(HIGH_LOOKBACK, n - FORWARD_DAYS):
        if i in is_pass_divergence:
            high_val = rolling_high.iloc[i]
            if pd.isna(high_val) or high_val <= 0:
                continue
            price2 = close_values[i]
            drop_pct = (high_val - price2) / high_val * 100

            lookback_start = max(0, i - RECENT_HIGH_LOOKBACK)
            resistance = float(close_values[lookback_start:i].max()) if i > lookback_start else price2
            future_window = close_values[i+1:i+1+FORWARD_DAYS]
            new_high_achieved = bool((future_window > resistance).any())

            entry = price2
            fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

            rows.append({"drop_pct": drop_pct, "new_high_achieved": new_high_achieved,
                        "fwd_return": fwd_return, "win": fwd_return > 0})
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def new_high_rate(recs):
    if not recs:
        return None
    return float(np.mean([r["new_high_achieved"] for r in recs]) * 100)


def main():
    all_rows = []
    for t in TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)

    print(f"\n=== 정석 다이버전스(pass) 총 {len(all_rows)}건, {FORWARD_DAYS}일 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_rows)
    baseline_nh = new_high_rate(all_rows)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}% | 새고점달성률 {baseline_nh:.1f}%\n")

    print("=== 6개월 고점 대비 하락폭(%) 구간별 ===")
    bands = [(0, 15, "-15%이내"), (15, 25, "-15~-25%"), (25, 40, "-25~-40%"), (40, 999, "-40%이상")]
    for lo, hi, label in bands:
        sub = [r for r in all_rows if lo <= r["drop_pct"] < hi]
        wr, avg = win_rate(sub)
        nh = new_high_rate(sub)
        if wr is not None:
            print(f"  {label}: 표본{len(sub)}건 | 승률 {wr:.1f}%(기준대비{wr-baseline_wr:+.1f}%p) | 새고점달성률 {nh:.1f}%(기준대비{nh-baseline_nh:+.1f}%p) | 평균수익률 {avg:+.2f}%")


if __name__ == "__main__":
    main()
