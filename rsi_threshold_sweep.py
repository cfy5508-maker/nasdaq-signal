"""
rsi_threshold_sweep.py
사용법: python rsi_threshold_sweep.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 눌림목 셋업(20일선/60일선 ±3% 근접) 조건은 고정하고,
      RSI 임계값을 35~65까지 5단위로 스윕해서 각각 승률을 비교.
      "RSI >= N" 최적값을 찾는다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MA_TOLERANCE_PCT = 0.03
RSI_THRESHOLDS = [35, 40, 45, 50, 55, 60, 65]


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 300:
        return None
    c = hist["Close"]
    close_values = c.values
    sma20 = c.rolling(20).mean()
    sma60 = c.rolling(60).mean()
    rsi = RSIIndicator(c, window=14).rsi()
    n = len(hist)

    rows = []
    for i in range(280, n - FORWARD_DAYS):
        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        sma60_now = float(sma60.iloc[i]) if not pd.isna(sma60.iloc[i]) else None
        rsi_now = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else None
        price_now = close_values[i]
        if sma20_now is None or sma60_now is None or rsi_now is None:
            continue

        near = (abs(price_now - sma20_now) / sma20_now <= MA_TOLERANCE_PCT) or \
               (abs(price_now - sma60_now) / sma60_now <= MA_TOLERANCE_PCT)
        if not near:
            continue

        entry = price_now
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1
        rows.append({"rsi": rsi_now, "fwd_return": fwd_return, "win": fwd_return > 0})
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_rows = []
    for t in LARGECAP_TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)

    print(f"\n=== 20/60일선 근접(±3%) 신호 총 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"기준선(RSI 무관): 승률 {baseline_wr:.1f}% | 평균수익률 {baseline_avg:+.2f}%\n")

    print("=== RSI 임계값 스윕 (>= N 이상 vs 미만) ===")
    for th in RSI_THRESHOLDS:
        above = [r for r in all_rows if r["rsi"] >= th]
        below = [r for r in all_rows if r["rsi"] < th]
        wr_a, avg_a = win_rate(above)
        wr_b, avg_b = win_rate(below)
        print(f"\n[RSI >= {th}]")
        if wr_a is not None:
            print(f"  이상: 표본 {len(above)}({len(above)/len(all_rows)*100:.1f}%) | 승률 {wr_a:.1f}% | 기준대비 {wr_a-baseline_wr:+.1f}%p | 평균수익률 {avg_a:+.2f}%")
        if wr_b is not None:
            print(f"  미만: 표본 {len(below)}({len(below)/len(all_rows)*100:.1f}%) | 승률 {wr_b:.1f}% | 기준대비 {wr_b-baseline_wr:+.1f}%p | 평균수익률 {avg_b:+.2f}%")

    print("\n=== RSI 10단위 구간별 승률 (세밀 분석) ===")
    for lo, hi in [(0,30),(30,40),(40,45),(45,50),(50,55),(55,60),(60,70),(70,100)]:
        sub = [r for r in all_rows if lo <= r["rsi"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  RSI {lo}~{hi}: 표본 {len(sub)}({len(sub)/len(all_rows)*100:.1f}%) | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")


if __name__ == "__main__":
    main()
