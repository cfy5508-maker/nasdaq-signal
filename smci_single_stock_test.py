"""
smci_single_stock_test.py
사용법: python smci_single_stock_test.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: SMCI 하나에 대해, 지난 250거래일 동안 발생한 모든 다이버전스를
      날짜순으로 나열하고, 각 시점의 RSI Z-score와 실제 10일 뒤
      승패를 낱낱이 보여준다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

TICKER = "SMCI"
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = 250
MIN_HISTORY = 260


def main():
    hist = yf.Ticker(TICKER).history(period="2y")
    c = hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    macd_hist = MACD(c).macd_diff()

    n = len(hist)
    start = max(MIN_HISTORY, n - LOOKBACK_DAYS - FORWARD_DAYS)
    end = n - FORWARD_DAYS

    print(f"=== {TICKER} 개별 종목 다이버전스 전수 확인 ({FORWARD_DAYS}일 뒤 기준) ===\n")

    rows = []
    for i in range(start, end):
        recent = c.iloc[max(0, i-90):i+1]
        if len(recent) < 90:
            continue

        low1_idx = recent.iloc[:45].idxmin()
        low2_idx = recent.iloc[45:].idxmin()
        bullish_divergence = bool(c[low2_idx] < c[low1_idx] and rsi[low2_idx] > rsi[low1_idx])

        mh = macd_hist.iloc[max(0, i-90):i+1]
        if len(mh) < 90:
            continue
        macd_rising = bool(mh.iloc[-1] > mh.iloc[:-1].min())

        if not (bullish_divergence and macd_rising):
            continue

        low2_pos = hist.index.get_loc(low2_idx)
        rsi_window = rsi.iloc[max(0, low2_pos-252):low2_pos+1].dropna()
        if len(rsi_window) < 60:
            continue
        rsi_mean = float(rsi_window.mean())
        rsi_std = float(rsi_window.std())
        if rsi_std == 0 or np.isnan(rsi_std):
            continue
        rsi_at_low2 = float(rsi[low2_idx])
        zscore = (rsi_at_low2 - rsi_mean) / rsi_std

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1
        date_str = str(hist.index[i].date())

        rows.append({
            "date": date_str, "zscore": zscore, "rsi_mean": rsi_mean, "rsi_std": rsi_std,
            "rsi_at_low2": rsi_at_low2, "fwd_return": fwd_return, "win": fwd_return > 0,
        })

    if not rows:
        print("다이버전스 신호 없음")
        return

    rows.sort(key=lambda r: r["zscore"])

    print(f"{'날짜':>12} | {'Z-score':>8} | {'RSI평균':>7} | {'RSI표준편차':>9} | {'그시점RSI':>8} | {'10일뒤수익':>10} | {'승패':>4}")
    print("-" * 80)
    for r in rows:
        result = "승리" if r["win"] else "패배"
        print(f"{r['date']:>12} | {r['zscore']:>8.2f} | {r['rsi_mean']:>7.1f} | {r['rsi_std']:>9.1f} | "
              f"{r['rsi_at_low2']:>8.1f} | {r['fwd_return']*100:>+9.2f}% | {result:>4}")

    print(f"\n총 다이버전스 신호: {len(rows)}건")
    wins = sum(1 for r in rows if r["win"])
    print(f"전체 승률: {wins}/{len(rows)} = {wins/len(rows)*100:.1f}%")

    n_third = len(rows) // 3
    if n_third > 0:
        lower = rows[:n_third]
        middle = rows[n_third:2*n_third]
        upper = rows[2*n_third:]
        for label, group in [("Z-score 낮은 1/3(가장 극단)", lower), ("중간 1/3", middle), ("Z-score 높은 1/3(덜 극단)", upper)]:
            w = sum(1 for r in group if r["win"])
            print(f"  {label}: {w}/{len(group)} = {w/len(group)*100:.1f}%")


if __name__ == "__main__":
    main()
