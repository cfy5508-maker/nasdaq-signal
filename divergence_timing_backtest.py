"""
divergence_timing_backtest.py
사용법: python divergence_timing_backtest.py

동작: 4단계(다이버전스) pass 신호가 뜬 날들을 고정하고, 그 이후
      1,2,3,5,7,10,15,20 영업일 각각의 승률/평균수익률을 계산해서
      "다이버전스 신호가 뜨면 보통 며칠 안에 상승 전환이 일어나는지" 확인.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

TICKERS = ["NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
           "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH"]
LOOKBACK_DAYS = 250
MIN_HISTORY = 260
HORIZONS = [1, 2, 3, 5, 7, 10, 15, 20]
MAX_HORIZON = max(HORIZONS)


def find_divergence_days(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    c = hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    macd = MACD(c)
    macd_hist = macd.macd_diff()

    n = len(hist)
    start = max(MIN_HISTORY, n - LOOKBACK_DAYS - MAX_HORIZON)
    end = n - MAX_HORIZON

    results = []
    for i in range(start, end):
        recent = c.iloc[max(0, i-90):i+1]
        recent_rsi = rsi.iloc[max(0, i-90):i+1]
        if len(recent) < 90:
            continue
        low1_idx = recent.iloc[:45].idxmin()
        low2_idx = recent.iloc[45:].idxmin()
        bullish_divergence = bool(c[low2_idx] < c[low1_idx] and rsi[low2_idx] > rsi[low1_idx])

        mh = macd_hist.iloc[max(0, i-90):i+1]
        if len(mh) < 90:
            continue
        macd_rising = bool(mh.iloc[-1] > mh.iloc[:-1].min())

        if bullish_divergence and macd_rising:
            entry_price = float(c.iloc[i])
            fwd_prices = {h: float(c.iloc[i+h]) for h in HORIZONS if i+h < n}
            if len(fwd_prices) == len(HORIZONS):
                results.append({"ticker": ticker, "entry": entry_price, "fwd": fwd_prices})
    return results


def main():
    all_signals = []
    for t in TICKERS:
        recs = find_divergence_days(t)
        if recs is None:
            print(f"{t}: 데이터 없음")
            continue
        all_signals.extend(recs)
        print(f"{t}: 다이버전스 신호 {len(recs)}건")

    if not all_signals:
        print("신호 없음")
        return

    print(f"\n=== 전체 {len(all_signals)}건 다이버전스 신호, 기간별 승률/평균수익률 ===\n")
    print(f"{'경과일':>6} | {'승률':>7} | {'평균수익률':>10} | {'중앙값':>8}")
    print("-" * 45)

    for h in HORIZONS:
        returns = np.array([(s["fwd"][h] / s["entry"] - 1) for s in all_signals])
        win_rate = (returns > 0).mean() * 100
        avg = returns.mean() * 100
        median = np.median(returns) * 100
        print(f"{h:>4}일 | {win_rate:>6.1f}% | {avg:>+9.2f}% | {median:>+7.2f}%")

    print("\n해석: 승률이 가장 먼저 안정적으로 높아지는 지점이 '다이버전스 신호의 유효 반응 기간'.")
    print("그 이후 승률이 계속 오르면 장기 보유가 유리, 정체/하락하면 그 지점이 최적 청산 시점.")


if __name__ == "__main__":
    main()
