"""
pre_rebound_threshold_backtest.py
사용법: python pre_rebound_threshold_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 중소형주에서 다이버전스 발생 후 5~7일 지난 시점의 신호들을 모아서,
      "신저점(low2) 대비 오늘까지 이미 몇 % 반등했는지"를 잘게 나눈 구간별로
      이후 FORWARD_DAYS 뒤 승률을 확인한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

SMALLCAP_TICKERS = [
    "RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
    "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ",
    "FUBO", "PATH", "ASAN", "BBAI", "SOUN", "RUN", "ENVX", "JOBY", "ACHR", "LAZR",
    "QS", "FSLY", "PLUG", "NIO", "XPEV", "BYND", "WOLF", "GPRO", "DNA", "HOOD",
    "AI", "PTON", "SNAP", "LYFT", "DASH", "PINS", "ETSY", "BILL", "TTD", "ROKU",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = 500
MIN_HISTORY = 260

REBOUND_BANDS = [(-999, 0), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 20), (20, 999)]


def find_signals(ticker):
    hist = yf.Ticker(ticker).history(period="3y")
    if hist.empty:
        return None
    c, l = hist["Close"], hist["Low"]
    rsi = RSIIndicator(c, window=14).rsi()
    macd_hist = MACD(c).macd_diff()

    n = len(hist)
    start = max(MIN_HISTORY, n - LOOKBACK_DAYS - FORWARD_DAYS)
    end = n - FORWARD_DAYS

    results = []
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
        days_since_low2 = i - low2_pos

        if not (5 <= days_since_low2 < 7):
            continue

        low2_price = float(c[low2_idx])
        pre_rebound = (float(c.iloc[i]) - low2_price) / low2_price * 100

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({"pre_rebound": pre_rebound, "fwd_return": fwd_return})
    return results


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_signals = []
    for t in SMALLCAP_TICKERS:
        recs = find_signals(t)
        if recs:
            all_signals.extend(recs)

    print(f"=== 중소형주 5~7일 구간 신호 {len(all_signals)}건, {FORWARD_DAYS}영업일 뒤 기준 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_signals)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    print("[신저점 대비 선행 반등폭(%) 구간별 승률]")
    print(f"{'구간':>12} | {'표본':>6} | {'승률':>7} | {'평균수익률':>10} | {'기준선대비':>10}")
    print("-" * 65)
    for lo, hi in REBOUND_BANDS:
        sub = [s for s in all_signals if lo <= s["pre_rebound"] < hi]
        wr, avg = win_rate(sub)
        if wr is None:
            continue
        gap = wr - baseline_wr
        lo_label = str(lo) if lo > -999 else "~"
        hi_label = str(hi) if hi < 999 else "이상"
        print(f"{lo_label:>4}~{hi_label:<6}% | {len(sub):>6} | {wr:>6.1f}% | {avg:>+9.2f}% | {gap:>+9.1f}%p")

    print("\n해석: 승률이 뚜렷하게 오르기 시작하는 반등폭 구간이")
    print("'매수 진입에 필요한 최소 선행 반등폭' 기준이 됨.")


if __name__ == "__main__":
    main()
