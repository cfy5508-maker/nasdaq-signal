"""
divergence_strength_backtest.py
사용법: python divergence_strength_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 다이버전스의 "강도"를 두 가지로 측정해서 승률과의 관계를 확인.
      1) rsi_slope: 두 번째 저점 RSI - 첫 번째 저점 RSI (클수록 뚜렷한 다이버전스)
      2) price_drop_ratio: 두 번째 저점이 첫 번째 저점 대비 몇% 더 낮았는지
      각각 구간별로 나눠서 승률이 강도에 비례하는지 확인.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

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
    "QS", "FSLY", "PLUG", "NIO", "XPEV", "BYND", "WOLF", "GPRO", "DNA", "HOOD",
    "AI", "PTON", "SNAP", "LYFT", "DASH", "PINS", "ETSY", "BILL", "TTD", "ROKU",
]
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS
LARGECAP_SET = set(LARGECAP_TICKERS)

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = 250
MIN_HISTORY = 260

RSI_SLOPE_BANDS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 999)]
PRICE_DROP_BANDS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 10), (10, 999)]


def find_signals(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    c = hist["Close"]
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

        rsi_slope = float(rsi[low2_idx] - rsi[low1_idx])
        price_drop_ratio = float((c[low1_idx] - c[low2_idx]) / c[low1_idx] * 100)

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "ticker": ticker,
            "is_large": ticker in LARGECAP_SET,
            "rsi_slope": rsi_slope,
            "price_drop_ratio": price_drop_ratio,
            "fwd_return": fwd_return,
        })
    return results


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def analyze_group(signals, label):
    baseline_wr, baseline_avg = win_rate(signals)
    if baseline_wr is None:
        print(f"=== {label}: 표본 없음 ===\n")
        return
    print(f"=== {label} ({len(signals)}건, {FORWARD_DAYS}영업일 뒤 기준) ===")
    print(f"기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    rsi_slopes = np.array([s["rsi_slope"] for s in signals])
    price_drops = np.array([s["price_drop_ratio"] for s in signals])
    returns = np.array([s["fwd_return"] for s in signals])
    corr_rsi = np.corrcoef(rsi_slopes, returns)[0, 1] if len(signals) > 1 else float("nan")
    corr_price = np.corrcoef(price_drops, returns)[0, 1] if len(signals) > 1 else float("nan")
    print(f"상관계수: RSI기울기-수익률 = {corr_rsi:+.3f} | 가격하락폭-수익률 = {corr_price:+.3f}\n")

    print("[RSI 기울기(두저점 RSI 차이) 구간별]")
    print(f"{'구간':>10} | {'표본':>6} | {'승률':>7} | {'평균수익률':>10} | {'기준선대비':>10}")
    for lo, hi in RSI_SLOPE_BANDS:
        sub = [s for s in signals if lo <= s["rsi_slope"] < hi]
        wr, avg = win_rate(sub)
        if wr is None:
            continue
        gap = wr - baseline_wr
        hi_label = str(hi) if hi < 999 else "이상"
        print(f"{lo:>4}~{hi_label:<4} | {len(sub):>6} | {wr:>6.1f}% | {avg:>+9.2f}% | {gap:>+9.1f}%p")
    print()

    print("[가격 하락폭(%, 두저점간) 구간별]")
    print(f"{'구간':>10} | {'표본':>6} | {'승률':>7} | {'평균수익률':>10} | {'기준선대비':>10}")
    for lo, hi in PRICE_DROP_BANDS:
        sub = [s for s in signals if lo <= s["price_drop_ratio"] < hi]
        wr, avg = win_rate(sub)
        if wr is None:
            continue
        gap = wr - baseline_wr
        hi_label = str(hi) if hi < 999 else "이상"
        print(f"{lo:>4}~{hi_label:<4}% | {len(sub):>6} | {wr:>6.1f}% | {avg:>+9.2f}% | {gap:>+9.1f}%p")
    print()


def main():
    all_signals = []
    for t in TICKERS:
        recs = find_signals(t)
        if recs is None:
            print(f"{t}: 데이터 없음")
            continue
        all_signals.extend(recs)
        print(f"{t}: 다이버전스 신호 {len(recs)}건")

    if not all_signals:
        print("신호 없음")
        return

    large = [s for s in all_signals if s["is_large"]]
    small = [s for s in all_signals if not s["is_large"]]

    print()
    analyze_group(all_signals, "전체(대형주+중소형주 통합)")
    analyze_group(large, "대형주만")
    analyze_group(small, "중소형주만")

    print("해석: 상관계수가 양수이고 구간별 승률이 단조증가하면")
    print("'강도가 클수록 신뢰도 높은 진짜 다이버전스'라는 가설이 지지됨.")


if __name__ == "__main__":
    main()
