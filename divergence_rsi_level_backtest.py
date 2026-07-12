"""
divergence_rsi_level_backtest.py
사용법: python divergence_rsi_level_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 다이버전스를 만든 두 번째 저점(low2)의 RSI 수준을
      1) 절대값(0~100)
      2) 52주 상대 백분위
      두 방식으로 각각 구간을 나눠서 승률을 비교한다.
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

ABS_BANDS = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 999)]
PCTILE_BANDS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 999)]


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

        low2_rsi_abs = float(rsi[low2_idx])

        low2_pos = hist.index.get_loc(low2_idx)
        rsi_window = rsi.iloc[max(0, low2_pos-252):low2_pos+1].dropna()
        low2_rsi_pctile = float((rsi_window < low2_rsi_abs).mean() * 100) if len(rsi_window) > 0 else 50.0

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "ticker": ticker,
            "is_large": ticker in LARGECAP_SET,
            "low2_rsi_abs": low2_rsi_abs,
            "low2_rsi_pctile": low2_rsi_pctile,
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

    print("[다이버전스 저점의 RSI 절대값 구간별]")
    print(f"{'구간':>10} | {'표본':>6} | {'승률':>7} | {'평균수익률':>10} | {'기준선대비':>10}")
    for lo, hi in ABS_BANDS:
        sub = [s for s in signals if lo <= s["low2_rsi_abs"] < hi]
        wr, avg = win_rate(sub)
        if wr is None:
            continue
        gap = wr - baseline_wr
        hi_label = str(hi) if hi < 999 else "이상"
        print(f"{lo:>4}~{hi_label:<4} | {len(sub):>6} | {wr:>6.1f}% | {avg:>+9.2f}% | {gap:>+9.1f}%p")
    print()

    print("[다이버전스 저점의 RSI 52주 백분위 구간별]")
    print(f"{'구간':>10} | {'표본':>6} | {'승률':>7} | {'평균수익률':>10} | {'기준선대비':>10}")
    for lo, hi in PCTILE_BANDS:
        sub = [s for s in signals if lo <= s["low2_rsi_pctile"] < hi]
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

    print("해석: 절대값 구간별 승률과 상대값(백분위) 구간별 승률의 패턴을 비교해서")
    print("어느 기준이 더 안정적인 판정 근거인지 확인.")


if __name__ == "__main__":
    main()
