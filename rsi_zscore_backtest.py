"""
rsi_zscore_backtest.py
사용법: python rsi_zscore_backtest.py [FORWARD_DAYS] [large|small]
기본값: FORWARD_DAYS=10, large
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
    "W", "CHWY", "PENN", "ABNB", "DOCU", "ZM", "BROS", "CAVA", "SG", "APP",
    "SMR", "OKLO", "VKTX", "RXRX", "ARQQ", "DAVE", "NU", "CELH", "ELF", "FIGS",
]

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
GROUP = sys.argv[2] if len(sys.argv) > 2 else "large"
TICKERS = LARGECAP_TICKERS if GROUP == "large" else SMALLCAP_TICKERS
LOOKBACK_DAYS = 250
MIN_HISTORY = 260

ZSCORE_BANDS = [(-999, -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0), (0, 999)]
ZSCORE_BANDS_FINE = [(-999, -2.3), (-2.3, -2.1), (-2.1, -1.9), (-1.9, -1.7), (-1.7, -1.5),
                      (-1.5, -1.3), (-1.3, -1.1), (-1.1, -0.9), (-0.9, -0.7), (-0.7, -0.5),
                      (-0.5, -0.3), (-0.3, 0)]


def find_signals(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < MIN_HISTORY + 20:
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

        results.append({"zscore": zscore, "fwd_return": fwd_return})
    return results


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_signals = []
    for t in TICKERS:
        try:
            recs = find_signals(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if recs:
            all_signals.extend(recs)
        print(f"{t}: {len(recs) if recs else 0}건")

    label = "대형주" if GROUP == "large" else "중소형주"
    print(f"\n=== {label} 다이버전스 신호 {len(all_signals)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_signals)
    print(f"기준선: 승률 {baseline_wr:.1f}%\n")

    zscores = np.array([s["zscore"] for s in all_signals])
    print(f"Z-score 분포: 평균 {zscores.mean():.2f} | 중앙값 {np.median(zscores):.2f} | 최소 {zscores.min():.2f} | 최대 {zscores.max():.2f}\n")

    print("[RSI Z-score 구간별 승률]")
    for lo, hi in ZSCORE_BANDS:
        sub = [s for s in all_signals if lo <= s["zscore"] < hi]
        wr, avg = win_rate(sub)
        if wr is None:
            continue
        gap = wr - baseline_wr
        lo_label = str(lo) if lo > -999 else "~"
        hi_label = str(hi) if hi < 999 else "이상"
        print(f"  {lo_label}~{hi_label}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {gap:+.1f}%p")

    print("\n[RSI Z-score 세밀 구간(0.2단위) 승률]")
    for lo, hi in ZSCORE_BANDS_FINE:
        sub = [s for s in all_signals if lo <= s["zscore"] < hi]
        wr, avg = win_rate(sub)
        if wr is None:
            continue
        gap = wr - baseline_wr
        lo_label = str(lo) if lo > -999 else "~"
        print(f"  {lo_label}~{hi}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {gap:+.1f}%p")

    print("\n해석: Z-score가 더 낮을수록(평균보다 표준편차 몇 개 더 아래) 승률이 오르면")
    print("'평균 대비 이례적으로 눌린 정도'가 진짜 신뢰도 지표라는 뜻.")


if __name__ == "__main__":
    main()
