"""
rsi_zone_and_wait_backtest.py
사용법: python rsi_zone_and_wait_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 중소형주(70종목으로 확대)에서 다이버전스 발생 시점 RSI 구간
      (절대값 20~25, 백분위 20~30%)과 "대기일수(0~3/3~5/5~7/7~10/10+일)"
      조건을 결합해서 승률을 검증. RSI구간별로 즉시매매 vs 대기매매 비교.
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
    "W", "CHWY", "PENN", "ABNB", "DOCU", "ZM", "BROS", "CAVA", "SG", "APP",
    "SMR", "OKLO", "VKTX", "RXRX", "ARQQ", "DAVE", "NU", "CELH", "ELF", "FIGS",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = 250
MIN_HISTORY = 260


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

        days_since_low2 = i - low2_pos

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "ticker": ticker,
            "low2_rsi_abs": low2_rsi_abs,
            "low2_rsi_pctile": low2_rsi_pctile,
            "days_since_low2": days_since_low2,
            "fwd_return": fwd_return,
        })
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
        if recs is None:
            print(f"{t}: 데이터 없음(티커 확인 필요)")
            continue
        all_signals.extend(recs)

    print(f"=== 중소형주 전체 다이버전스 신호 {len(all_signals)}건 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_signals)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    rsi_zones = [
        ("RSI절대값 20~25", lambda s: 20 <= s["low2_rsi_abs"] < 25),
        ("RSI백분위 20~30%", lambda s: 20 <= s["low2_rsi_pctile"] < 30),
    ]

    for zone_label, zone_filter in rsi_zones:
        zone_signals = [s for s in all_signals if zone_filter(s)]
        zone_wr, zone_avg = win_rate(zone_signals)
        print(f"=== {zone_label} 전체 (표본 {len(zone_signals)}건) ===")
        if zone_wr is not None:
            gap = zone_wr - baseline_wr
            print(f"  즉시(대기무관): 승률 {zone_wr:.1f}% | 평균수익률 {zone_avg:+.2f}% | 기준선 대비 {gap:+.1f}%p")

        for lo, hi, label in [(0, 3, "0~3일(즉시권)"), (3, 5, "3~5일"), (5, 7, "5~7일"), (7, 10, "7~10일"), (10, 999, "10일이상")]:
            sub = [s for s in zone_signals if lo <= s["days_since_low2"] < hi]
            wr, avg = win_rate(sub)
            if wr is None:
                print(f"  {label}: 표본 없음")
                continue
            gap = wr - baseline_wr
            print(f"  {label}: 표본 {len(sub)}건 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}% | 기준선 대비 {gap:+.1f}%p")
        print()

    print("해석: RSI구간 단독보다 '5~7일'과 결합했을 때 승률이 더 오르면")
    print("그 조합이 중소형주 최종 매매 조건의 핵심이 됨.")


if __name__ == "__main__":
    main()
