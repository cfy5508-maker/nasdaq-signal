"""
rsi_trajectory_backtest.py
사용법: python rsi_trajectory_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 다이버전스를 만든 저점(low2)의 RSI 52주 백분위를 구간별로 나누되,
      "5일 전 백분위 대비 지금이 더 높은지(회복중) 낮은지(하락중)"를
      추가로 구분해서, 같은 구간이라도 궤적에 따라 승률이 다른지 확인한다.
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

PCTILE_BANDS = [(0, 15), (15, 20), (20, 30), (30, 999)]


def rolling_pctile_at(rsi, pos, window=252):
    start = max(0, pos - window)
    w = rsi.iloc[start:pos+1].dropna()
    if len(w) == 0:
        return None
    val = rsi.iloc[pos]
    return float((w < val).mean() * 100)


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

        low2_pos = hist.index.get_loc(low2_idx)
        if low2_pos < 5:
            continue

        pctile_now = rolling_pctile_at(rsi, low2_pos)
        pctile_5ago = rolling_pctile_at(rsi, low2_pos - 5)
        if pctile_now is None or pctile_5ago is None:
            continue

        trending_up = pctile_now > pctile_5ago
        pctile_change = pctile_now - pctile_5ago

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "pctile_now": pctile_now,
            "trending_up": trending_up,
            "pctile_change": pctile_change,
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
        if recs:
            all_signals.extend(recs)

    print(f"=== 중소형주 전체 다이버전스 신호 {len(all_signals)}건 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_signals)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    for lo, hi in PCTILE_BANDS:
        band_signals = [s for s in all_signals if lo <= s["pctile_now"] < hi]
        band_wr, band_avg = win_rate(band_signals)
        hi_label = str(hi) if hi < 999 else "이상"
        print(f"=== 백분위 {lo}~{hi_label}% 구간 (표본 {len(band_signals)}건) ===")
        if band_wr is not None:
            gap = band_wr - baseline_wr
            print(f"  전체: 승률 {band_wr:.1f}% | 기준선 대비 {gap:+.1f}%p")

        up = [s for s in band_signals if s["trending_up"]]
        down = [s for s in band_signals if not s["trending_up"]]
        for label, sub in [("5일전보다 상승중(회복중)", up), ("5일전보다 하락중(진행중)", down)]:
            wr, avg = win_rate(sub)
            if wr is None:
                print(f"    {label}: 표본 없음")
                continue
            gap = wr - baseline_wr
            print(f"    {label}: 표본 {len(sub)}건 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}% | 기준선 대비 {gap:+.1f}%p")
        print()

    print("해석: 같은 백분위 구간이라도 '상승중'과 '하락중'의 승률이 크게 다르면")
    print("RSI 궤적(방향)이 백분위 수준 자체보다 더 중요한 신호일 수 있음.")


if __name__ == "__main__":
    main()
