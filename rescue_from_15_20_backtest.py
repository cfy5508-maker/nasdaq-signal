"""
rescue_from_15_20_backtest.py
사용법: python rescue_from_15_20_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 다이버전스가 RSI 백분위 20~30% 구간에서 뜬 신호들을,
      "최근 10일 안에 15~20% 구간을 거쳐서 올라왔는지(구제형)"와
      "15~20%를 거치지 않고 곧장 20~30%대에 머문 경우(순수형)"로 나눠서
      승률을 비교한다.
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


def rolling_pctile_series(rsi, window=252):
    result = pd.Series(index=rsi.index, dtype=float)
    for j in range(window, len(rsi)):
        w = rsi.iloc[j-window:j]
        result.iloc[j] = (w < rsi.iloc[j]).mean() * 100
    return result


def find_signals(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    c = hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    macd_hist = MACD(c).macd_diff()
    pctile_series = rolling_pctile_series(rsi)

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
        pctile_now = pctile_series.iloc[low2_pos] if not pd.isna(pctile_series.iloc[low2_pos]) else None
        if pctile_now is None:
            continue

        if not (20 <= pctile_now < 30):
            continue

        window_start = max(0, low2_pos - 10)
        recent_pctiles = pctile_series.iloc[window_start:low2_pos]
        passed_through_15_20 = bool(((recent_pctiles >= 15) & (recent_pctiles < 20)).any())

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "passed_through_15_20": passed_through_15_20,
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

    print(f"=== RSI백분위 20~30% 구간 다이버전스 신호 {len(all_signals)}건 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_signals)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    rescued = [s for s in all_signals if s["passed_through_15_20"]]
    pure = [s for s in all_signals if not s["passed_through_15_20"]]

    for label, sub in [("구제형(최근10일내 15~20% 거침)", rescued), ("순수형(15~20% 안 거침)", pure)]:
        wr, avg = win_rate(sub)
        if wr is None:
            print(f"{label}: 표본 없음")
            continue
        gap = wr - baseline_wr
        print(f"{label}: 표본 {len(sub)}건 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}% | 기준선 대비 {gap:+.1f}%p")

    print("\n해석: '구제형'의 승률이 '순수형'과 비슷하거나 더 높으면,")
    print("15~20%를 거쳐 20~30%로 회복하는 게 안전한 매수 시점이라는 가설이 지지됨.")


if __name__ == "__main__":
    main()
