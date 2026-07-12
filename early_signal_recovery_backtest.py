"""
early_signal_recovery_backtest.py
사용법: python early_signal_recovery_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: RSI 백분위 15~20% 구간에서 다이버전스가 발생한 시점을 기록해두고,
      그 이후(10일 이내) RSI 백분위가 20~30%로 회복되는 "그 순간"을
      매수 시점으로 잡는다. 이때 다이버전스 재확인은 하지 않는다.
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
RECOVERY_WINDOW = 10


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

    early_signal_positions = []
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

        pctile_at_i = pctile_series.iloc[i] if not pd.isna(pctile_series.iloc[i]) else None
        if pctile_at_i is not None and 15 <= pctile_at_i < 20:
            early_signal_positions.append(i)

    results = []
    for sig_pos in early_signal_positions:
        for j in range(sig_pos + 1, min(sig_pos + 1 + RECOVERY_WINDOW, end)):
            pctile_j = pctile_series.iloc[j] if not pd.isna(pctile_series.iloc[j]) else None
            if pctile_j is None:
                continue
            if 20 <= pctile_j < 30:
                entry = float(c.iloc[j])
                if j + FORWARD_DAYS >= n:
                    break
                fwd_return = float(c.iloc[j + FORWARD_DAYS]) / entry - 1
                results.append({"fwd_return": fwd_return})
                break
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

    print(f"=== '15~20%에서 다이버전스 -> {RECOVERY_WINDOW}일내 20~30% 회복' 신호 {len(all_signals)}건 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    wr, avg = win_rate(all_signals)
    print(f"승률 {wr:.1f}% | 평균수익률 {avg:+.2f}%")
    print("\n(비교 참고: 기존 방식 '20~30%에서 다이버전스 재확인' 전체 승률은 51.8%,")
    print(" 그중 15~20%를 거친 구제형만 승률 53.9% 였음)")


if __name__ == "__main__":
    main()
