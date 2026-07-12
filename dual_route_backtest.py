"""
dual_route_backtest.py
사용법: python dual_route_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10
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

    divergence_days = {}
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
        if bullish_divergence and macd_rising:
            divergence_days[i] = True

    results = []
    for i in range(start, end):
        if i not in divergence_days:
            continue

        pctile_now = pctile_series.iloc[i] if not pd.isna(pctile_series.iloc[i]) else None
        if pctile_now is None:
            continue
        if not (20 <= pctile_now < 30):
            continue

        pctile_5ago = pctile_series.iloc[i-5] if i >= 5 and not pd.isna(pctile_series.iloc[i-5]) else None
        if pctile_5ago is None:
            continue

        trending_down = pctile_now < pctile_5ago
        decline_speed = pctile_5ago - pctile_now
        route_B = bool(trending_down and decline_speed <= 15)

        window_start = max(0, i - RECOVERY_WINDOW)
        route_A = False
        for k in range(window_start, i):
            if k in divergence_days:
                pctile_k = pctile_series.iloc[k] if not pd.isna(pctile_series.iloc[k]) else None
                if pctile_k is not None and 15 <= pctile_k < 20:
                    route_A = True
                    break

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({"route_A": route_A, "route_B": route_B, "fwd_return": fwd_return})
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

    print(f"=== 다이버전스+20~30% 구간 신호 {len(all_signals)}건 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    only_B = [s for s in all_signals if s["route_B"] and not s["route_A"]]
    only_A = [s for s in all_signals if s["route_A"] and not s["route_B"]]
    both = [s for s in all_signals if s["route_A"] and s["route_B"]]
    either = [s for s in all_signals if s["route_A"] or s["route_B"]]
    neither = [s for s in all_signals if not s["route_A"] and not s["route_B"]]

    for label, sub in [
        ("경로B만 (이른진입)", only_B),
        ("경로A만 (확인진입)", only_A),
        ("둘 다 겹침", both),
        ("합집합(A or B)", either),
        ("둘 다 아님(참고)", neither),
    ]:
        wr, avg = win_rate(sub)
        if wr is None:
            print(f"{label}: 표본 없음")
            continue
        print(f"{label}: 표본 {len(sub)}건 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}%")

    print("\n해석: '합집합' 승률이 각 단독 경로보다 크게 떨어지지 않으면")
    print("두 경로를 병렬 운영해서 매매기회를 늘리는 게 유효함.")
    print("'둘 다 겹침'의 승률이 가장 높다면 그게 최고신뢰 신호.")


if __name__ == "__main__":
    main()
