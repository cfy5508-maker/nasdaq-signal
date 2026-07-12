"""
zone_20_30_threshold_sweep_backtest.py
사용법: python zone_20_30_threshold_sweep_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: "다이버전스+RSI백분위20~30%+궤적하락중" 그룹 안에서,
      1) RSI백분위 하락속도(5일간) 임계값
      2) 20일선 대비 이격도 임계값
      을 각각 스윕해서 어느 지점에서 승률이 최적화되는지 확인하고,
      두 조건을 결합했을 때의 최종 승률도 확인한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

SMALLCAP_TICKERS = [
    "CRSP", "BEAM", "EDIT", "NTLA", "SANA", "GTLB", "DDOG", "NET", "ESTC", "FROG",
    "IOT", "S", "DOCN", "FVRR", "UPWK", "TOST", "SQSP", "YELP", "ANGI", "CFLT",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = 250
MIN_HISTORY = 260

SPEED_THRESHOLDS = [15, 18, 20, 22, 25, 30]
DIST_THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


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
    sma20 = c.rolling(20).mean()

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

        if not (20 <= pctile_now < 30):
            continue
        if not (pctile_now < pctile_5ago):
            continue

        decline_speed = pctile_5ago - pctile_now

        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        dist_from_sma20 = (float(c.iloc[i]) - sma20_now) / sma20_now * 100 if sma20_now else float("nan")

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "decline_speed": decline_speed,
            "dist_from_sma20": dist_from_sma20,
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

    print(f"=== 기준 그룹(다이버전스+20~30%+궤적하락중) {len(all_signals)}건 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_signals)
    print(f"기준 승률: {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    print("=== 하락속도 <= 임계값 스윕 ===")
    print(f"{'임계값':>8} | {'표본':>6} | {'승률':>7} | {'기준대비':>8}")
    for th in SPEED_THRESHOLDS:
        sub = [s for s in all_signals if s["decline_speed"] <= th]
        wr, avg = win_rate(sub)
        if wr is None:
            continue
        gap = wr - baseline_wr
        print(f"{th:>7} | {len(sub):>6} | {wr:>6.1f}% | {gap:>+7.1f}%p")

    print("\n=== 20일선 이격도 <= 임계값 스윕 ===")
    print(f"{'임계값':>8} | {'표본':>6} | {'승률':>7} | {'기준대비':>8}")
    for th in DIST_THRESHOLDS:
        sub = [s for s in all_signals if not (isinstance(s["dist_from_sma20"], float) and np.isnan(s["dist_from_sma20"])) and s["dist_from_sma20"] <= th]
        wr, avg = win_rate(sub)
        if wr is None:
            continue
        gap = wr - baseline_wr
        print(f"{th:>7} | {len(sub):>6} | {wr:>6.1f}% | {gap:>+7.1f}%p")

    print("\n=== 두 조건 결합(하락속도<=20 AND 이격도<=3.0) ===")
    combo = [s for s in all_signals
             if s["decline_speed"] <= 20
             and not (isinstance(s["dist_from_sma20"], float) and np.isnan(s["dist_from_sma20"]))
             and s["dist_from_sma20"] <= 3.0]
    wr, avg = win_rate(combo)
    if wr is not None:
        gap = wr - baseline_wr
        print(f"표본 {len(combo)}건 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}% | 기준 대비 {gap:+.1f}%p")
    else:
        print("표본 없음")

    print("\n해석: 임계값을 좁힐수록 표본이 줄지만 승률이 오르는지가 핵심.")
    print("승률 개선폭 대비 표본 손실이 과하면 그 임계값은 과최적화 위험.")


if __name__ == "__main__":
    main()
