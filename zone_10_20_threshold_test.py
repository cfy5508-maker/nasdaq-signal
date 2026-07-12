"""
zone_10_20_threshold_test.py
사용법: python zone_10_20_threshold_test.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: RSI백분위 10~20% 구간(15~20%가 기존 최악 구간이었음)에
      "하락속도<=20, 20일선이격도<=3.0" 필터를 적용했을 때
      승률이 개선되는지 확인.
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

        if not (10 <= pctile_now < 20):
            continue

        decline_speed = pctile_5ago - pctile_now
        trending_up = pctile_now > pctile_5ago

        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        dist_from_sma20 = (float(c.iloc[i]) - sma20_now) / sma20_now * 100 if sma20_now else float("nan")

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "pctile_now": pctile_now,
            "trending_up": trending_up,
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

    print(f"=== RSI백분위 10~20% 구간 전체 {len(all_signals)}건 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_signals)
    print(f"필터 적용 전(10~20% 전체): 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    for lo, hi in [(10, 15), (15, 20)]:
        sub = [s for s in all_signals if lo <= s["pctile_now"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {lo}~{hi}%: 표본 {len(sub)}건, 승률 {wr:.1f}%")
    print()

    downtrend = [s for s in all_signals if not s["trending_up"]]
    wr, avg = win_rate(downtrend)
    if wr is not None:
        print(f"궤적 하락중만: 표본 {len(downtrend)}건 | 승률 {wr:.1f}% | 기준 대비 {wr-baseline_wr:+.1f}%p\n")
    else:
        print("궤적 하락중: 표본 없음\n")

    for th in [15, 20]:
        filtered = [s for s in downtrend if s["decline_speed"] <= th]
        wr_f, avg_f = win_rate(filtered)
        print(f"=== 궤적하락중 + 하락속도<={th} (이격도 조건 제외) ===")
        if wr_f is not None:
            gap = wr_f - baseline_wr
            print(f"표본 {len(filtered)}건 | 승률 {wr_f:.1f}% | 평균수익률 {avg_f:+.2f}% | 기준 대비 {gap:+.1f}%p\n")
        else:
            print("표본 없음\n")

    print("\n해석: 필터 적용 후 승률이 기준선(10~20% 전체)보다 뚜렷이 오르면")
    print("이 필터가 RSI구간과 무관한 보편적 조건이라는 뜻.")
    print("반대로 개선이 없으면 20~30% 구간에서만 특별히 작동했던 것.")


if __name__ == "__main__":
    main()
