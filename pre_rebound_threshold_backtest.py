"""
pre_rebound_threshold_backtest.py
사용법: python pre_rebound_threshold_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 중소형주에서 다이버전스 발생 후 5~7일 지난 시점의 신호들을 모아서,
      "신저점(low2) 대비 오늘까지 이미 몇 % 반등했는지"를 잘게 나눈 구간별로
      이후 FORWARD_DAYS 뒤 승률을 확인한다.
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
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = 500
MIN_HISTORY = 260

REBOUND_BANDS = [(-999, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]


def detect_bullish_engulfing(o, h, l, c):
    return c.iloc[-1] > o.iloc[-2] and o.iloc[-1] < c.iloc[-2] and c.iloc[-1] > c.iloc[-2] and o.iloc[-2] > c.iloc[-2]


def detect_hammer(o, h, l, c):
    body = abs(c.iloc[-1] - o.iloc[-1])
    lower_wick = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
    upper_wick = h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])
    return body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5


def detect_morning_star(o, h, l, c):
    if len(c) < 3:
        return False
    d1_bear = c.iloc[-3] < o.iloc[-3]
    d2_small = abs(c.iloc[-2] - o.iloc[-2]) < abs(c.iloc[-3] - o.iloc[-3]) * 0.4
    d3_bull = c.iloc[-1] > o.iloc[-1] and c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2
    return d1_bear and d2_small and d3_bull


def trigger_with_breakout(o, h, l, c):
    n = len(c)
    for lookback in range(1, 4):
        idx = n - 1 - lookback
        if idx < 3:
            continue
        sub_o, sub_h, sub_l, sub_c = o.iloc[:idx+1], h.iloc[:idx+1], l.iloc[:idx+1], c.iloc[:idx+1]
        is_hammer = detect_hammer(sub_o, sub_h, sub_l, sub_c)
        is_engulf = detect_bullish_engulfing(sub_o, sub_h, sub_l, sub_c)
        is_star = detect_morning_star(sub_o, sub_h, sub_l, sub_c)
        if (is_hammer or is_engulf or is_star) and c.iloc[-1] > h.iloc[idx]:
            return True
    return False


def find_signals(ticker):
    hist = yf.Ticker(ticker).history(period="3y")
    if hist.empty:
        return None
    o, h, c, l = hist["Open"], hist["High"], hist["Close"], hist["Low"]
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
        days_since_low2 = i - low2_pos

        if not (5 <= days_since_low2 < 7):
            continue

        low2_price = float(c[low2_idx])
        pre_rebound = (float(c.iloc[i]) - low2_price) / low2_price * 100

        # 호재성 급등 배제: 10% 이상 반등은 제외
        if pre_rebound >= 10:
            continue

        sub_o, sub_h, sub_l, sub_c = o.iloc[:i+1], h.iloc[:i+1], l.iloc[:i+1], c.iloc[:i+1]
        has_trigger_candle = trigger_with_breakout(sub_o, sub_h, sub_l, sub_c)

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({"pre_rebound": pre_rebound, "has_trigger_candle": has_trigger_candle, "fwd_return": fwd_return})
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

    print(f"=== 중소형주 5~7일 구간 신호 {len(all_signals)}건, {FORWARD_DAYS}영업일 뒤 기준 ===")
    print("(반등폭 10% 이상은 호재성 급등 배제를 위해 제외됨)\n")
    if not all_signals:
        print("신호 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_signals)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    print("[신저점 대비 선행 반등폭(%, 0~10% 구간) 세분화]")
    print(f"{'구간':>12} | {'표본':>6} | {'승률':>7} | {'평균수익률':>10} | {'기준선대비':>10}")
    print("-" * 65)
    for lo, hi in REBOUND_BANDS:
        sub = [s for s in all_signals if lo <= s["pre_rebound"] < hi]
        wr, avg = win_rate(sub)
        if wr is None:
            continue
        gap = wr - baseline_wr
        lo_label = str(lo) if lo > -999 else "~"
        hi_label = str(hi) if hi < 999 else "이상"
        print(f"{lo_label:>4}~{hi_label:<6}% | {len(sub):>6} | {wr:>6.1f}% | {avg:>+9.2f}% | {gap:>+9.1f}%p")

    print("\n[캔들 모양(반전캔들+돌파) 유무별 승률]")
    has_candle = [s for s in all_signals if s["has_trigger_candle"]]
    no_candle = [s for s in all_signals if not s["has_trigger_candle"]]
    for label, sub in [("반전캔들+돌파 있음", has_candle), ("반전캔들+돌파 없음", no_candle)]:
        wr, avg = win_rate(sub)
        if wr is None:
            print(f"  {label}: 표본 없음")
            continue
        gap = wr - baseline_wr
        print(f"  {label}: 표본 {len(sub)}일 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}% | 기준선 대비 {gap:+.1f}%p")

    print("\n[반등폭 0~10% 구간을 캔들 유무로 교차분석]")
    for lo, hi in REBOUND_BANDS:
        if lo < 0:
            continue
        sub_band = [s for s in all_signals if lo <= s["pre_rebound"] < hi]
        sub_candle = [s for s in sub_band if s["has_trigger_candle"]]
        sub_no_candle = [s for s in sub_band if not s["has_trigger_candle"]]
        wr_c, avg_c = win_rate(sub_candle)
        wr_nc, avg_nc = win_rate(sub_no_candle)
        print(f"  반등폭 {lo}~{hi}%:")
        if wr_c is not None:
            print(f"    캔들 있음: 표본 {len(sub_candle)}일 | 승률 {wr_c:.1f}%")
        if wr_nc is not None:
            print(f"    캔들 없음: 표본 {len(sub_no_candle)}일 | 승률 {wr_nc:.1f}%")

    print("\n해석: 승률이 뚜렷하게 오르기 시작하는 반등폭 구간과,")
    print("캔들 유무가 그 안에서 추가로 변별력을 갖는지가 핵심 확인 포인트.")


if __name__ == "__main__":
    main()
