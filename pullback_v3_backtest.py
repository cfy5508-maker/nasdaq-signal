"""
pullback_v3_backtest.py
사용법: python pullback_v3_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식(눌림목 v3 - 상승채널 + OBV):
  게이트(전부 충족):
    1. 최근스윙고점 > 직전스윙고점 (Higher High)
    2. 최근스윙저점 > 직전스윙저점 (Higher Low)
    3. 직전스윙고점 -> 최근스윙저점 구간에 하루 -5%+ 급락 캔들 없음
    4. 종가 > 40일선
    5. OBV 최근스윙저점 > OBV 직전스윙저점 (매수세 추세 안 꺾임)
  참고: 트리거캔들(해머/엔걸핑/모닝스타+돌파, 또는 긴아래꼬리) 여부
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.volume import OnBalanceVolumeIndicator

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SWING_ORDER = 5
CRASH_THRESHOLD = -0.05  # 하루 -5% 이상 급락


def find_swing_lows(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n - order):
        window = close_values[i-order:i+order+1]
        if close_values[i] == window.min() and (window == close_values[i]).sum() == 1:
            positions.append(i)
    return positions


def find_swing_highs(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n - order):
        window = close_values[i-order:i+order+1]
        if close_values[i] == window.max() and (window == close_values[i]).sum() == 1:
            positions.append(i)
    return positions


def detect_hammer(o, h, l, c, i):
    body = abs(c.iloc[i] - o.iloc[i])
    lower_wick = min(o.iloc[i], c.iloc[i]) - l.iloc[i]
    upper_wick = h.iloc[i] - max(o.iloc[i], c.iloc[i])
    return bool(body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5)


def detect_bullish_engulfing(o, h, l, c, i):
    if i < 1:
        return False
    return bool(c.iloc[i] > o.iloc[i-1] and o.iloc[i] < c.iloc[i-1] and c.iloc[i] > c.iloc[i-1] and o.iloc[i-1] > c.iloc[i-1])


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 300:
        return None
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    close_values = c.values
    daily_ret = c.pct_change()
    sma40 = c.rolling(40).mean()
    obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    obv_values = obv.values
    n = len(hist)

    swing_lows_full = find_swing_lows(close_values, order=SWING_ORDER)
    swing_highs_full = find_swing_highs(close_values, order=SWING_ORDER)
    swing_lows_obv_full = find_swing_lows(obv_values, order=SWING_ORDER)

    rows = []
    start = 300
    end = n - FORWARD_DAYS
    for i in range(start, end):
        # i 시점까지 "완전히 확정된" 스윙 포인트만 사용 (p+order<=i 여야 미래참조 없음)
        lows_upto = [p for p in swing_lows_full if p + SWING_ORDER <= i]
        highs_upto = [p for p in swing_highs_full if p + SWING_ORDER <= i]
        obv_lows_upto = [p for p in swing_lows_obv_full if p + SWING_ORDER <= i]

        if len(lows_upto) < 2 or len(highs_upto) < 2 or len(obv_lows_upto) < 2:
            continue

        recent_low, prior_low = lows_upto[-1], lows_upto[-2]
        recent_high, prior_high = highs_upto[-1], highs_upto[-2]

        cond1 = close_values[recent_high] > close_values[prior_high]
        cond2 = close_values[recent_low] > close_values[prior_low]
        if not (cond1 and cond2):
            continue

        # 직전스윙고점 -> 최근스윙저점 구간 (시간순 확인)
        if prior_high >= recent_low:
            continue
        segment_ret = daily_ret.iloc[prior_high+1:recent_low+1]
        cond3 = bool((segment_ret >= CRASH_THRESHOLD).all()) if len(segment_ret) > 0 else True

        sma40_now = float(sma40.iloc[i]) if not pd.isna(sma40.iloc[i]) else None
        cond4 = bool(sma40_now is not None and close_values[i] > sma40_now)

        recent_obv_low, prior_obv_low = obv_lows_upto[-1], obv_lows_upto[-2]
        cond5 = bool(obv_values[recent_obv_low] > obv_values[prior_obv_low])

        gate_pass = cond1 and cond2 and cond3 and cond4 and cond5
        if not gate_pass:
            continue

        has_trigger = detect_hammer(o, h, l, c, i) or detect_bullish_engulfing(o, h, l, c, i)

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        rows.append({"has_trigger": has_trigger, "fwd_return": fwd_return, "win": fwd_return > 0})
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_rows = []
    for t in LARGECAP_TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)
            print(f"{t}: 게이트 통과 {len(rows)}건")

    print(f"\n=== 눌림목 v3(상승채널+OBV) 게이트 통과 총 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    wr, avg = win_rate(all_rows)
    print(f"전체: 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}%\n")

    trig = [r for r in all_rows if r["has_trigger"]]
    no_trig = [r for r in all_rows if not r["has_trigger"]]
    wr_t, avg_t = win_rate(trig)
    wr_n, avg_n = win_rate(no_trig)
    if wr_t is not None:
        print(f"트리거캔들 있음: 표본 {len(trig)} | 승률 {wr_t:.1f}% | 기준대비 {wr_t-wr:+.1f}%p")
    if wr_n is not None:
        print(f"트리거캔들 없음: 표본 {len(no_trig)} | 승률 {wr_n:.1f}% | 기준대비 {wr_n-wr:+.1f}%p")


if __name__ == "__main__":
    main()
