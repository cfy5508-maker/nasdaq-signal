"""
reversal_candle_backtest.py
사용법: python reversal_candle_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식(2~3단계 게이트 통과 신호에 대해):
  게이트: 진짜 스윙저점 다이버전스(간격3~30일, 오차범위 포함), 정석(PASS)만 대상
  검증: 저점2(다이버전스 확정일) "당일" 캔들이 반전형(해머/불리시엔걸핑/긴아래꼬리)인지
        - 돌파 확정 요구 없음(기존 trigger_with_breakout과 다름)
        - 이게 승률과 관련 있는지 확인
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

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
]
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MIN_GAP_DAYS = 3
MAX_GAP_DAYS = 30
TOLERANCE_PCT = 0.03


def find_realtime_lows(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n):
        past_window = close_values[i - order:i]
        if close_values[i] < past_window.min():
            positions.append(i)
    return positions


def is_hammer(o, h, l, c, pos):
    body = abs(c.iloc[pos] - o.iloc[pos])
    lower_wick = min(o.iloc[pos], c.iloc[pos]) - l.iloc[pos]
    upper_wick = h.iloc[pos] - max(o.iloc[pos], c.iloc[pos])
    return bool(body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5)


def is_bullish_engulfing(o, h, l, c, pos):
    if pos < 1:
        return False
    return bool(c.iloc[pos] > o.iloc[pos-1] and o.iloc[pos] < c.iloc[pos-1] and
                c.iloc[pos] > c.iloc[pos-1] and o.iloc[pos-1] > c.iloc[pos-1])


def is_long_lower_wick(o, h, l, c, pos):
    """단순 긴 아래꼬리: 해머보다 완화된 기준(상단꼬리 제한 없음)"""
    body = abs(c.iloc[pos] - o.iloc[pos])
    lower_wick = min(o.iloc[pos], c.iloc[pos]) - l.iloc[pos]
    total_range = h.iloc[pos] - l.iloc[pos]
    if total_range <= 0:
        return False
    return bool(lower_wick / total_range >= 0.5)  # 전체 캔들 범위의 절반 이상이 아래꼬리


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 280:
        return None
    o, h, l, c = hist["Open"], hist["High"], hist["Low"], hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    close_values = c.values
    n = len(hist)

    realtime_lows_full = find_realtime_lows(close_values, order=5)

    rows = []
    lows_upto = []
    li = 0
    for i in range(260, n - FORWARD_DAYS):
        while li < len(realtime_lows_full) and realtime_lows_full[li] <= i:
            lows_upto.append(realtime_lows_full[li])
            li += 1
        if len(lows_upto) < 2:
            continue
        pos2 = lows_upto[-1]
        candidates = [p for p in lows_upto if p != pos2 and MIN_GAP_DAYS <= (pos2 - p) <= MAX_GAP_DAYS]
        if not candidates:
            continue
        pos1 = min(candidates, key=lambda p: close_values[p])
        is_today_new_low = (pos2 == i)
        price2 = float(c.iloc[pos2])
        price_today = float(c.iloc[i])
        within_tol = bool(price_today >= price2 and (price_today - price2) / price2 <= TOLERANCE_PCT)
        if not (is_today_new_low or within_tol):
            continue

        price1 = float(c.iloc[pos1])
        rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
        if pd.isna(rsi1) or pd.isna(rsi2):
            continue
        if not (price2 < price1 and rsi2 > rsi1):
            continue  # 정석(PASS) 다이버전스만 대상

        hammer_at_low2 = is_hammer(o, h, l, c, pos2)
        engulf_at_low2 = is_bullish_engulfing(o, h, l, c, pos2)
        long_wick_at_low2 = is_long_lower_wick(o, h, l, c, pos2)
        has_reversal = bool(hammer_at_low2 or engulf_at_low2)

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        rows.append({"has_reversal": has_reversal, "hammer": hammer_at_low2,
                     "engulf": engulf_at_low2, "long_wick": long_wick_at_low2,
                     "fwd_return": fwd_return, "win": fwd_return > 0})
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_rows = []
    for t in TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)

    print(f"\n=== 정석 다이버전스(PASS) 신호 총 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%\n")

    print("=== 저점2 당일 반전캔들(해머 또는 불리시엔걸핑) 여부 ===")
    has = [r for r in all_rows if r["has_reversal"]]
    none_ = [r for r in all_rows if not r["has_reversal"]]
    wr_h, avg_h = win_rate(has)
    wr_n, avg_n = win_rate(none_)
    if wr_h is not None:
        print(f"  있음: 표본 {len(has)}건 | 승률 {wr_h:.1f}% | 기준대비 {wr_h-baseline_wr:+.1f}%p | 평균수익률 {avg_h:+.2f}%")
    if wr_n is not None:
        print(f"  없음: 표본 {len(none_)}건 | 승률 {wr_n:.1f}% | 기준대비 {wr_n-baseline_wr:+.1f}%p | 평균수익률 {avg_n:+.2f}%")

    print("\n=== 저점2 당일 해머만 별도 ===")
    hammer_rows = [r for r in all_rows if r["hammer"]]
    wr_hm, avg_hm = win_rate(hammer_rows)
    if wr_hm is not None:
        print(f"  해머 있음: 표본 {len(hammer_rows)}건 | 승률 {wr_hm:.1f}% | 기준대비 {wr_hm-baseline_wr:+.1f}%p")

    print("\n=== 저점2 당일 불리시엔걸핑만 별도 ===")
    engulf_rows = [r for r in all_rows if r["engulf"]]
    wr_e, avg_e = win_rate(engulf_rows)
    if wr_e is not None:
        print(f"  엔걸핑 있음: 표본 {len(engulf_rows)}건 | 승률 {wr_e:.1f}% | 기준대비 {wr_e-baseline_wr:+.1f}%p")

    print("\n=== 저점2 당일 '긴 아래꼬리'(완화된 기준: 전체범위의 50%+ 아래꼬리) ===")
    wick_yes = [r for r in all_rows if r["long_wick"]]
    wick_no = [r for r in all_rows if not r["long_wick"]]
    wr_wy, avg_wy = win_rate(wick_yes)
    wr_wn, avg_wn = win_rate(wick_no)
    if wr_wy is not None:
        print(f"  있음: 표본 {len(wick_yes)}건 | 승률 {wr_wy:.1f}% | 기준대비 {wr_wy-baseline_wr:+.1f}%p")
    if wr_wn is not None:
        print(f"  없음: 표본 {len(wick_no)}건 | 승률 {wr_wn:.1f}% | 기준대비 {wr_wn-baseline_wr:+.1f}%p")


if __name__ == "__main__":
    main()
