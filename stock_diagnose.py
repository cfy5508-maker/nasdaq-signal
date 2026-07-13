"""
stock_diagnose.py
사용법: python stock_diagnose.py TICKER

동작: 지정한 종목에 대해 fetch_indicators.py와 동일한 다이버전스 로직으로
      실시간 저점 목록, 저점1(구간최저가)/저점2(최근) 비교, 간격, 오차범위,
      최종 pass/warn/fail 판정 과정을 전부 출력해서 진단한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

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


def detect_bullish_engulfing(o, h, l, c):
    return bool(c.iloc[-1] > o.iloc[-2] and o.iloc[-1] < c.iloc[-2] and c.iloc[-1] > c.iloc[-2] and o.iloc[-2] > c.iloc[-2])


def detect_hammer(o, h, l, c):
    body = abs(c.iloc[-1] - o.iloc[-1])
    lower_wick = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
    upper_wick = h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])
    return bool(body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5)


def detect_morning_star(o, h, l, c):
    d1_bear = c.iloc[-3] < o.iloc[-3]
    d2_small = abs(c.iloc[-2] - o.iloc[-2]) < abs(c.iloc[-3] - o.iloc[-3]) * 0.4
    d3_bull = c.iloc[-1] > o.iloc[-1] and c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2
    return bool(d1_bear and d2_small and d3_bull)


def detect_long_lower_wick(o, h, l, c):
    body = abs(c.iloc[-1] - o.iloc[-1])
    lower_wick = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
    total_range = h.iloc[-1] - l.iloc[-1]
    if total_range <= 0:
        return False
    return bool((lower_wick / total_range) >= 0.5)


def trigger_with_breakout(o, h, l, c):
    """최근 1~3일 안에 반전캔들이 확정되고, 그 이후 종가가 그 캔들의 고점을 돌파했는지 확인."""
    n = len(c)
    detail = []
    for lookback in range(1, 4):
        idx = n - 1 - lookback
        if idx < 3:
            continue
        sub_o, sub_h, sub_l, sub_c = o.iloc[:idx+1], h.iloc[:idx+1], l.iloc[:idx+1], c.iloc[:idx+1]
        is_hammer = detect_hammer(sub_o, sub_h, sub_l, sub_c)
        is_engulf = detect_bullish_engulfing(sub_o, sub_h, sub_l, sub_c)
        is_star = detect_morning_star(sub_o, sub_h, sub_l, sub_c) if idx + 1 >= 3 else False
        breakout = c.iloc[-1] > h.iloc[idx]
        detail.append((lookback, is_hammer, is_engulf, is_star, breakout, float(h.iloc[idx])))
        if (is_hammer or is_engulf or is_star) and breakout:
            pattern_name = "해머" if is_hammer else ("불리시엔걸핑" if is_engulf else "모닝스타")
            return True, pattern_name, lookback, detail
    return False, None, None, detail


def main():
    if len(sys.argv) < 2:
        print("사용법: python stock_diagnose.py TICKER")
        sys.exit(1)
    ticker = sys.argv[1].upper()

    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty:
        print(f"{ticker}: 데이터 없음")
        return
    o, h, l, c = hist["Open"], hist["High"], hist["Low"], hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    close_values = c.values
    n = len(close_values)

    print(f"=== {ticker} 진단 ===\n")
    print(f"전체 데이터 길이: {n}일\n")

    realtime_lows = find_realtime_lows(close_values, order=5)
    print(f"실시간 저점 후보 개수: {len(realtime_lows)}\n")

    print("[최근 저점 후보 10개]")
    for pos in realtime_lows[-10:]:
        date_str = str(hist.index[pos].date())
        print(f"  {date_str}: 종가 {float(c.iloc[pos]):.2f} | RSI {float(rsi.iloc[pos]):.1f}")

    if len(realtime_lows) < 2:
        print("\n저점이 2개 미만이라 다이버전스 판정 불가")
        return

    pos2 = realtime_lows[-1]
    candidates = [p for p in realtime_lows if p != pos2 and MIN_GAP_DAYS <= (pos2 - p) <= MAX_GAP_DAYS]
    if not candidates:
        print("\n간격 조건(3~30일) 안에 드는 이전 저점이 없어 다이버전스 판정 불가")
        return
    pos1 = min(candidates, key=lambda p: close_values[p])
    gap_days = pos2 - pos1
    is_today_new_low = (pos2 == n - 1)

    price1, price2 = float(c.iloc[pos1]), float(c.iloc[pos2])
    rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
    price_today = float(c.iloc[-1])

    print(f"\n[의미있는 저점 비교 - 저점1은 구간 내 최저가로 선정]")
    print(f"  저점1(구간최저): {hist.index[pos1].date()} 종가 {price1:.2f} RSI {rsi1:.1f}")
    print(f"  저점2(최근): {hist.index[pos2].date()} 종가 {price2:.2f} RSI {rsi2:.1f}")
    print(f"  간격: {gap_days}일 (허용범위 {MIN_GAP_DAYS}~{MAX_GAP_DAYS}일)")
    print(f"  오늘이 저점2인가: {is_today_new_low}")
    print(f"  오늘 종가: {price_today:.2f}")

    within_tolerance = bool(price_today >= price2 and (price_today - price2) / price2 <= TOLERANCE_PCT)
    print(f"  저점2 대비 오차범위(3%) 이내인가: {within_tolerance}")
    print(f"  신선함/오차범위 조건 통과: {is_today_new_low or within_tolerance}")

    if is_today_new_low or within_tolerance:
        price_lower_low = price2 < price1
        rsi_improved = rsi2 > rsi1
        print(f"\n  가격이 더 낮아졌는가(price2<price1): {price_lower_low}")
        print(f"  RSI가 개선됐는가(rsi2>rsi1): {rsi_improved}")
        if price_lower_low and rsi_improved:
            print("\n  => 판정: PASS (정석 다이버전스)")
        elif rsi_improved:
            print("\n  => 판정: WARN (이중바닥 성격)")
        else:
            print("\n  => 판정: FAIL (RSI도 개선 안 됨)")
    else:
        print("\n  => 판정: FAIL (게이트 조건 자체 미달)")

    # Z-score도 참고로 같이 출력
    rsi_window_1y = rsi.dropna()
    if len(rsi_window_1y) >= 30:
        rsi_mean_1y = float(rsi_window_1y.mean())
        rsi_std_1y = float(rsi_window_1y.std())
        rsi_last = float(rsi.iloc[-1])
        if rsi_std_1y > 0:
            zscore = (rsi_last - rsi_mean_1y) / rsi_std_1y
            print(f"\n[참고] 오늘 RSI Z-score(1년 평균 대비): {zscore:.2f} (오늘RSI {rsi_last:.1f}, 1년평균 {rsi_mean_1y:.1f}, 1년표준편차 {rsi_std_1y:.1f})")

    # 트리거캔들 진단
    print(f"\n[트리거캔들 진단]")
    today_hammer = detect_hammer(o, h, l, c)
    today_engulf = detect_bullish_engulfing(o, h, l, c)
    today_wick = detect_long_lower_wick(o, h, l, c)
    print(f"  오늘 캔들: 시가{float(o.iloc[-1]):.2f} 고가{float(h.iloc[-1]):.2f} 저가{float(l.iloc[-1]):.2f} 종가{float(c.iloc[-1]):.2f}")
    print(f"  오늘이 해머인가: {today_hammer}")
    print(f"  오늘이 불리시엔걸핑인가: {today_engulf}")
    print(f"  오늘이 긴아래꼬리(완화기준, 범위50%+)인가: {today_wick}")

    breakout_ok, pattern, lookback, detail = trigger_with_breakout(o, h, l, c)
    print(f"\n  [최근 1~3일 반전캔들+돌파 확인]")
    for lb, is_h, is_e, is_s, brk, high_val in detail:
        print(f"    {lb}일 전 캔들: 해머={is_h} 엔걸핑={is_e} 모닝스타={is_s} | 그날고점{high_val:.2f} 대비 오늘종가 돌파={brk}")
    if breakout_ok:
        print(f"\n  => 판정: PASS ({pattern}, {lookback}일 전 확정 + 돌파 완료)")
    elif today_wick:
        print(f"\n  => 판정: WARN (긴 아래꼬리만 있음, 돌파 확정은 없음)")
    else:
        print(f"\n  => 판정: FAIL (반전캔들도 없고 긴아래꼬리도 없음)")


if __name__ == "__main__":
    main()
