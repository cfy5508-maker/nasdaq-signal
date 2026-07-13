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


def main():
    if len(sys.argv) < 2:
        print("사용법: python stock_diagnose.py TICKER")
        sys.exit(1)
    ticker = sys.argv[1].upper()

    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty:
        print(f"{ticker}: 데이터 없음")
        return
    c = hist["Close"]
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
    if len(rsi_window_1y) >= 60:
        rsi_mean_1y = float(rsi_window_1y.mean())
        rsi_std_1y = float(rsi_window_1y.std())
        rsi_last = float(rsi.iloc[-1])
        if rsi_std_1y > 0:
            zscore = (rsi_last - rsi_mean_1y) / rsi_std_1y
            print(f"\n[참고] 오늘 RSI Z-score(1년 평균 대비): {zscore:.2f} (오늘RSI {rsi_last:.1f}, 1년평균 {rsi_mean_1y:.1f}, 1년표준편차 {rsi_std_1y:.1f})")


if __name__ == "__main__":
    main()
