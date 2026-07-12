"""
pep_diagnose.py
사용법: python pep_diagnose.py

동작: PEP(펩시)에 대해 지금 fetch_indicators.py와 동일한 로직으로
      실시간 저점 목록, 최근 두 저점의 가격/RSI, 간격, 다이버전스 판정
      과정을 전부 출력해서 왜 다이버전스가 안 뜨는지(또는 뜨는지) 진단한다.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

TICKER = "PEP"
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
    hist = yf.Ticker(TICKER).history(period="1y")
    c = hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    close_values = c.values
    n = len(close_values)

    print(f"=== {TICKER} 진단 ===\n")
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

    pos1, pos2 = realtime_lows[-2], realtime_lows[-1]
    gap_days = pos2 - pos1
    is_today_new_low = (pos2 == n - 1)

    price1, price2 = float(c.iloc[pos1]), float(c.iloc[pos2])
    rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
    price_today = float(c.iloc[-1])

    print(f"\n[최근 두 저점 비교]")
    print(f"  저점1: {hist.index[pos1].date()} 종가 {price1:.2f} RSI {rsi1:.1f}")
    print(f"  저점2: {hist.index[pos2].date()} 종가 {price2:.2f} RSI {rsi2:.1f}")
    print(f"  간격: {gap_days}일 (허용범위 {MIN_GAP_DAYS}~{MAX_GAP_DAYS}일)")
    print(f"  오늘이 저점2인가: {is_today_new_low}")
    print(f"  오늘 종가: {price_today:.2f}")

    within_tolerance = bool(price_today >= price2 and (price_today - price2) / price2 <= TOLERANCE_PCT)
    print(f"  저점2 대비 오차범위(3%) 이내인가: {within_tolerance}")

    gap_ok = MIN_GAP_DAYS <= gap_days <= MAX_GAP_DAYS
    print(f"\n  간격 조건 통과: {gap_ok}")
    print(f"  신선함/오차범위 조건 통과: {is_today_new_low or within_tolerance}")

    if gap_ok and (is_today_new_low or within_tolerance):
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


if __name__ == "__main__":
    main()
