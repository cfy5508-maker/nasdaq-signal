"""
smci_swing_divergence_test.py
사용법: python smci_swing_divergence_test.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 진짜 국지적 스윙 저점(전후 ORDER일보다 낮은 지점)만 저점으로
      인정해서 불리쉬 다이버전스를 탐지한다. 연속된 두 스윙 저점 사이:
      가격은 더 낮아지고 RSI는 더 높아지는 경우만 진짜 다이버전스로 카운트.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

TICKER = "SMCI"
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ORDER = 5
MAX_GAP = 60


def find_swing_lows(c, order=5):
    swing_positions = []
    values = c.values
    n = len(values)
    for i in range(order, n - order):
        window = values[i-order:i+order+1]
        if values[i] == window.min() and (window == values[i]).sum() == 1:
            swing_positions.append(i)
    return swing_positions


def main():
    hist = yf.Ticker(TICKER).history(period="2y")
    c = hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    macd_hist = MACD(c).macd_diff()
    n = len(hist)

    swing_positions = find_swing_lows(c, order=ORDER)
    print(f"=== {TICKER}: 최근 2년간 진짜 스윙 저점 {len(swing_positions)}개 발견 ===\n")

    print("[스윙 저점 목록 (최근 20개)]")
    for pos in swing_positions[-20:]:
        date_str = str(hist.index[pos].date())
        print(f"  {date_str}: 종가 {float(c.iloc[pos]):.2f} | RSI {float(rsi.iloc[pos]):.1f}")

    print(f"\n[연속 스윙저점 쌍에서 불리쉬 다이버전스 판정]")
    rows = []
    for idx in range(len(swing_positions) - 1):
        pos1 = swing_positions[idx]
        pos2 = swing_positions[idx + 1]
        gap = pos2 - pos1
        if gap > MAX_GAP:
            continue

        price1, price2 = float(c.iloc[pos1]), float(c.iloc[pos2])
        rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
        if pd.isna(rsi1) or pd.isna(rsi2):
            continue

        is_divergence = bool(price2 < price1 and rsi2 > rsi1)
        date1, date2 = str(hist.index[pos1].date()), str(hist.index[pos2].date())

        marker = "다이버전스 O" if is_divergence else "다이버전스 X"
        print(f"  {date1}(가{price1:.1f},R{rsi1:.1f}) -> {date2}(가{price2:.1f},R{rsi2:.1f}) | 간격{gap}일 | {marker}")

        if is_divergence:
            if pos2 + FORWARD_DAYS < n:
                entry = float(c.iloc[pos2])
                fwd_return = float(c.iloc[pos2 + FORWARD_DAYS]) / entry - 1
                rows.append({"date": date2, "fwd_return": fwd_return, "win": fwd_return > 0})

    print(f"\n=== 진짜 불리쉬 다이버전스 {len(rows)}건 (10일 뒤 기준) ===")
    for r in rows:
        result = "승리" if r["win"] else "패배"
        print(f"  {r['date']}: {r['fwd_return']*100:+.2f}% | {result}")

    if rows:
        wins = sum(1 for r in rows if r["win"])
        print(f"\n승률: {wins}/{len(rows)} = {wins/len(rows)*100:.1f}%")
    else:
        print("\n다이버전스 신호 없음")


if __name__ == "__main__":
    main()
