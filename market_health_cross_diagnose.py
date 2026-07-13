"""
market_health_cross_diagnose.py
사용법: python market_health_cross_diagnose.py [ETF]
기본값: ETF=QQQ

동작: fetch_indicators.py의 golden/dead cross 로직을 그대로 재현해서,
      최근 15일간 20/40/60일선의 실제 값과 above/below 상태를 하루하루
      출력한다. 왜 golden_cross_recent가 True/False로 나오는지 직접 확인 가능.
"""
import sys
import pandas as pd
import yfinance as yf

ETF = sys.argv[1].upper() if len(sys.argv) > 1 else "QQQ"
CROSS_LOOKBACK = 10


def main():
    hist = yf.Ticker(ETF).history(period="1y")
    close = hist["Close"]
    n = len(close)

    sma20 = close.rolling(20).mean()
    sma40 = close.rolling(40).mean()
    sma60 = close.rolling(60).mean()
    above_20_40 = sma20 > sma40
    above_20_60 = sma20 > sma60

    print(f"=== {ETF} 최근 15일간 20/40/60일선 상태 ===\n")
    print(f"{'날짜':12} {'종가':>10} {'20일선':>10} {'40일선':>10} {'60일선':>10} {'20>40':>7} {'20>60':>7}")
    for i in range(n - 15, n):
        d = close.index[i].date()
        c_val = float(close.iloc[i])
        s20 = sma20.iloc[i]
        s40 = sma40.iloc[i]
        s60 = sma60.iloc[i]
        a40 = above_20_40.iloc[i]
        a60 = above_20_60.iloc[i]
        print(f"{str(d):12} {c_val:10.2f} {s20:10.2f} {s40:10.2f} {s60:10.2f} {str(bool(a40)):>7} {str(bool(a60)):>7}")

    print(f"\n=== 최근 {CROSS_LOOKBACK}일 이내 크로스 탐색(20-40일선) ===")
    golden_40, dead_40 = [], []
    for j in range(n - CROSS_LOOKBACK + 1, n):
        a_now, a_prev = above_20_40.iloc[j], above_20_40.iloc[j - 1]
        if pd.isna(a_now) or pd.isna(a_prev):
            continue
        d = close.index[j].date()
        if a_now and not a_prev:
            golden_40.append(d)
            print(f"  {d}: 골든크로스 발생(20일선이 40일선 위로)")
        elif not a_now and a_prev:
            dead_40.append(d)
            print(f"  {d}: 데드크로스 발생(20일선이 40일선 아래로)")
    if not golden_40 and not dead_40:
        print("  (크로스 없음)")

    print(f"\n=== 최근 {CROSS_LOOKBACK}일 이내 크로스 탐색(20-60일선) ===")
    golden_60, dead_60 = [], []
    for j in range(n - CROSS_LOOKBACK + 1, n):
        a_now, a_prev = above_20_60.iloc[j], above_20_60.iloc[j - 1]
        if pd.isna(a_now) or pd.isna(a_prev):
            continue
        d = close.index[j].date()
        if a_now and not a_prev:
            golden_60.append(d)
            print(f"  {d}: 골든크로스 발생(20일선이 60일선 위로)")
        elif not a_now and a_prev:
            dead_60.append(d)
            print(f"  {d}: 데드크로스 발생(20일선이 60일선 아래로)")
    if not golden_60 and not dead_60:
        print("  (크로스 없음)")

    print(f"\n=== 최종 판정 ===")
    print(f"golden_cross_recent(20-40, 최근 것 기준): {bool(golden_40) and (not dead_40 or golden_40[-1] > dead_40[-1])}")
    print(f"dead_cross_recent(20-60, 최근 것 기준): {bool(dead_60) and (not golden_60 or dead_60[-1] > golden_60[-1])}")


if __name__ == "__main__":
    main()
