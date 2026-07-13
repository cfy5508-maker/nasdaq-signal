"""
pullback_gate_diagnose.py
사용법: python pullback_gate_diagnose.py TICKER

동작: fetch_indicators.py의 눌림목 게이트(2_pullback_gate) 로직을 그대로
      재현해서, 20/40/60일선 값과 근접 여부, 60일선 수렴 무시 여부까지
      단계별로 출력한다.
"""
import sys
import pandas as pd
import yfinance as yf

PULLBACK_MA_TOLERANCE = 0.03


def main():
    if len(sys.argv) < 2:
        print("사용법: python pullback_gate_diagnose.py TICKER")
        sys.exit(1)
    ticker = sys.argv[1].upper()

    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty:
        print(f"{ticker}: 데이터 없음")
        return
    c = hist["Close"]
    last_close = float(c.iloc[-1])

    sma20 = c.rolling(20).mean()
    sma40 = c.rolling(40).mean()
    sma60 = c.rolling(60).mean()

    sma20_now = float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else None
    sma40_now = float(sma40.iloc[-1]) if not pd.isna(sma40.iloc[-1]) else None
    sma60_now = float(sma60.iloc[-1]) if not pd.isna(sma60.iloc[-1]) else None

    print(f"=== {ticker} 눌림목 게이트 진단 ===\n")
    print(f"오늘 종가: {last_close:.2f}")
    print(f"20일선: {sma20_now:.2f}" if sma20_now else "20일선: 데이터 없음")
    print(f"40일선: {sma40_now:.2f}" if sma40_now else "40일선: 데이터 없음")
    print(f"60일선: {sma60_now:.2f}" if sma60_now else "60일선: 데이터 없음")

    dist_sma20 = abs(last_close - sma20_now) / sma20_now if sma20_now else None
    dist_sma40 = abs(last_close - sma40_now) / sma40_now if sma40_now else None
    near_sma20 = bool(dist_sma20 is not None and dist_sma20 <= PULLBACK_MA_TOLERANCE)
    near_sma40 = bool(dist_sma40 is not None and dist_sma40 <= PULLBACK_MA_TOLERANCE)

    print(f"\n20일선과의 거리: {dist_sma20*100:.2f}% (근접기준 3%)" if dist_sma20 is not None else "")
    print(f"40일선과의 거리: {dist_sma40*100:.2f}% (근접기준 3%)" if dist_sma40 is not None else "")
    print(f"20일선 근접 여부: {near_sma20}")
    print(f"40일선 근접 여부: {near_sma40}")

    sma60_is_lowest = bool(sma60_now is not None and sma20_now is not None and sma40_now is not None
                           and sma60_now < sma20_now and sma60_now < sma40_now)
    print(f"\n60일선이 가장 아래인가(20일선, 40일선 둘 다보다 낮음): {sma60_is_lowest}")

    if sma60_now is not None and sma20_now is not None:
        dist20_60 = abs(sma20_now - sma60_now) / sma60_now if sma60_now > 0 else None
        near20_60 = bool(dist20_60 is not None and dist20_60 <= PULLBACK_MA_TOLERANCE)
        print(f"20일선-60일선 거리: {dist20_60*100:.2f}% -> 수렴여부: {near20_60}" if dist20_60 is not None else "")
    else:
        near20_60 = False

    if sma60_now is not None and sma40_now is not None:
        dist40_60 = abs(sma40_now - sma60_now) / sma60_now if sma60_now > 0 else None
        near40_60 = bool(dist40_60 is not None and dist40_60 <= PULLBACK_MA_TOLERANCE)
        print(f"40일선-60일선 거리: {dist40_60*100:.2f}% -> 수렴여부: {near40_60}" if dist40_60 is not None else "")
    else:
        near40_60 = False

    ma_converged_ignore = sma60_is_lowest and (near20_60 or near40_60)
    near_ma_signal = near_sma20 or near_sma40

    print(f"\n20/40일선 근접신호(near_ma_signal): {near_ma_signal}")
    print(f"60일선 수렴 무시 조건(ma_converged_ignore): {ma_converged_ignore}")

    if ma_converged_ignore and near_ma_signal:
        pullback_gate = False
    else:
        pullback_gate = near_ma_signal

    print(f"\n최종 게이트 통과 여부: {pullback_gate}")
    print(f"=> 판정: {'PASS' if pullback_gate else 'FAIL'}")


if __name__ == "__main__":
    main()
