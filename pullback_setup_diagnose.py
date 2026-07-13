"""
pullback_setup_diagnose.py
사용법: python pullback_setup_diagnose.py TICKER

동작: fetch_indicators.py의 눌림목 셋업/트리거 로직을 그대로 재현해서,
      최근 1~3일 각각에 대해 "셋업(20/40일선 근접+RSI50이상)"이었는지,
      그리고 그 셋업일 고가를 오늘 종가가 돌파했는지를 하나씩 보여준다.
"""
import sys
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

PULLBACK_MA_TOLERANCE = 0.03


def main():
    if len(sys.argv) < 2:
        print("사용법: python pullback_setup_diagnose.py TICKER")
        sys.exit(1)
    ticker = sys.argv[1].upper()

    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty:
        print(f"{ticker}: 데이터 없음")
        return
    o, h, l, c = hist["Open"], hist["High"], hist["Low"], hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    sma20 = c.rolling(20).mean()
    sma40 = c.rolling(40).mean()
    n = len(c)

    print(f"=== {ticker} 눌림목 셋업/트리거 진단 ===\n")
    print(f"오늘 종가: {float(c.iloc[-1]):.2f}\n")

    setup_confirmed = False
    setup_days_ago = None
    trigger_confirmed = False
    trigger_lag = None

    for days_ago in range(1, 4):
        idx = n - 1 - days_ago
        if idx < 40:
            print(f"  {days_ago}일 전: 데이터 부족")
            continue
        setup_price = float(c.iloc[idx])
        setup_sma20 = float(sma20.iloc[idx]) if not pd.isna(sma20.iloc[idx]) else None
        setup_sma40 = float(sma40.iloc[idx]) if not pd.isna(sma40.iloc[idx]) else None
        setup_rsi = float(rsi.iloc[idx]) if not pd.isna(rsi.iloc[idx]) else None
        date_str = str(hist.index[idx].date())

        if setup_sma20 is None or setup_sma40 is None or setup_rsi is None:
            print(f"  {days_ago}일 전({date_str}): 지표 데이터 부족")
            continue

        dist20 = abs(setup_price - setup_sma20) / setup_sma20
        dist40 = abs(setup_price - setup_sma40) / setup_sma40
        near20 = dist20 <= PULLBACK_MA_TOLERANCE
        near40 = dist40 <= PULLBACK_MA_TOLERANCE
        setup_near = near20 or near40
        rsi_ok = setup_rsi >= 50

        print(f"  {days_ago}일 전({date_str}): 종가{setup_price:.2f} 20일선{setup_sma20:.2f}(거리{dist20*100:.2f}%,근접={near20}) 40일선{setup_sma40:.2f}(거리{dist40*100:.2f}%,근접={near40}) RSI{setup_rsi:.1f}(50이상={rsi_ok})")

        if setup_near and rsi_ok:
            setup_confirmed = True
            setup_days_ago = days_ago
            setup_high = float(h.iloc[idx])
            today_close = float(c.iloc[-1])
            breakout = today_close > setup_high
            print(f"    => 셋업 조건 충족! 그날 고가: {setup_high:.2f}, 오늘종가: {today_close:.2f}, 돌파여부: {breakout}")
            if breakout:
                trigger_confirmed = True
                trigger_lag = days_ago
            break
        else:
            print(f"    => 셋업 조건 미충족 (근접={setup_near}, RSI50이상={rsi_ok})")

    print(f"\n=== 최종 판정 ===")
    if trigger_confirmed:
        print(f"status: pass (셋업 {setup_days_ago}일전 확정 + 오늘 돌파 완료)")
    elif setup_confirmed:
        print(f"status: warn (셋업 {setup_days_ago}일전 발생, 아직 돌파 대기중)")
    else:
        print(f"status: fail (셋업 자체가 없음)")


if __name__ == "__main__":
    main()
