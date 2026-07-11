"""
backtest_rsi_percentile.py
사용법: python backtest_rsi_percentile.py [TICKER] [FORWARD_DAYS]
기본값: TICKER=NVDA, FORWARD_DAYS=10

동작: 1년 롤링 백분위 기준 RSI 눌림목 신호(15~35%)가 뜬 날 이후
      N영업일 뒤 수익률을 계산해서 승률/평균수익률을 산출하고,
      전체 기간 무작위 기준선과 비교한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

TICKER = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
FORWARD_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
LOOKBACK = 252  # 52주 롤링 윈도우(거래일 기준)


def main():
    hist = yf.Ticker(TICKER).history(period="3y")
    if hist.empty:
        print(f"데이터 없음: {TICKER}")
        return

    close = hist["Close"]
    rsi = RSIIndicator(close, window=14).rsi()

    df = pd.DataFrame({"close": close, "rsi": rsi}).dropna()

    # 롤링 1년(252거래일) 기준 그날 RSI의 백분위
    def rolling_percentile(series, window):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_vals = series.iloc[i-window:i]
            result.iloc[i] = (window_vals < series.iloc[i]).mean() * 100
        return result

    df["rsi_pctile"] = rolling_percentile(df["rsi"], LOOKBACK)
    df = df.dropna(subset=["rsi_pctile"])

    # N영업일 뒤 수익률
    df["fwd_return"] = df["close"].shift(-FORWARD_DAYS) / df["close"] - 1
    df = df.dropna(subset=["fwd_return"])

    def stats_for(mask, label):
        sub = df[mask]
        if len(sub) == 0:
            print(f"{label}: 표본 없음")
            return
        win_rate = (sub["fwd_return"] > 0).mean() * 100
        avg_return = sub["fwd_return"].mean() * 100
        median_return = sub["fwd_return"].median() * 100
        print(f"{label}: 표본 {len(sub)}일 | 승률 {win_rate:.1f}% | 평균수익률 {avg_return:+.2f}% | 중앙값 {median_return:+.2f}%")

    print(f"=== {TICKER}, {FORWARD_DAYS}영업일 뒤 수익률 기준 ===\n")

    # 1) pass 구간(15~35%)
    pass_mask = (df["rsi_pctile"] >= 15) & (df["rsi_pctile"] <= 35)
    stats_for(pass_mask, "PASS 신호(백분위 15~35%)")

    # 2) warn 구간
    warn_mask = ((df["rsi_pctile"] >= 5) & (df["rsi_pctile"] < 15)) | \
                ((df["rsi_pctile"] > 35) & (df["rsi_pctile"] <= 45))
    stats_for(warn_mask, "WARN 신호(5~15% 또는 35~45%)")

    # 3) fail 구간
    fail_mask = ~pass_mask & ~warn_mask
    stats_for(fail_mask, "FAIL 구간(그 외)")

    # 4) 전체 기준선(아무 날짜나)
    all_mask = pd.Series(True, index=df.index)
    stats_for(all_mask, "전체 기준선(모든 날짜)")

    print(f"\n해석 기준: PASS 승률이 전체 기준선보다 유의미하게(5%p 이상) 높아야 신호로서 의미 있음")


if __name__ == "__main__":
    main()
