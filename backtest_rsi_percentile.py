"""
backtest_rsi_percentile.py
사용법: python backtest_rsi_percentile.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 여러 티커에 대해 1년 롤링 백분위 기준 RSI 눌림목 신호(15~35%)가 뜬 날 이후
      N영업일 뒤 수익률을 계산해서 승률/평균수익률을 산출하고,
      전체 기간 무작위 기준선과 비교한다. 종목별 + 전체 통합 결과를 모두 출력.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

TICKERS = ["NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD"]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK = 252  # 52주 롤링 윈도우(거래일 기준)


def rolling_percentile(series, window):
    result = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series)):
        window_vals = series.iloc[i-window:i]
        result.iloc[i] = (window_vals < series.iloc[i]).mean() * 100
    return result


def build_df(ticker):
    hist = yf.Ticker(ticker).history(period="3y")
    if hist.empty:
        return None
    close = hist["Close"]
    rsi = RSIIndicator(close, window=14).rsi()
    df = pd.DataFrame({"close": close, "rsi": rsi}).dropna()
    df["rsi_pctile"] = rolling_percentile(df["rsi"], LOOKBACK)
    df = df.dropna(subset=["rsi_pctile"])
    df["fwd_return"] = df["close"].shift(-FORWARD_DAYS) / df["close"] - 1
    df = df.dropna(subset=["fwd_return"])
    df["ticker"] = ticker
    return df


def stats_for(df, mask, label):
    sub = df[mask]
    if len(sub) == 0:
        print(f"  {label}: 표본 없음")
        return None
    win_rate = (sub["fwd_return"] > 0).mean() * 100
    avg_return = sub["fwd_return"].mean() * 100
    median_return = sub["fwd_return"].median() * 100
    print(f"  {label}: 표본 {len(sub)}일 | 승률 {win_rate:.1f}% | 평균수익률 {avg_return:+.2f}% | 중앙값 {median_return:+.2f}%")
    return {"n": len(sub), "win_rate": win_rate, "avg_return": avg_return}


def main():
    all_dfs = []
    print(f"=== 종목별 결과 (신호 후 {FORWARD_DAYS}영업일 뒤 수익률 기준) ===\n")

    for t in TICKERS:
        df = build_df(t)
        if df is None:
            print(f"{t}: 데이터 없음\n")
            continue
        all_dfs.append(df)

        print(f"[{t}]")
        pass_mask = (df["rsi_pctile"] >= 15) & (df["rsi_pctile"] <= 35)
        all_mask = pd.Series(True, index=df.index)
        stats_for(df, pass_mask, "PASS(15~35%)")
        stats_for(df, all_mask, "전체 기준선")
        print()

    if not all_dfs:
        print("데이터를 하나도 못 가져왔습니다.")
        return

    combined = pd.concat(all_dfs, ignore_index=False)

    print("=== 10개 종목 통합 결과 ===\n")
    pass_mask = (combined["rsi_pctile"] >= 15) & (combined["rsi_pctile"] <= 35)
    warn_mask = ((combined["rsi_pctile"] >= 5) & (combined["rsi_pctile"] < 15)) | \
                ((combined["rsi_pctile"] > 35) & (combined["rsi_pctile"] <= 45))
    fail_mask = ~pass_mask & ~warn_mask
    all_mask = pd.Series(True, index=combined.index)

    pass_stats = stats_for(combined, pass_mask, "PASS(백분위 15~35%)")
    stats_for(combined, warn_mask, "WARN(5~15% 또는 35~45%)")
    stats_for(combined, fail_mask, "FAIL(그 외)")
    baseline_stats = stats_for(combined, all_mask, "전체 기준선(모든 날짜, 모든 종목)")

    print()
    if pass_stats and baseline_stats:
        gap = pass_stats["win_rate"] - baseline_stats["win_rate"]
        verdict = "의미 있는 신호로 판단 가능" if gap >= 5 else "기준선 대비 우위 불충분 — 구간 재조정 필요"
        print(f"PASS 승률({pass_stats['win_rate']:.1f}%) - 전체 기준선({baseline_stats['win_rate']:.1f}%) = {gap:+.1f}%p")
        print(f"판정: {verdict}")


if __name__ == "__main__":
    main()
