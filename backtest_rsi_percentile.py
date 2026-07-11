"""
backtest_rsi_percentile.py
사용법: python backtest_rsi_percentile.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 10개 종목에 대해 RSI 백분위 구간을 여러 개(5~10, 10~20, ... 등)로 나눠서
      스윕하며, 각 구간에서 N영업일 뒤 승률/평균수익률이 전체 기준선 대비
      얼마나 우위인지 비교한다. 어느 구간이 실제로 신호로서 유효한지 찾기 위함.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

TICKERS = ["NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD"]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK = 252

# 스윕할 백분위 구간들 (하한, 상한)
BANDS = [
    (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35),
    (35, 40), (40, 45), (45, 50),
    (0, 10), (0, 15), (0, 20), (5, 15), (5, 20), (10, 25), (15, 35),
]


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


def compute_stats(sub):
    if len(sub) == 0:
        return None
    return {
        "n": len(sub),
        "win_rate": (sub["fwd_return"] > 0).mean() * 100,
        "avg_return": sub["fwd_return"].mean() * 100,
    }


def main():
    all_dfs = []
    for t in TICKERS:
        df = build_df(t)
        if df is not None:
            all_dfs.append(df)
        else:
            print(f"{t}: 데이터 없음")

    if not all_dfs:
        print("데이터를 하나도 못 가져왔습니다.")
        return

    combined = pd.concat(all_dfs, ignore_index=False)

    baseline = compute_stats(combined)
    print(f"=== 전체 기준선 ===")
    print(f"표본 {baseline['n']}일 | 승률 {baseline['win_rate']:.1f}% | 평균수익률 {baseline['avg_return']:+.2f}%\n")

    print(f"=== 구간별 스윕 결과 ({FORWARD_DAYS}영업일 뒤 수익률 기준, 10개 종목 통합) ===")
    print(f"{'구간':>12} | {'표본':>6} | {'승률':>7} | {'평균수익률':>10} | {'기준선 대비':>10}")
    print("-" * 60)

    results = []
    for lo, hi in BANDS:
        mask = (combined["rsi_pctile"] >= lo) & (combined["rsi_pctile"] < hi)
        stats = compute_stats(combined[mask])
        if stats is None or stats["n"] < 20:
            continue
        gap = stats["win_rate"] - baseline["win_rate"]
        results.append((lo, hi, stats, gap))
        print(f"{lo:>4}~{hi:<4}%   | {stats['n']:>6} | {stats['win_rate']:>6.1f}% | {stats['avg_return']:>+9.2f}% | {gap:>+9.1f}%p")

    print()
    results.sort(key=lambda r: r[3], reverse=True)
    print("=== 승률 우위 상위 5개 구간 ===")
    for lo, hi, stats, gap in results[:5]:
        print(f"{lo}~{hi}%: 승률 {stats['win_rate']:.1f}% (기준선 대비 {gap:+.1f}%p), 표본 {stats['n']}일")

    print()
    best = results[0]
    if best[3] >= 5:
        print(f"결론: {best[0]}~{best[1]}% 구간이 가장 유의미한 우위({best[3]:+.1f}%p)를 보임. 이 구간으로 재설정 고려.")
    else:
        print("결론: 어느 구간도 5%p 이상의 뚜렷한 우위를 보이지 않음. RSI 백분위 단독으로는 신뢰도 낮은 신호일 가능성.")


if __name__ == "__main__":
    main()
