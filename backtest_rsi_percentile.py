"""
backtest_rsi_percentile.py
사용법: python backtest_rsi_percentile.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 10개 종목 각각에 대해, RSI 백분위 구간별 승률이 "그 종목 자체의 전체
      기준선 승률" 대비 얼마나 우위인지(gap)를 구간마다 계산한다.
      평균 우위가 크더라도 종목마다 들쭉날쭉하면(표준편차 큼) 좋은 신호가
      아니라고 보고, "평균 우위는 양(+)이면서 종목간 편차가 작은" 구간을 찾는다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

TICKERS = ["TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH"]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK = 252
MIN_SAMPLES_PER_TICKER = 10  # 이보다 표본 적으면 그 종목은 해당 구간에서 제외

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
    return df


def win_rate(sub):
    if len(sub) == 0:
        return None
    return (sub["fwd_return"] > 0).mean() * 100


def main():
    per_ticker = {}
    per_ticker_baseline = {}

    for t in TICKERS:
        df = build_df(t)
        if df is None:
            print(f"{t}: 데이터 없음")
            continue
        per_ticker[t] = df
        per_ticker_baseline[t] = win_rate(df)

    if not per_ticker:
        print("데이터를 하나도 못 가져왔습니다.")
        return

    print("=== 종목별 자체 기준선 승률 ===")
    for t, b in per_ticker_baseline.items():
        print(f"  {t}: {b:.1f}%")
    print()

    print(f"=== 구간별 '종목 대비 우위(gap)' 일관성 분석 ({FORWARD_DAYS}영업일 뒤 수익률) ===")
    print(f"{'구간':>10} | {'평균gap':>8} | {'gap표준편차':>10} | {'유효종목수':>8} | {'평가'}")
    print("-" * 65)

    results = []
    for lo, hi in BANDS:
        gaps = []
        for t, df in per_ticker.items():
            mask = (df["rsi_pctile"] >= lo) & (df["rsi_pctile"] < hi)
            sub = df[mask]
            if len(sub) < MIN_SAMPLES_PER_TICKER:
                continue
            wr = win_rate(sub)
            gaps.append(wr - per_ticker_baseline[t])

        if len(gaps) < 6:  # 최소 6개 종목 이상에서 표본 확보돼야 신뢰
            continue

        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        n_valid = len(gaps)

        # 평가: 평균 우위가 양수(+)이고, 표준편차가 평균보다 작아야("일관성 있음") 좋은 신호
        if mean_gap > 0 and std_gap < abs(mean_gap):
            verdict = "일관된 우위"
        elif mean_gap > 0:
            verdict = "우위는 있으나 종목별 편차 큼"
        else:
            verdict = "우위 없음"

        results.append((lo, hi, mean_gap, std_gap, n_valid, verdict))
        print(f"{lo:>3}~{hi:<3}%  | {mean_gap:>+7.1f}p | {std_gap:>9.1f}p | {n_valid:>8} | {verdict}")

    print()
    # 정렬: "일관된 우위"인 것들 중 평균gap 큰 순
    consistent = [r for r in results if r[5] == "일관된 우위"]
    consistent.sort(key=lambda r: r[2], reverse=True)

    print("=== 일관되게 우위를 보이는 구간 (평균gap > 0, 표준편차 < 평균gap) ===")
    if consistent:
        for lo, hi, mean_gap, std_gap, n, verdict in consistent:
            print(f"  {lo}~{hi}%: 평균우위 {mean_gap:+.1f}%p, 표준편차 {std_gap:.1f}%p, 유효종목 {n}개")
        best = consistent[0]
        print(f"\n결론: {best[0]}~{best[1]}% 구간이 가장 일관되게 우위를 보임. 이 구간으로 재설정 권장.")
    else:
        print("  없음")
        print("\n결론: 모든 구간이 종목별로 들쭉날쭉하거나 우위 자체가 없음.")
        print("RSI 백분위만으로는 범용 신호로 쓰기 어려워 보임 — 3단계 가중치를 낮추거나,")
        print("추세 강도(ADX)별로 나눠서 재검증하는 걸 권장.")


if __name__ == "__main__":
    main()
