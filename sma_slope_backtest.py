"""
sma_slope_backtest.py
사용법: python sma_slope_backtest.py [FORWARD_DAYS] [LOOKBACK_DAYS]
기본값: FORWARD_DAYS=10, LOOKBACK_DAYS=150

동작: 6단계(추세지속) 대체 후보로 "20일선 기울기"를 검증.
      slope = (sma20 - sma20.shift(5)) / sma20.shift(5) * 100  (5일간 20일선 변화율 %)
      여러 임계값(0%, 0.5%, 1%, 1.5%, 2%, 3%)을 스윕하며 "slope >= 임계값"인
      날들의 이후 N영업일 승률을 종목별 자체 기준선과 비교.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD"]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
MIN_HISTORY = 30

THRESHOLDS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]


def build_df(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    c = hist["Close"]
    sma20 = c.rolling(20).mean()
    slope = (sma20 - sma20.shift(5)) / sma20.shift(5) * 100

    df = pd.DataFrame({"close": c, "sma20": sma20, "slope": slope}).dropna()
    df = df.iloc[-(LOOKBACK_DAYS + FORWARD_DAYS + MIN_HISTORY):]
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

    print(f"=== 20일선 기울기(5일간 변화율%) 임계값별 신호, {FORWARD_DAYS}영업일 뒤 승률 ===")
    print(f"{'기울기>=':>9} | {'평균gap':>8} | {'gap표준편차':>10} | {'유효종목수':>8} | {'평가'}")
    print("-" * 68)

    results = []
    for thresh in THRESHOLDS:
        gaps = []
        total_n = 0
        for t, df in per_ticker.items():
            mask = df["slope"] >= thresh
            sub = df[mask]
            if len(sub) < 10:
                continue
            wr = win_rate(sub)
            gaps.append(wr - per_ticker_baseline[t])
            total_n += len(sub)

        if len(gaps) < 5:
            print(f"{thresh:>7}%  | 유효 종목 부족(표본<10인 종목 다수)")
            continue

        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        n_valid = len(gaps)

        if mean_gap > 0 and std_gap < abs(mean_gap):
            verdict = "일관된 우위"
        elif mean_gap > 0:
            verdict = "우위는 있으나 편차 큼"
        else:
            verdict = "우위 없음"

        results.append((thresh, mean_gap, std_gap, n_valid, total_n, verdict))
        print(f"{thresh:>7}%  | {mean_gap:>+7.1f}p | {std_gap:>9.1f}p | {n_valid:>8} | {verdict} (총표본 {total_n}일)")

    print()
    consistent = [r for r in results if r[5] == "일관된 우위"]
    consistent.sort(key=lambda r: r[1], reverse=True)

    print("=== 일관되게 우위를 보이는 기울기 임계값 ===")
    if consistent:
        for thresh, mean_gap, std_gap, n, total_n, verdict in consistent:
            print(f"  기울기>={thresh}%: 평균우위 {mean_gap:+.1f}%p, 표준편차 {std_gap:.1f}%p, 유효종목 {n}개, 총표본 {total_n}일")
        best = consistent[0]
        print(f"\n결론: 기울기>={best[0]}% 가 가장 일관되게 우위를 보임. 6단계를 이걸로 교체 고려.")
    else:
        print("  없음")
        print("\n결론: 20일선 기울기도 일관된 우위를 못 보임.")
        print("6단계(추세지속) 자체를 가중치 축소하거나 제거하는 게 더 정직한 선택일 수 있음.")


if __name__ == "__main__":
    main()
