"""
adx_threshold_backtest.py
사용법: python adx_threshold_backtest.py [FORWARD_DAYS] [LOOKBACK_DAYS]
기본값: FORWARD_DAYS=10, LOOKBACK_DAYS=150

동작: 10개 종목에 대해 ADX 임계값을 여러 개(10,12,15,18,20,25,30)로 스윕하며,
      "ADX>=임계값 and +DI>-DI"(추세지속 조건) 였던 날들의 이후 N영업일 승률을
      전체 기준선과 비교. 종목별 자체 기준선 대비 우위의 일관성도 확인.
"""
import sys, os
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import ADXIndicator

TICKERS = ["NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD"]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
MIN_HISTORY = 40
THRESHOLDS = [10, 12, 15, 18, 20, 25, 30]


def build_df(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    h, l, c = hist["High"], hist["Low"], hist["Close"]
    adx_ind = ADXIndicator(h, l, c, window=14)
    df = pd.DataFrame({
        "close": c,
        "adx": adx_ind.adx(),
        "plus_di": adx_ind.adx_pos(),
        "minus_di": adx_ind.adx_neg(),
    }).dropna()
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

    print(f"=== ADX 임계값별 '추세지속(ADX>=X and +DI>-DI)' 신호, {FORWARD_DAYS}영업일 뒤 승률 ===")
    print(f"{'ADX>=':>6} | {'평균gap':>8} | {'gap표준편차':>10} | {'유효종목수':>8} | {'평가'}")
    print("-" * 65)

    results = []
    for thresh in THRESHOLDS:
        gaps = []
        total_n = 0
        for t, df in per_ticker.items():
            mask = (df["adx"] >= thresh) & (df["plus_di"] > df["minus_di"])
            sub = df[mask]
            if len(sub) < 10:
                continue
            wr = win_rate(sub)
            gaps.append(wr - per_ticker_baseline[t])
            total_n += len(sub)

        if len(gaps) < 5:
            print(f"{thresh:>5}  | 유효 종목 부족(표본<10인 종목 다수)")
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
        print(f"{thresh:>5}  | {mean_gap:>+7.1f}p | {std_gap:>9.1f}p | {n_valid:>8} | {verdict} (총표본 {total_n}일)")

    print()
    consistent = [r for r in results if r[5] == "일관된 우위"]
    consistent.sort(key=lambda r: r[1], reverse=True)

    print("=== 일관되게 우위를 보이는 ADX 임계값 ===")
    if consistent:
        for thresh, mean_gap, std_gap, n, total_n, verdict in consistent:
            print(f"  ADX>={thresh}: 평균우위 {mean_gap:+.1f}%p, 표준편차 {std_gap:.1f}%p, 유효종목 {n}개, 총표본 {total_n}일")
        best = consistent[0]
        print(f"\n결론: ADX>={best[0]} 가 가장 일관되게 우위를 보임. 6단계 임계값으로 재설정 고려.")
    else:
        print("  없음")
        print("\n결론: 어느 임계값도 일관된 우위를 못 보임. ADX 단독보다 다른 조합이 필요할 수 있음.")


if __name__ == "__main__":
    main()
