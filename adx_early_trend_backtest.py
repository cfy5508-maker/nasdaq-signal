"""
adx_early_trend_backtest.py
사용법: python adx_early_trend_backtest.py [FORWARD_DAYS] [LOOKBACK_DAYS]
기본값: FORWARD_DAYS=10, LOOKBACK_DAYS=150

동작: "ADX가 낮은 수준에서 막 올라오기 시작하는 초입"을 여러 절대수준 구간
      (0~15, 0~20, 0~25, 10~20, 10~25, 15~25, 15~30)으로 나누고,
      각 구간 안에서 "ADX가 5일 전보다 상승 중 + +DI>-DI"인 날들의
      이후 N영업일 승률을 종목별 자체 기준선과 비교한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import ADXIndicator

TICKERS = ["NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD"]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
MIN_HISTORY = 40

ZONES = [(0, 15), (0, 20), (0, 25), (10, 20), (10, 25), (15, 25), (15, 30)]


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
    df["adx_rising"] = df["adx"] > df["adx"].shift(5)
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

    print(f"=== ADX 초입구간별 '저수준+상승전환+DI우세' 신호, {FORWARD_DAYS}영업일 뒤 승률 ===")
    print(f"{'ADX구간':>10} | {'평균gap':>8} | {'gap표준편차':>10} | {'유효종목수':>8} | {'평가'}")
    print("-" * 68)

    results = []
    for lo, hi in ZONES:
        gaps = []
        total_n = 0
        for t, df in per_ticker.items():
            mask = (df["adx"] >= lo) & (df["adx"] < hi) & (df["adx_rising"]) & (df["plus_di"] > df["minus_di"])
            sub = df[mask]
            if len(sub) < 10:
                continue
            wr = win_rate(sub)
            gaps.append(wr - per_ticker_baseline[t])
            total_n += len(sub)

        if len(gaps) < 5:
            print(f"{lo:>3}~{hi:<3}  | 유효 종목 부족(표본<10인 종목 다수)")
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

        results.append((lo, hi, mean_gap, std_gap, n_valid, total_n, verdict))
        print(f"{lo:>3}~{hi:<3}  | {mean_gap:>+7.1f}p | {std_gap:>9.1f}p | {n_valid:>8} | {verdict} (총표본 {total_n}일)")

    print()
    consistent = [r for r in results if r[6] == "일관된 우위"]
    consistent.sort(key=lambda r: r[2], reverse=True)

    print("=== 일관되게 우위를 보이는 ADX 초입 구간 ===")
    if consistent:
        for lo, hi, mean_gap, std_gap, n, total_n, verdict in consistent:
            print(f"  ADX {lo}~{hi}: 평균우위 {mean_gap:+.1f}%p, 표준편차 {std_gap:.1f}%p, 유효종목 {n}개, 총표본 {total_n}일")
        best = consistent[0]
        print(f"\n결론: ADX {best[0]}~{best[1]} 구간(저수준+상승전환)이 가장 일관되게 우위를 보임.")
        print("6단계 로직을 'ADX가 이 구간에서 5일 전보다 상승 중'으로 재설계 고려.")
    else:
        print("  없음")
        print("\n결론: 초입 구간으로 봐도 일관된 우위가 없음.")
        print("ADX 계열 지표 자체가 이 6단계 목적(추세지속 확인)에 안 맞을 가능성 — ")
        print("다른 지표(예: 20일선 기울기, 가격모멘텀)로 교체 검토 필요.")


if __name__ == "__main__":
    main()
