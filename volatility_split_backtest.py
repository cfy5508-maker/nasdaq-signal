"""
volatility_split_backtest.py
사용법: python volatility_split_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 대형주 40종목(기존 두 세트 통합)의 최근 2년 변동성(일간수익률 연율화 표준편차)을
      계산해서 상위 20개(고변동성)/하위 20개(저변동성)로 재분류하고,
      각 그룹에서 4단계(다이버전스)·5단계(거래량) 승률을 따로 검증한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator

ALL_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = 250
MIN_HISTORY = 260


def compute_volatility(ticker):
    hist = yf.Ticker(ticker).history(period="2y")["Close"]
    if hist.empty or len(hist) < 100:
        return None
    daily_ret = hist.pct_change().dropna()
    annualized_vol = daily_ret.std() * (252 ** 0.5) * 100
    return annualized_vol


def build_df(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    c, v = hist["Close"], hist["Volume"]
    rsi = RSIIndicator(c, window=14).rsi()
    macd_hist = MACD(c).macd_diff()
    obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()

    n = len(hist)
    start = max(MIN_HISTORY, n - LOOKBACK_DAYS - FORWARD_DAYS)
    end = n - FORWARD_DAYS

    records = []
    for i in range(start, end):
        recent = c.iloc[max(0, i-90):i+1]
        if len(recent) < 90:
            continue
        low1_idx = recent.iloc[:45].idxmin()
        low2_idx = recent.iloc[45:].idxmin()
        bullish_divergence = bool(c[low2_idx] < c[low1_idx] and rsi[low2_idx] > rsi[low1_idx])

        mh = macd_hist.iloc[max(0, i-90):i+1]
        if len(mh) < 90:
            continue
        macd_rising = bool(mh.iloc[-1] > mh.iloc[:-1].min())

        stage4 = "pass" if (bullish_divergence and macd_rising) else \
                 "warn" if (bullish_divergence or macd_rising) else "fail"

        obv_signal = bool(obv.iloc[i] > obv.iloc[i-20] and c.iloc[i] < c.iloc[i-20]) if i >= 20 else False
        stage5 = "pass" if obv_signal else "warn"

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1
        records.append({"stage4": stage4, "stage5": stage5, "fwd_return": fwd_return})
    return records


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    print("=== 종목별 변동성(연율화, %) 계산 중 ===\n")
    vol_map = {}
    for t in ALL_TICKERS:
        vol = compute_volatility(t)
        if vol is not None:
            vol_map[t] = vol
            print(f"  {t}: {vol:.1f}%")

    sorted_tickers = sorted(vol_map.items(), key=lambda x: x[1], reverse=True)
    high_vol = [t for t, v in sorted_tickers[:20]]
    low_vol = [t for t, v in sorted_tickers[-20:]]

    print(f"\n고변동성 상위 20개: {high_vol}")
    print(f"저변동성 하위 20개: {low_vol}\n")

    for group_name, group_tickers in [("고변동성", high_vol), ("저변동성", low_vol)]:
        all_records = []
        for t in group_tickers:
            recs = build_df(t)
            if recs:
                all_records.extend(recs)

        baseline_wr, baseline_avg = win_rate(all_records)
        print(f"=== {group_name} 그룹 ({len(group_tickers)}종목, 표본 {len(all_records)}일, {FORWARD_DAYS}일 뒤) ===")
        print(f"전체 기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%")

        for stage_key, label in [("stage4", "4단계 다이버전스"), ("stage5", "5단계 거래량")]:
            print(f"  [{label}]")
            for status in ["pass", "warn", "fail"]:
                sub = [r for r in all_records if r[stage_key] == status]
                wr, avg = win_rate(sub)
                if wr is None:
                    print(f"    {status}: 표본 없음")
                    continue
                gap = wr - baseline_wr
                print(f"    {status}: 표본 {len(sub)}일 | 승률 {wr:.1f}% | 기준선 대비 {gap:+.1f}%p")
        print()


if __name__ == "__main__":
    main()
