"""
pullback_backtest.py
사용법: python pullback_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식(신규진입과 다른 눌림목 전용):
  게이트: 최근 60일간 10% 이상 상승(상승추세 확인)
  조정폭: 최근 60일 고점 대비 하락률
  Z-score: 최근 90일 RSI 평균/표준편차 기준(1년 전체가 아님)
  ADX: 22 이상

각 신호에 대해 조정폭 구간별/Z-score 구간별 승률을 확인.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

LARGECAP = ["ORCL", "ADBE", "TXN", "LOW", "GS"]
SMALLCAP = ["CHPT", "AFRM", "NIO", "DKNG", "PLUG"]
TICKERS = LARGECAP + SMALLCAP

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
UPTREND_WINDOW = 60
UPTREND_MIN_GAIN = 0.10  # 60일간 10% 이상 상승


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 300:
        return None
    h, l, c = hist["High"], hist["Low"], hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    adx_series = ADXIndicator(h, l, c, window=14).adx()
    n = len(hist)

    rows = []
    start = 300
    end = n - FORWARD_DAYS
    for i in range(start, end):
        if i < UPTREND_WINDOW:
            continue
        price_now = float(c.iloc[i])
        price_60d_ago = float(c.iloc[i - UPTREND_WINDOW])
        if price_60d_ago <= 0:
            continue
        uptrend_gain = price_now / price_60d_ago - 1

        recent_high = float(c.iloc[i - UPTREND_WINDOW:i + 1].max())
        pullback_pct = (recent_high - price_now) / recent_high * 100

        # 게이트: 최근 60일간 고점 기준 10% 이상 상승했던 적 있어야 함(현재가 아니라 고점 기준)
        gain_from_start_to_high = recent_high / price_60d_ago - 1
        if gain_from_start_to_high < UPTREND_MIN_GAIN:
            continue
        if pullback_pct <= 0:
            continue  # 조정이 아니라 신고가 갱신 중

        rsi_window_90 = rsi.iloc[max(0, i - 90):i + 1].dropna()
        if len(rsi_window_90) < 60:
            continue
        rsi_mean_90 = float(rsi_window_90.mean())
        rsi_std_90 = float(rsi_window_90.std())
        if rsi_std_90 <= 0:
            continue
        rsi_last = float(rsi.iloc[i])
        pullback_zscore = (rsi_last - rsi_mean_90) / rsi_std_90

        adx_val = float(adx_series.iloc[i]) if not pd.isna(adx_series.iloc[i]) else None

        entry = price_now
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        rows.append({
            "ticker": ticker, "pullback_pct": pullback_pct, "zscore": pullback_zscore,
            "adx": adx_val, "fwd_return": fwd_return, "win": fwd_return > 0,
        })
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_rows = []
    for t in TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)
            print(f"{t}: 눌림목 후보 {len(rows)}건")

    print(f"\n=== 눌림목 후보 총 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    print("[조정폭(%) 구간별]")
    for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 999)]:
        sub = [r for r in all_rows if lo <= r["pullback_pct"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {lo}~{hi if hi < 999 else '+'}%: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[눌림 Z-score(최근90일 기준) 구간별]")
    for lo, hi in [(-999, -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0), (0, 999)]:
        sub = [r for r in all_rows if lo <= r["zscore"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            lo_l = str(lo) if lo > -999 else "~"
            hi_l = str(hi) if hi < 999 else "이상"
            print(f"  {lo_l}~{hi_l}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[ADX 구간별]")
    for lo, hi in [(0, 15), (15, 20), (20, 25), (25, 30), (30, 999)]:
        sub = [r for r in all_rows if r["adx"] is not None and lo <= r["adx"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {lo}~{hi if hi < 999 else '+'}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")


if __name__ == "__main__":
    main()
