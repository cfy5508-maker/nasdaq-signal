"""
pullback_v2_backtest.py
사용법: python pullback_v2_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식(대형주 중심, 6개 관측 지표 - 예측성 요소 배제):
  게이트: 최근 60일 고점이 60일전 대비 10%+ 상승했던 적 있음(상승추세 확인) + 현재 조정중
  1. 조정폭: 고점 대비 오늘 하락률
  2. 이평선 근접도: 20일선/50일선 대비 오늘 종가 이격도
  3. 고점 이후 경과일수
  4. 거래량 추세: 최근5일 vs 이전5일
  5. 오늘 캔들 모양: 반전캔들(해머/불리시엔걸핑) 여부
  6. ADX
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
UPTREND_WINDOW = 60
UPTREND_MIN_GAIN = 0.10


def detect_hammer(o, h, l, c, i):
    body = abs(c.iloc[i] - o.iloc[i])
    lower_wick = min(o.iloc[i], c.iloc[i]) - l.iloc[i]
    upper_wick = h.iloc[i] - max(o.iloc[i], c.iloc[i])
    return bool(body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5)


def detect_bullish_engulfing(o, h, l, c, i):
    if i < 1:
        return False
    return bool(c.iloc[i] > o.iloc[i-1] and o.iloc[i] < c.iloc[i-1] and c.iloc[i] > c.iloc[i-1] and o.iloc[i-1] > c.iloc[i-1])


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 300:
        return None
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    rsi = RSIIndicator(c, window=14).rsi()
    adx_series = ADXIndicator(h, l, c, window=14).adx()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
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

        window_close = c.iloc[i - UPTREND_WINDOW:i + 1]
        recent_high = float(window_close.max())
        high_pos_rel = window_close.values.argmax()
        high_pos = i - UPTREND_WINDOW + high_pos_rel

        gain_from_start_to_high = recent_high / price_60d_ago - 1
        if gain_from_start_to_high < UPTREND_MIN_GAIN:
            continue

        pullback_pct = (recent_high - price_now) / recent_high * 100
        if pullback_pct <= 0:
            continue

        days_since_high = i - high_pos

        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        sma50_now = float(sma50.iloc[i]) if not pd.isna(sma50.iloc[i]) else None
        dist_sma20 = (price_now - sma20_now) / sma20_now * 100 if sma20_now else None
        dist_sma50 = (price_now - sma50_now) / sma50_now * 100 if sma50_now else None

        vol_recent5 = float(v.iloc[i-5:i].mean()) if i >= 5 else None
        vol_prior5 = float(v.iloc[i-10:i-5].mean()) if i >= 10 else None
        vol_declining = bool(vol_recent5 < vol_prior5) if (vol_recent5 and vol_prior5) else None

        is_hammer = detect_hammer(o, h, l, c, i)
        is_engulf = detect_bullish_engulfing(o, h, l, c, i)
        has_reversal_candle = bool(is_hammer or is_engulf)

        adx_val = float(adx_series.iloc[i]) if not pd.isna(adx_series.iloc[i]) else None

        entry = price_now
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        rows.append({
            "ticker": ticker, "pullback_pct": pullback_pct, "days_since_high": days_since_high,
            "dist_sma20": dist_sma20, "dist_sma50": dist_sma50, "vol_declining": vol_declining,
            "has_reversal_candle": has_reversal_candle, "adx": adx_val,
            "fwd_return": fwd_return, "win": fwd_return > 0,
        })
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_rows = []
    for t in LARGECAP_TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)

    print(f"\n=== 대형주 눌림목 후보 총 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    print("[1. 조정폭(%) 구간별]")
    for lo, hi in [(0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 999)]:
        sub = [r for r in all_rows if lo <= r["pullback_pct"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {lo}~{hi if hi < 999 else '+'}%: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[2. 20일선 대비 이격도(%) 구간별]")
    for lo, hi in [(-999, -8), (-8, -5), (-5, -2), (-2, 0), (0, 999)]:
        sub = [r for r in all_rows if r["dist_sma20"] is not None and lo <= r["dist_sma20"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            lo_l = str(lo) if lo > -999 else "~"
            print(f"  {lo_l}~{hi if hi<999 else '+'}%: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[2. 50일선 대비 이격도(%) 구간별]")
    for lo, hi in [(-999, -8), (-8, -5), (-5, -2), (-2, 0), (0, 999)]:
        sub = [r for r in all_rows if r["dist_sma50"] is not None and lo <= r["dist_sma50"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            lo_l = str(lo) if lo > -999 else "~"
            print(f"  {lo_l}~{hi if hi<999 else '+'}%: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[3. 고점 이후 경과일수 구간별]")
    for lo, hi in [(0, 3), (3, 5), (5, 10), (10, 15), (15, 20), (20, 999)]:
        sub = [r for r in all_rows if lo <= r["days_since_high"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {lo}~{hi if hi<999 else '+'}일: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[4. 거래량 감소 여부]")
    for label, val in [("감소함", True), ("감소안함", False)]:
        sub = [r for r in all_rows if r["vol_declining"] == val]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {label}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[5. 반전캔들 여부]")
    for label, val in [("있음", True), ("없음", False)]:
        sub = [r for r in all_rows if r["has_reversal_candle"] == val]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {label}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[6. ADX 구간별]")
    for lo, hi in [(0, 15), (15, 20), (20, 25), (25, 30), (30, 999)]:
        sub = [r for r in all_rows if r["adx"] is not None and lo <= r["adx"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {lo}~{hi if hi < 999 else '+'}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")


if __name__ == "__main__":
    main()
