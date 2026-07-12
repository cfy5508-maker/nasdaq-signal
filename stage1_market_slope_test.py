"""
stage1_market_slope_test.py
사용법: python stage1_market_slope_test.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식 (1,3,4,5단계):
  게이트(3단계): 진짜 스윙저점 기반 불리쉬 다이버전스, 간격 3일 이상
  조건(4단계): Z-score <= -1.0 (종목별 개별 계산)
  조건(5단계): ADX >= 22
  조건(1단계): 나스닥 200일선 "기울기"(20일 전 대비 상승/하락 중인지)

10종목 신호에 대해 1단계 기울기별 승률을 비교.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

LARGECAP_NEW = ["ORCL", "ADBE", "TXN", "LOW", "GS"]
SMALLCAP_NEW = ["CHPT", "AFRM", "NIO", "DKNG", "PLUG"]
TICKERS = LARGECAP_NEW + SMALLCAP_NEW

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ORDER = 5
MAX_GAP = 60
MIN_GAP = 3
ADX_MIN = 22


def find_realtime_lows(c, order=5):
    positions = []
    values = c.values
    n = len(values)
    for i in range(order, n):
        past_window = values[i-order:i]
        if values[i] < past_window.min():
            positions.append(i)
    return positions


def get_market_slope(market_hist, pos, lookback=20):
    close = market_hist["Close"]
    if pos < 220:
        return None
    sma200_now = close.iloc[pos-199:pos+1].mean()
    sma200_prev = close.iloc[pos-199-lookback:pos+1-lookback].mean()
    return float(sma200_now - sma200_prev)


def analyze_ticker(ticker, market_hist):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 280:
        return None
    o, h, l, c = hist["Open"], hist["High"], hist["Low"], hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    adx_series = ADXIndicator(h, l, c, window=14).adx()
    n = len(hist)

    low_positions = find_realtime_lows(c, order=ORDER)

    rows = []
    for idx in range(len(low_positions) - 1):
        pos1 = low_positions[idx]
        pos2 = low_positions[idx + 1]
        gap = pos2 - pos1
        if gap > MAX_GAP or gap < MIN_GAP:
            continue

        price1, price2 = float(c.iloc[pos1]), float(c.iloc[pos2])
        rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
        if pd.isna(rsi1) or pd.isna(rsi2):
            continue

        is_divergence = bool(price2 < price1 and rsi2 > rsi1)
        if not is_divergence:
            continue

        rsi_window = rsi.iloc[max(0, pos2-252):pos2+1].dropna()
        zscore = None
        if len(rsi_window) >= 60:
            rsi_mean = float(rsi_window.mean())
            rsi_std = float(rsi_window.std())
            if rsi_std > 0:
                zscore = (rsi2 - rsi_mean) / rsi_std
        if zscore is None or zscore > -1.0:
            continue

        adx_val = float(adx_series.iloc[pos2]) if not pd.isna(adx_series.iloc[pos2]) else None
        if adx_val is None or adx_val < ADX_MIN:
            continue

        date2 = str(hist.index[pos2].date())
        market_idx = market_hist.index.get_indexer([hist.index[pos2]], method="nearest")[0]
        slope = get_market_slope(market_hist, market_idx)

        if pos2 + FORWARD_DAYS < n:
            entry = float(c.iloc[pos2])
            fwd_return = float(c.iloc[pos2 + FORWARD_DAYS]) / entry - 1
            rows.append({"ticker": ticker, "date": date2, "zscore": zscore, "adx": adx_val,
                         "slope": slope, "fwd_return": fwd_return, "win": fwd_return > 0})
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    print("나스닥(^IXIC) 데이터 로딩...")
    market_hist = yf.Ticker("^IXIC").history(period="3y")

    all_rows = []
    print(f"\n=== 3~5단계 통과 신호에 1단계(나스닥 200일선 기울기) 결합 ===\n")
    for t in TICKERS:
        try:
            rows = analyze_ticker(t, market_hist)
        except Exception as e:
            print(f"{t}: 에러 ({e})")
            continue
        if rows is None:
            print(f"{t}: 데이터 없음")
            continue
        for r in rows:
            slope_str = f"{r['slope']:+.1f}" if r["slope"] is not None else "N/A"
            print(f"  [{r['ticker']}] {r['date']}: Z={r['zscore']:.2f} ADX={r['adx']:.1f} | "
                  f"시장기울기{slope_str} | {r['fwd_return']*100:+.2f}% | {'승리' if r['win'] else '패배'}")
        all_rows.extend(rows)

    print(f"\n=== 종합 ===")
    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"3~5단계까지: 표본 {len(all_rows)}건 | 승률 {baseline_wr:.1f}%\n")

    up = [r for r in all_rows if r["slope"] is not None and r["slope"] > 0]
    down = [r for r in all_rows if r["slope"] is not None and r["slope"] <= 0]

    wr_up, avg_up = win_rate(up)
    wr_down, avg_down = win_rate(down)
    if wr_up is not None:
        print(f"시장 200일선 상승중(기울기>0): 표본 {len(up)}건 | 승률 {wr_up:.1f}% | 기준대비 {wr_up-baseline_wr:+.1f}%p")
    if wr_down is not None:
        print(f"시장 200일선 하락중(기울기<=0): 표본 {len(down)}건 | 승률 {wr_down:.1f}% | 기준대비 {wr_down-baseline_wr:+.1f}%p")


if __name__ == "__main__":
    main()
