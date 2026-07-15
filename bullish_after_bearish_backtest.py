"""
bullish_after_bearish_backtest.py
사용법: python bullish_after_bearish_backtest.py [FORWARD_DAYS] [RECENT_DAYS]
기본값: FORWARD_DAYS=60, RECENT_DAYS=30

동작: 정석 볼리시 다이버전스(신규진입 3단계 게이트와 동일 정의: 스윙저점
price2<=price1, rsi2>rsi1, 간격 5~30일) 발생 시점 기준으로, 그 직전
RECENT_DAYS 거래일 이내에 베어리시 다이버전스(6개월 최고점 대비,
nasdaq_chart.html의 detectDivergence 베어리시 판정과 동일: 종가가 6개월
최고가의 97% 이상 + RSI는 그 고점 시점보다 3포인트 이상 낮음)가 한 번이라도
떴었는지에 따라 두 그룹으로 나누고, 각 그룹의 이후 실패율(=신저점 달성률,
bullish_divergence_low_backtest.py와 동일하게 "저점 찍고 3% 반등 확인"
기준)을 비교한다.

가설: 최근에 베어리시 다이버전스가 있었다면, 그 뒤에 뜨는 볼리시 다이버전스는
신뢰도가 낮다(=실패율이 더 높다). 맞다면 fetch_indicators.py 3단계에서
"최근 베어리시 발생 시 볼리시 다이버전스 무효화/감점" 로직의 근거가 된다.

종목 범위: 대형주 50 + 소형주 50 = 100종목(기존 백테스트들과 동일 유니버스).
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
RECENT_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# 볼리시 게이트 (fetch_indicators.py 3단계와 동일)
ORDER = 5
MIN_GAP_DAYS = 5
MAX_GAP_DAYS = 30
REBOUND_CONFIRM_PCT = 0.03   # 저점 확정에 필요한 최소 반등폭(운영 코드 TOLERANCE_PCT와 동일)

# 베어리시 판정 (nasdaq_chart.html detectDivergence와 동일)
BEAR_LOOKBACK_DAYS = 126     # 6개월
BEAR_NEAR_HIGH_RATIO = 0.97
BEAR_RSI_MIN_DECLINE = 3

HISTORY_PERIOD = "15y"
MIN_HISTORY_DAYS = BEAR_LOOKBACK_DAYS + 40

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
]
SMALLCAP_TICKERS = [
    "RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
    "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ",
    "FUBO", "PATH", "ASAN", "BBAI", "SOUN", "RUN", "ENVX", "JOBY", "ACHR", "LAZR",
    "QS", "FSLY", "PLUG", "NIO", "XPEV", "BYND", "WOLF", "GPRO", "DNA", "HOOD",
    "AI", "PTON", "SNAP", "LYFT", "DASH", "PINS", "ETSY", "BILL", "TTD", "ROKU",
]
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS


def find_realtime_lows(values, order=5):
    positions = []
    n = len(values)
    for i in range(order, n):
        window = values[i - order:i]
        if values[i] < window.min():
            positions.append(i)
    return positions


def compute_bearish_firing(close, rsi):
    """매일(day t)에 대해, 직전 6개월 최고점 대비 베어리시 다이버전스가
    떠 있는 상태인지(True/False) 배열로 반환."""
    n = len(close)
    firing = np.zeros(n, dtype=bool)
    for t in range(BEAR_LOOKBACK_DAYS, n):
        if np.isnan(rsi[t]):
            continue
        window_close = close[t - BEAR_LOOKBACK_DAYS:t]
        window_rsi = rsi[t - BEAR_LOOKBACK_DAYS:t]
        if np.all(np.isnan(window_rsi)):
            continue
        high_rel = int(np.nanargmax(window_close))
        high_p = window_close[high_rel]
        high_rsi = window_rsi[high_rel]
        if (not np.isnan(high_rsi)
                and close[t] >= high_p * BEAR_NEAR_HIGH_RATIO
                and rsi[t] < high_rsi - BEAR_RSI_MIN_DECLINE):
            firing[t] = True
    return firing


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD)
    if hist.empty or len(hist) < MIN_HISTORY_DAYS:
        return []
    close = hist["Close"].values
    rsi = RSIIndicator(pd.Series(close), window=14).rsi().values

    bearish_firing = compute_bearish_firing(close, rsi)
    swing_lows = find_realtime_lows(close, ORDER)

    last_low_idx = None
    last_low_price = None
    last_low_rsi = None
    records = []

    for idx in swing_lows:
        price2, rsi2 = close[idx], rsi[idx]
        if np.isnan(rsi2):
            continue

        if (last_low_idx is not None
                and MIN_GAP_DAYS <= (idx - last_low_idx) <= MAX_GAP_DAYS
                and not np.isnan(last_low_rsi)):
            price1, rsi1 = last_low_price, last_low_rsi
            is_classic_divergence = (price2 <= price1) and (rsi2 > rsi1)
            if is_classic_divergence:
                recent_bearish = bool(bearish_firing[max(0, idx - RECENT_DAYS):idx].any())

                future_close = close[idx + 1: idx + 1 + FORWARD_DAYS]
                achieved = False
                if len(future_close) > 0:
                    below = future_close < price2
                    if below.any():
                        trough_rel = int(np.argmin(future_close))
                        trough_price = future_close[trough_rel]
                        if trough_price < price2:
                            after_trough = future_close[trough_rel + 1:]
                            if len(after_trough) > 0:
                                rebound_pct = (after_trough.max() - trough_price) / trough_price
                                if rebound_pct >= REBOUND_CONFIRM_PCT:
                                    achieved = True

                records.append({
                    "ticker": ticker, "low_idx": idx,
                    "recent_bearish": recent_bearish, "achieved": achieved,
                })
        last_low_idx, last_low_price, last_low_rsi = idx, price2, rsi2

    return records


def main():
    all_records = []
    for t in TICKERS:
        try:
            recs = analyze_ticker(t)
            all_records.extend(recs)
            print(f"{t}: 다이버전스 {len(recs)}건")
        except Exception as e:
            print(f"{t}: 실패 ({e})")

    if not all_records:
        print("표본 없음")
        return

    df = pd.DataFrame(all_records)

    print(f"\n=== 직전 {RECENT_DAYS}거래일 내 베어리시 다이버전스 유무별, 이후 볼리시 다이버전스 신저점(실패) 달성률 (FORWARD_DAYS = {FORWARD_DAYS}) ===")
    summary = df.groupby("recent_bearish").agg(
        표본수=("achieved", "count"),
        신저점달성률=("achieved", "mean"),
    )
    summary["신저점달성률"] = (summary["신저점달성률"] * 100).round(1)
    summary.index = summary.index.map(lambda x: "최근 베어리시 있음" if x else "최근 베어리시 없음")
    print(summary.to_string())

    print(f"\n전체 표본: {len(df)}건")


if __name__ == "__main__":
    main()
