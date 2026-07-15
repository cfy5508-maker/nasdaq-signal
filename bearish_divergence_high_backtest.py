"""
bullish_divergence_low_backtest.py
사용법: python bullish_divergence_low_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=60

동작: 대형주 50 + 소형주 50 = 100종목(bearish_divergence_high_backtest.py와
동일 유니버스, 관심종목 28개로는 표본이 너무 작아 확대)에서, fetch_indicators.py
3단계 게이트가 쓰는 "정석 볼리시 다이버전스"(가격 lower low + RSI higher low)가
몇 번 연속으로(그 사이 신저점 없이) 발생했는지 세고, 그 연속횟수(1~5회+)에
따라 다음 FORWARD_DAYS 안에 신저점(가격이 이번 다이버전스 저점보다 더
낮아지는 것)을 달성했는지 확인한다.

정의 (fetch_indicators.py 3단계 게이트와 동일):
  - 스윙저점: "오늘 종가가 직전 order일 종가의 최솟값보다 낮다"
    (find_realtime_lows와 동일, 미래 미참조)
  - 정석 다이버전스: 연속된 두 스윙저점 사이 간격 MIN_GAP~MAX_GAP일(5~30일,
    운영 코드와 동일), price2 <= price1(저점 갱신) AND rsi2 > rsi1(RSI는
    개선) - 운영 코드처럼 RSI 개선폭에 최소 임계값을 두지 않는다(그냥
    rsi2>rsi1이면 인정).
  - 신저점 달성: 다이버전스 확정일(low2) 이후 FORWARD_DAYS 거래일 이내에,
    종가가 price2보다 낮은 저점을 찍고, 그 저점 이후 그 저점가 대비
    REBOUND_CONFIRM_PCT(3%) 이상 반등하는 지점까지 나오면 '달성'으로
    판정한다. 그냥 살짝 밀린 것(작은 눌림)이나, 저점을 찍었어도 아직
    반등 없이 계속 흘러내리는 중인 것("웅덩이"가 아직 안 파인 상태)은
    카운트하지 않는다 - 저점이 찍히고 반등까지 확인돼야 "웅덩이가
    파였다(=신저점이 확정됐다)"고 본다.
  - 연속횟수(streak): 신저점이 달성되면 다음 다이버전스는 streak+1로
    이어짐(=같은 하락 사이클 안에서 몇 번째 시도인지). 신저점 없이
    반등이 유지되면(=성공) streak는 다음 사이클을 위해 0으로 리셋.

주의: 탐색적 백테스트다. MIN_GAP/MAX_GAP/FORWARD_DAYS는 운영 코드값을
그대로 썼지만, "신저점 달성률이 연속횟수에 따라 어떻게 변하는지"는 아직
검증된 적 없는 새로운 질문이다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
ORDER = 5
MIN_GAP_DAYS = 5           # fetch_indicators.py 운영값과 동일
MAX_GAP_DAYS = 30          # fetch_indicators.py 운영값과 동일
REBOUND_CONFIRM_PCT = 0.03  # 저점 확정에 필요한 최소 반등폭(3%) - 운영 코드의
                             # TOLERANCE_PCT("이미 확실히 반등했다" 판정 기준)와 동일
HISTORY_PERIOD = "15y"     # max로 하면 장기 무중단 상승장 하나가 표본을 독식하는 문제가 있어 제한

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
    """fetch_indicators.py의 find_realtime_lows와 동일(미래 미참조)."""
    positions = []
    n = len(values)
    for i in range(order, n):
        window = values[i - order:i]
        if values[i] < window.min():
            positions.append(i)
    return positions


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD)
    if hist.empty or len(hist) < 40:   # RSI(14)+스윙쌍 1개 나올 최소치로 완화(기존 100)
        return []
    close = hist["Close"].values
    rsi = RSIIndicator(pd.Series(close), window=14).rsi().values

    swing_lows = find_realtime_lows(close, ORDER)

    streak = 0
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
                streak += 1
                # 신저점 달성 판정: 그냥 price2보다 더 빠지기만 하면 되는 게 아니라,
                # "저점을 찍고 그 이후 유의미하게(3%, 운영 코드 TOLERANCE_PCT와 동일
                # 기준) 반등"까지 나와야 진짜 웅덩이(=확정된 신저점)로 본다.
                # 계속 흘러내리기만 하고 반등이 안 나온 구간은 아직 "저점 확정"이
                # 아니라서 카운트하지 않는다.
                future_close = close[idx + 1: idx + 1 + FORWARD_DAYS]
                achieved = False
                days_to = None
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
                                    days_to = trough_rel + 1
                records.append({
                    "ticker": ticker, "low_idx": idx, "streak": streak,
                    "achieved": achieved, "days_to": days_to,
                })
                if not achieved:
                    streak = 0   # 신저점 없이 반등 유지(성공) -> 다음 사이클을 위해 리셋
            else:
                streak = 0
        last_low_idx, last_low_price, last_low_rsi = idx, price2, rsi2

    return records


def main():
    tickers = TICKERS
    all_records = []
    for t in tickers:
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
    df["streak_bucket"] = df["streak"].apply(lambda s: s if s <= 5 else 6)

    print("\n=== 연속 다이버전스 횟수별 이후 신저점 달성률 (FORWARD_DAYS =", FORWARD_DAYS, ") ===")
    summary = df.groupby("streak_bucket").agg(
        표본수=("achieved", "count"),
        신저점달성률=("achieved", "mean"),
    )
    summary["신저점달성률"] = (summary["신저점달성률"] * 100).round(1)
    achieved_days = df[df["achieved"]].groupby("streak_bucket")["days_to"].mean().round(1)
    summary["평균도달일수"] = achieved_days
    summary.index = summary.index.map(lambda x: f"{x}회" if x < 6 else "6회+")
    print(summary.to_string())

    print(f"\n전체 표본: {len(df)}건 / 전체 신저점달성률: {df['achieved'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
