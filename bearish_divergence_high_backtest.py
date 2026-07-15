"""
bearish_divergence_high_backtest.py
사용법: python bearish_divergence_high_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=60

동작: nasdaq_chart.html의 detectDivergence() 베어리시 판정 로직(6개월 내
최고점 대비)을 개별종목에 그대로 적용해서, 그 다이버전스가 "신저점 없이"
몇 번 연속으로 떴는지 세고, 연속횟수(1~5회+)에 따라 이후 실제로
신저점(그 6개월 구간의 바닥 이탈)을 달성했는지 확인한다.

정의 (nasdaq_chart.html의 detectDivergence 베어리시 분기와 동일 기준):
  - 매일(오늘 포함 이전 거래일 기준) 직전 6개월(126거래일) 구간에서
    최고가(high_p)와 그날의 RSI(high_rsi)를 찾는다.
  - 오늘 종가가 그 최고가의 97% 이상(cur_p >= high_p*0.97)이고,
    오늘 RSI가 그 최고가 시점 RSI보다 3포인트 이상 낮으면(cur_r < high_rsi-3)
    베어리시 다이버전스로 판정 (원본 JS 로직과 동일한 3포인트/97% 기준).
  - 디바운스: 롤링 6개월 창이 하루씩 이동하면서 "최고가 시점" 기준 자체가
    자주 바뀌어(어제 최고가였던 날이 창에서 빠지고 다른 날이 최고가가 되는
    식) 조건이 며칠 단위로 켜졌다 꺼졌다 하는 노이즈가 있다. 그래서 단순
    온오프 전환이 아니라, 한 번 발생을 세면 MIN_EVENT_GAP_DAYS(21거래일,
    약 1개월) 동안은 같은 상황으로 보고 재카운트하지 않는 쿨다운 방식을 쓴다.
  - 신저점 달성: 다이버전스 발생일 이후 FORWARD_DAYS 거래일 이내에,
    그 시점 6개월 구간의 최저가(바닥)를 종가가 하회하면 '달성'으로 판정.
  - 연속횟수(streak): 달성되면(신저점 나와서 사이클 종료) 다음 발생부터
    다시 1회로 리셋. 달성 안 되면(아직 안 무너짐) 다음 발생 때 streak+1로
    이어짐 - 즉 "신저점 없이 몇 번째 다이버전스 반복인지"를 센다.

종목 범위: 관심종목(28개)이 아니라 divergence_only_backtest.py와 동일한
대형주 50 + 소형주 50 = 100종목 유니버스로 확대해서 표본을 늘렸다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LOOKBACK_DAYS = 126          # 6개월 ≈ 21거래일 * 6
NEAR_HIGH_RATIO = 0.97       # nasdaq_chart.html detectDivergence와 동일
RSI_MIN_DECLINE = 3          # nasdaq_chart.html detectDivergence와 동일
MIN_EVENT_GAP_DAYS = 21      # 같은 발생으로 볼 쿨다운 기간(약 1개월)
HISTORY_PERIOD = "15y"       # max로 하면 KO/XOM/PG 등 수십년 종목의 무중단 상승장 하나가
                              # streak를 수백까지 누적시켜 통계를 독식하는 문제가 있어 제한
MIN_HISTORY_DAYS = LOOKBACK_DAYS + 20   # 최소 6개월+여유

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


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD)
    if hist.empty or len(hist) < MIN_HISTORY_DAYS:
        return []
    close = hist["Close"].values
    rsi = RSIIndicator(pd.Series(close), window=14).rsi().values
    n = len(close)

    streak = 0
    last_event_idx = -10 ** 9
    records = []

    for t in range(LOOKBACK_DAYS, n):
        if np.isnan(rsi[t]):
            continue
        window_close = close[t - LOOKBACK_DAYS:t]
        window_rsi = rsi[t - LOOKBACK_DAYS:t]
        if np.all(np.isnan(window_rsi)):
            continue

        high_rel = int(np.nanargmax(window_close))
        high_p = window_close[high_rel]
        high_rsi = window_rsi[high_rel]
        low_p = np.nanmin(window_close)   # 6개월 구간 바닥 - '신저점' 기준선

        is_firing = bool(
            not np.isnan(high_rsi)
            and close[t] >= high_p * NEAR_HIGH_RATIO
            and rsi[t] < high_rsi - RSI_MIN_DECLINE
        )

        if is_firing and (t - last_event_idx) >= MIN_EVENT_GAP_DAYS:
            streak += 1
            future = close[t + 1: t + 1 + FORWARD_DAYS]
            achieved = bool((future < low_p).any())
            days_to = int(np.argmax(future < low_p)) + 1 if achieved else None
            records.append({
                "ticker": ticker, "day_idx": t, "streak": streak,
                "achieved": achieved, "days_to": days_to,
            })
            last_event_idx = t
            if achieved:
                streak = 0   # 신저점으로 사이클 종료 -> 다음 발생은 다시 1회부터

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
    df["streak_bucket"] = df["streak"].apply(lambda s: s if s <= 5 else 6)

    print("\n=== 연속 다이버전스(6개월 최고점 대비) 횟수별 이후 신저점 달성률 (FORWARD_DAYS =", FORWARD_DAYS, ") ===")
    summary = df.groupby("streak_bucket").agg(
        표본수=("achieved", "count"),
        신저점달성률=("achieved", "mean"),
    )
    summary["신저점달성률"] = (summary["신저점달성률"] * 100).round(1)
    achieved_days = df[df["achieved"]].groupby("streak_bucket")["days_to"].mean().round(1)
    summary["평균도달일수"] = achieved_days
    summary.index = summary.index.map(lambda x: f"{x}회" if x < 6 else "6회+")
    print(summary.to_string())

    print(f"\n전체 표본: {len(df)}건 / 전체 신저점 달성률: {df['achieved'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
