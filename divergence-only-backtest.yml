"""
bearish_divergence_high_backtest.py
사용법: python bearish_divergence_high_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=60

동작: 관심종목(watchlist.json) 전체에서, 스윙고점(전고점) 대비 베어리시
다이버전스(가격은 higher/equal high, RSI는 lower high)가 "신저점 없이"
연속으로 몇 번 발생했는지 세고, 그 연속횟수(1회/2회/3회+)에 따라 이후
실제로 신저점(직전 스윙저점 이탈)을 달성했는지 여부를 확인한다.

정의 (신규진입 로직의 스윙저점 탐지 방식을 고점에 대칭 적용):
  - 스윙고점: "오늘 종가가 직전 order일 종가의 최댓값보다 높다" (미래를 보지
    않는 실시간 탐지 방식. find_realtime_lows와 완전히 대칭)
  - 스윙저점: 반대로 최솟값보다 낮은 지점 (fetch_indicators.py의
    find_realtime_lows와 동일 정의)
  - 베어리시 다이버전스: 연속된 두 스윙고점 사이 간격 MIN_GAP~MAX_GAP일,
    price2 >= price1 (고점 갱신 또는 유지) AND rsi1 - rsi2 >= RSI_MIN_DECLINE
    (RSI는 약해짐). 임계값 3포인트는 신규진입 히든다이버전스 판정과 동일 기준.
  - 연속횟수(streak): 직전 스윙저점 이탈(=신저점) 이후 처음 뜨면 1회, 신저점
    없이 다음 고점에서 다이버전스가 또 뜨면 2회... 로 누적. 신저점이 뜨거나
    다이버전스 조건이 깨지면(RSI가 같이 개선) streak 리셋.
  - 신저점 달성 판정: 다이버전스 확정일(두 번째 고점) 이후 FORWARD_DAYS
    거래일 이내에, 이 상승구간의 시작점이었던 직전 스윙저점 종가를
    하회하면 '달성'으로 판정.

주의: 이 스크립트는 탐색적 백테스트다. MIN_GAP/MAX_GAP/RSI_MIN_DECLINE/
FORWARD_DAYS 전부 조정 가능한 가정값이며, 아직 다른 다이버전스 백테스트들처럼
임계값 스윕 검증을 거치지 않았다.
"""
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
ORDER = 5
MIN_GAP_DAYS = 5
MAX_GAP_DAYS = 60          # 고점끼리는 저점보다 간격이 넓게 벌어지는 경우가 많아 30->60으로 완화
RSI_MIN_DECLINE = 3        # 신규진입 히든다이버전스 임계값과 동일
HISTORY_PERIOD = "3y"

WATCHLIST_PATH = "data/watchlist.json"


def load_watchlist():
    try:
        with open(WATCHLIST_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        # 리포 루트가 아닌 곳에서 실행할 경우 대비한 폴백
        return ["TOYO", "PFE", "STZ", "AMSC", "RKLB", "AVEX", "NXT", "SPCX", "LMT",
                "SMCI", "SIF", "LASR", "PEP", "TTWO", "VOYG", "INFQ", "FSLR", "IREN",
                "UUUU", "PL", "NFLX", "BIIB", "PGR", "WATT", "GOOGL", "IBM", "TER", "QNT"]


def find_realtime_extrema(values, order=5, mode="low"):
    """fetch_indicators.py의 find_realtime_lows와 동일 방식(미래 미참조).
    mode='low'면 저점, mode='high'면 고점 후보 인덱스 리스트를 반환한다."""
    positions = []
    n = len(values)
    for i in range(order, n):
        window = values[i - order:i]
        if mode == "low" and values[i] < window.min():
            positions.append(i)
        elif mode == "high" and values[i] > window.max():
            positions.append(i)
    return positions


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD)
    if hist.empty or len(hist) < 100:
        return []
    close = hist["Close"].values
    rsi = RSIIndicator(pd.Series(close), window=14).rsi().values

    swing_highs = find_realtime_extrema(close, ORDER, "high")
    swing_lows = find_realtime_extrema(close, ORDER, "low")

    events = sorted([(i, "high") for i in swing_highs] + [(i, "low") for i in swing_lows])

    streak = 0
    anchor_low_idx = None      # 현재 상승구간을 떠받치는 직전 스윙저점
    last_high_idx = None
    last_high_price = None
    last_high_rsi = None
    records = []

    for idx, kind in events:
        if kind == "low":
            anchor_low_idx = idx
            streak = 0
            last_high_idx = None
            last_high_price = None
            last_high_rsi = None
            continue

        price_h, rsi_h = close[idx], rsi[idx]
        if np.isnan(rsi_h):
            continue

        if (last_high_idx is not None
                and MIN_GAP_DAYS <= (idx - last_high_idx) <= MAX_GAP_DAYS
                and not np.isnan(last_high_rsi)):
            is_bearish_div = (price_h >= last_high_price) and ((last_high_rsi - rsi_h) >= RSI_MIN_DECLINE)
            if is_bearish_div:
                streak += 1
                ref_low_price = close[anchor_low_idx] if anchor_low_idx is not None else close[max(0, idx - 60):idx].min()
                future = close[idx + 1: idx + 1 + FORWARD_DAYS]
                achieved = bool((future < ref_low_price).any())
                days_to = int(np.argmax(future < ref_low_price)) + 1 if achieved else None
                records.append({
                    "ticker": ticker, "high_idx": idx, "streak": streak,
                    "achieved": achieved, "days_to": days_to,
                })
            else:
                streak = 0
        last_high_idx, last_high_price, last_high_rsi = idx, price_h, rsi_h

    return records


def main():
    tickers = load_watchlist()
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
    df["streak_bucket"] = df["streak"].apply(lambda s: s if s <= 3 else 4)

    print("\n=== 연속 다이버전스 횟수별 이후 신저점 달성률 (FORWARD_DAYS =", FORWARD_DAYS, ") ===")
    summary = df.groupby("streak_bucket").agg(
        표본수=("achieved", "count"),
        신저점달성률=("achieved", "mean"),
    )
    summary["신저점달성률"] = (summary["신저점달성률"] * 100).round(1)
    achieved_days = df[df["achieved"]].groupby("streak_bucket")["days_to"].mean().round(1)
    summary["평균도달일수"] = achieved_days
    summary.index = summary.index.map(lambda x: f"{x}회" if x < 4 else "4회+")
    print(summary.to_string())

    print(f"\n전체 표본: {len(df)}건 / 전체 신저점 달성률: {df['achieved'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
