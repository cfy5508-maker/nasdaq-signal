"""
zscore_bucket_backtest.py
사용법: python zscore_bucket_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=60

배경: fetch_indicators.py 4단계는 Z-score = (오늘RSI - 1년평균RSI) / 1년표준편차를
계산해서, Z가 낮을수록(-1.5 이하) 좋은 점수를 준다. 근데 "떨어지는 칼날" 가설:
극단적으로 낮은 Z-score(-2 이하)는 사실 아직 하락이 안 끝난 폭락 중일 수 있고,
오히려 적당히 회복된 중간 구간(-1.5~-0.5)이 "하락 동력이 죽고 반등이 시작되는
진짜 바닥"일 수 있다는 가설을 검증한다.

동작: 3단계 다이버전스(정석+이중바닥 통합) 발생 시점(저점2)마다, 그 시점 기준
"직전 252거래일(운영 코드의 1년 근사) RSI 평균/표준편차"로 Z-score를 계산하고,
구간별(< -2 / -2~-1.5 / -1.5~-1 / -1~-0.5 / -0.5~0 / 0+)로 신저점(=실패)
달성률을 비교한다.

  - 단조감소(왼쪽 갈수록 계속 좋아짐) → 지금 방식(낮을수록 좋다)이 맞음
  - U자형(중간이 제일 좋고 양 끝이 나쁨) → "적당한 Z-score가 진짜 바닥" 가설 지지

저점 탐지는 confirmed 방식(양방향 확정 - 미확정 저점 중복집계 문제 회피, 오늘
백테스트에서 이미 검증된 더 신뢰할 수 있는 방식)을 쓴다.

신저점(실패) 판정: 다이버전스 확정일(low2) 이후 FORWARD_DAYS 거래일 이내에,
가격이 price2보다 낮은 저점을 찍고 그 저점 대비 3% 이상 반등까지 확인되면 '달성'.

종목 범위: 대형주 134 + 소형주 98 = 232종목(기존 백테스트들과 동일 유니버스).
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60

ORDER = 5
MIN_GAP_DAYS = 5
MAX_GAP_DAYS = 30
DOUBLE_BOTTOM_MAX_PREMIUM_PCT = 0.10
REBOUND_CONFIRM_PCT = 0.03
RSI_ZSCORE_WINDOW = 252     # 운영 코드의 "1년" 근사
RSI_ZSCORE_MIN_SAMPLE = 30  # 운영 코드와 동일한 최소 표본 기준
HISTORY_PERIOD = "15y"

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
    "NKE", "PM", "MO", "CL", "KHC", "GILD", "AMGN", "BMY", "LLY", "ABT",
    "DHR", "HON", "CAT", "DE", "MMM", "RTX", "LMT", "NOC", "GD", "FDX",
    "UNP", "NSC", "CMCSA", "TMUS", "CHTR", "EA", "ADP", "INTU", "NOW", "PANW",
    "CRWD", "SNPS", "CDNS", "MU", "AMAT", "LRCX", "KLAC", "ASML", "SHW", "APD",
    "ECL", "DUK", "SO", "D", "AEP", "EXC", "PEG", "WM", "RSG", "PSA",
    "AMT", "PLD", "EQIX", "SPG", "O", "CCI", "SCHW", "AXP", "BLK", "SPGI",
    "MCO", "ICE", "CME", "USB", "PNC", "TFC", "COF", "MET", "PRU", "AIG",
    "TRV", "ALL", "HUM", "CI", "ELV", "CVS", "MDT", "SYK", "BSX", "ISRG",
    "VRTX", "REGN", "ZTS", "BDX",
]
SMALLCAP_TICKERS = [
    "RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
    "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ",
    "FUBO", "PATH", "ASAN", "BBAI", "SOUN", "RUN", "ENVX", "JOBY", "ACHR", "LAZR",
    "QS", "FSLY", "PLUG", "NIO", "XPEV", "BYND", "WOLF", "GPRO", "DNA", "HOOD",
    "AI", "PTON", "SNAP", "LYFT", "DASH", "PINS", "ETSY", "BILL", "TTD", "ROKU",
    "DOCU", "ZM", "TWLO", "OKTA", "DDOG", "NET", "ZS", "MDB", "TEAM", "WDAY",
    "VEEV", "HUBS", "ESTC", "PCTY", "PAYC", "APPF", "SQ", "SHOP", "W", "CHWY",
    "ABNB", "UBER", "MTCH", "BMBL", "YELP", "GRPN", "Z", "ZG", "RDFN", "COMP",
    "CPNG", "SE", "MELI", "JD", "PDD", "BABA", "NTES", "BIDU", "CARG", "VRM",
    "AN", "KMX", "WBD", "PARA", "LYV", "SIRI", "SPOT", "PENN",
]
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS


def find_confirmed_lows(values, order=5):
    """양방향 확정(전후 order일 모두보다 낮음) - 백테스트라 미래를 볼 수 있어서 가능."""
    positions = []
    n = len(values)
    for i in range(order, n - order):
        past_window = values[i - order:i]
        future_window = values[i + 1:i + 1 + order]
        if values[i] < past_window.min() and values[i] < future_window.min():
            positions.append(i)
    return positions


def bucket_zscore(z):
    if z < -2: return "<-2"
    if z < -1.5: return "-2~-1.5"
    if z < -1: return "-1.5~-1"
    if z < -0.5: return "-1~-0.5"
    if z < 0: return "-0.5~0"
    return "0+"


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD)
    if hist.empty or len(hist) < 60:
        return []
    close = hist["Close"].values
    rsi = RSIIndicator(pd.Series(close), window=14).rsi().values

    swing_lows = find_confirmed_lows(close, ORDER)

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
            is_classic = (rsi2 > rsi1) and (price2 < price1)
            is_double_bottom = (rsi2 > rsi1) and (price2 >= price1) and (price2 <= price1 * (1 + DOUBLE_BOTTOM_MAX_PREMIUM_PCT))
            if is_classic or is_double_bottom:
                # Z-score: 저점2(idx) 시점 기준 직전 252거래일 RSI 평균/표준편차로 계산
                win_start = max(0, idx - RSI_ZSCORE_WINDOW + 1)
                rsi_window = rsi[win_start:idx + 1]
                rsi_window_valid = rsi_window[~np.isnan(rsi_window)]
                zscore = None
                if len(rsi_window_valid) >= RSI_ZSCORE_MIN_SAMPLE:
                    rsi_mean = rsi_window_valid.mean()
                    rsi_std = rsi_window_valid.std()
                    if rsi_std > 0:
                        zscore = (rsi2 - rsi_mean) / rsi_std

                if zscore is None:
                    last_low_idx, last_low_price, last_low_rsi = idx, price2, rsi2
                    continue

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
                    "type": "classic" if is_classic else "double_bottom",
                    "zscore": zscore,
                    "achieved": achieved,
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
    df["bucket"] = df["zscore"].apply(bucket_zscore)
    order = ["<-2", "-2~-1.5", "-1.5~-1", "-1~-0.5", "-0.5~0", "0+"]

    print(f"\n=== Z-score 구간별 신저점 달성률 (confirmed 저점, FORWARD_DAYS={FORWARD_DAYS}) ===")
    summary = df.groupby("bucket").agg(
        표본수=("achieved", "count"),
        신저점달성률=("achieved", "mean"),
    )
    summary["신저점달성률"] = (summary["신저점달성률"] * 100).round(1)
    summary = summary.reindex(order).dropna(how="all")
    print(summary.to_string())
    print(f"\n전체 표본: {len(df)}건 / 전체 신저점달성률: {df['achieved'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
