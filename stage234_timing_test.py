"""
stage234_timing_test.py
사용법: python stage234_timing_test.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식(2~4단계) 통과 신호에 대해, "다이버전스 확정일로부터 매수 시점까지의
경과일수"를 0~3/3~5/5~7/7~10/10~15/15+일로 나눠서 승률을 비교.
대형주/중소형주 각각 따로 집계해서 최적 타이밍이 다른지 확인.
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
SMALLCAP_TICKERS = [
    "RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
    "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ",
    "FUBO", "PATH", "ASAN", "BBAI", "SOUN", "RUN", "ENVX", "JOBY", "ACHR", "LAZR",
    "QS", "FSLY", "PLUG", "NIO", "XPEV", "BYND", "WOLF", "GPRO", "DNA", "HOOD",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ORDER = 5
MAX_GAP = 60
MIN_GAP = 3
ADX_MIN = 22
RECHECK_WINDOW = 20  # 다이버전스 확정 후 최대 며칠까지 "지연매수"를 검토할지


def find_realtime_lows(c, order=5):
    positions = []
    values = c.values
    n = len(values)
    for i in range(order, n):
        past_window = values[i-order:i]
        if values[i] < past_window.min():
            positions.append(i)
    return positions


def analyze_ticker(ticker, is_large):
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
        if len(rsi_window) < 60:
            continue
        rsi_mean = float(rsi_window.mean())
        rsi_std = float(rsi_window.std())
        if rsi_std <= 0:
            continue
        zscore = (rsi2 - rsi_mean) / rsi_std
        if zscore > -1.0:
            continue

        adx_val = float(adx_series.iloc[pos2]) if not pd.isna(adx_series.iloc[pos2]) else None
        if adx_val is None or adx_val < ADX_MIN:
            continue

        # 다이버전스 확정 시점(pos2) 이후, 매수를 며칠 지연했을 때 승률이 어떻게 되는지
        # 매 지연일마다 그 시점 가격으로 진입했다고 가정
        for delay in range(0, RECHECK_WINDOW + 1):
            buy_pos = pos2 + delay
            if buy_pos + FORWARD_DAYS >= n:
                break
            entry = float(c.iloc[buy_pos])
            fwd_return = float(c.iloc[buy_pos + FORWARD_DAYS]) / entry - 1
            rows.append({"ticker": ticker, "is_large": is_large, "delay": delay,
                         "fwd_return": fwd_return, "win": fwd_return > 0})
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
            rows = analyze_ticker(t, True)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)
    for t in SMALLCAP_TICKERS:
        try:
            rows = analyze_ticker(t, False)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)

    print(f"\n=== 2~4단계 통과 신호, 지연일수별 승률(대형주/중소형주 분리) ===\n")

    bands = [(0, 3, "0~3일"), (3, 5, "3~5일"), (5, 7, "5~7일"), (7, 10, "7~10일"),
             (10, 15, "10~15일"), (15, 21, "15~20일")]

    for label_group, is_large_flag in [("대형주", True), ("중소형주", False)]:
        group = [r for r in all_rows if r["is_large"] == is_large_flag]
        print(f"--- {label_group} ---")
        for lo, hi, blabel in bands:
            sub = [r for r in group if lo <= r["delay"] < hi]
            wr, avg = win_rate(sub)
            if wr is not None:
                print(f"  {blabel}: 표본 {len(sub)}건 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}%")
        print()


if __name__ == "__main__":
    main()
