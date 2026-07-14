"""
divergence_since_high_backtest.py
사용법: python divergence_since_high_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 각 종목의 "6개월(126거래일) 고점" 이후, 정석 다이버전스(pass)가
      몇 번째로 발생했는지(1번째/2번째/3번째/4번째이상) 카운트하고,
      그 순번별로 이후 승률이 어떻게 달라지는지 확인한다.
      (가설: 고점 이후 다이버전스가 반복될수록 신뢰도가 떨어질 것)
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

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
]
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MIN_GAP_DAYS = 5
MAX_GAP_DAYS = 30
HIGH_LOOKBACK = 126  # 약 6개월


def find_realtime_lows(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n):
        past_window = close_values[i - order:i]
        if close_values[i] < past_window.min():
            positions.append(i)
    return positions


def calc_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 200:
        return None
    c = hist["Close"]
    close_values = c.values
    rsi = calc_rsi(c)
    n = len(hist)

    realtime_lows = find_realtime_lows(close_values, order=5)
    low_set = set(realtime_lows)

    # 각 저점(pos2)이 정석 다이버전스(pass)였는지 사전 계산
    is_pass_divergence = {}
    for idx, pos2 in enumerate(realtime_lows):
        candidates = [p for p in realtime_lows if p != pos2 and MIN_GAP_DAYS <= (pos2 - p) <= MAX_GAP_DAYS]
        if not candidates:
            continue
        pos1 = min(candidates, key=lambda p: close_values[p])
        price1, price2 = close_values[pos1], close_values[pos2]
        rsi1, rsi2 = rsi.iloc[pos1], rsi.iloc[pos2]
        if pd.isna(rsi1) or pd.isna(rsi2):
            continue
        if price2 < price1 and rsi2 > rsi1:
            is_pass_divergence[pos2] = True

    # 126일 롤링 최고가 - 오늘이 그 최고가를 새로 찍은 날인지 확인(causal)
    rolling_high = c.rolling(HIGH_LOOKBACK, min_periods=20).max()

    rows = []
    div_count_since_high = 0
    for i in range(HIGH_LOOKBACK, n - FORWARD_DAYS):
        # 오늘이 새로운 126일 고점이면 카운터 리셋
        if not pd.isna(rolling_high.iloc[i]) and close_values[i] >= rolling_high.iloc[i]:
            div_count_since_high = 0
        if i in is_pass_divergence:
            div_count_since_high += 1
            entry = float(c.iloc[i])
            fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1
            rows.append({"nth": div_count_since_high, "fwd_return": fwd_return, "win": fwd_return > 0})
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

    print(f"\n=== 정석 다이버전스(pass) 총 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%\n")

    print("=== 6개월 고점 이후 몇 번째 다이버전스인지 별 승률 ===")
    for n_val in [1, 2, 3, 4]:
        if n_val < 4:
            sub = [r for r in all_rows if r["nth"] == n_val]
            label = f"{n_val}번째"
        else:
            sub = [r for r in all_rows if r["nth"] >= n_val]
            label = f"{n_val}번째 이상"
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {label}: 표본{len(sub)}건 | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p | 평균수익률 {avg:+.2f}%")


if __name__ == "__main__":
    main()
