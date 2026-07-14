"""
deadcat_trend_confirm_backtest.py
사용법: python deadcat_trend_confirm_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 정석 다이버전스(pass)가 확정된 이후, FORWARD_DAYS 동안 단 하루도
      저점2 종가 아래로 다시 떨어지지 않았는지(=진짜 추세전환 성공)를 판정하고,
      "고점(6개월) 이후 경과일수" 구간별로 이 추세전환 성공률을 비교한다.
      (10일 뒤 스냅샷 수익률이 아니라, 그 기간 내내 저점을 지켰는지를 본다)
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
HIGH_LOOKBACK = 126


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

    is_pass_divergence = {}
    for pos2 in realtime_lows:
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

    rolling_high = c.rolling(HIGH_LOOKBACK, min_periods=20).max()

    rows = []
    div_count_since_high = 0
    RECENT_HIGH_LOOKBACK = 20  # 저점2 이전 20일 종가 최고치 = "직전 저항선"
    for i in range(HIGH_LOOKBACK, n - FORWARD_DAYS):
        if not pd.isna(rolling_high.iloc[i]) and close_values[i] >= rolling_high.iloc[i]:
            div_count_since_high = 0
        if i in is_pass_divergence:
            div_count_since_high += 1
            price2 = close_values[i]

            lookback_start = max(0, i - RECENT_HIGH_LOOKBACK)
            resistance = float(close_values[lookback_start:i].max()) if i > lookback_start else price2

            future_window = close_values[i+1:i+1+FORWARD_DAYS]
            new_high_achieved = bool((future_window > resistance).any())  # 직전 저항선을 실제로 돌파했는지

            max_return = float(future_window.max() / price2 - 1) if len(future_window) > 0 else None
            final_return = float(future_window[-1] / price2 - 1) if len(future_window) > 0 else None

            rows.append({"nth": div_count_since_high, "new_high_achieved": new_high_achieved,
                        "max_return": max_return, "final_return": final_return})
    return rows


def new_high_rate(recs):
    if not recs:
        return None
    return float(np.mean([r["new_high_achieved"] for r in recs]) * 100)


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

    print(f"\n=== 정석 다이버전스(pass) 총 {len(all_rows)}건, {FORWARD_DAYS}일 이내 저점유지 여부 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    baseline_rate = new_high_rate(all_rows)
    print(f"전체 기준선(새고점 달성률=직전20일 저항선 돌파): {baseline_rate:.1f}%\n")

    print("=== 고점(6개월) 이후 몇 번째 다이버전스인지별 '새고점 달성률' ===")
    for n_val in [1, 2, 3, 4]:
        if n_val < 4:
            sub = [r for r in all_rows if r["nth"] == n_val]
            label = f"{n_val}번째"
        else:
            sub = [r for r in all_rows if r["nth"] >= n_val]
            label = f"{n_val}번째 이상"
        rate = new_high_rate(sub)
        if rate is not None:
            avg_max = np.mean([r["max_return"] for r in sub]) * 100
            avg_final = np.mean([r["final_return"] for r in sub]) * 100
            print(f"  {label}: 표본{len(sub)}건 | 새고점달성률 {rate:.1f}% | 기준대비 {rate-baseline_rate:+.1f}%p | 기간중최고평균 +{avg_max:.2f}% | 종료시점평균 {avg_final:+.2f}%")


if __name__ == "__main__":
    main()
