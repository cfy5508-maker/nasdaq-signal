"""
adx_direction_backtest.py
사용법: python adx_direction_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식(2~4단계 게이트 통과 신호에 대해):
  게이트: 진짜 스윙저점 다이버전스(간격3~30일, 오차범위 포함)
  ADX 절대값과 별개로, "5일 전 대비 ADX가 오르는 중인지"를 추가로 확인.
  특히 ADX<22(현재 fail 구간)를 "회복중"과 "계속하락"으로 나눠
  회복중일 때 승률이 실제로 더 좋은지 검증.
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
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MIN_GAP_DAYS = 3
MAX_GAP_DAYS = 30
TOLERANCE_PCT = 0.03


def find_realtime_lows(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n):
        past_window = close_values[i - order:i]
        if close_values[i] < past_window.min():
            positions.append(i)
    return positions


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 280:
        return None
    h, l, c = hist["High"], hist["Low"], hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    adx_series = ADXIndicator(h, l, c, window=14).adx()
    close_values = c.values
    n = len(hist)

    realtime_lows_full = find_realtime_lows(close_values, order=5)

    rows = []
    # 매일(i) 시점에서, 그 시점까지의 데이터만으로 저점을 판정(과거 재현)
    lows_upto = []
    li = 0
    for i in range(260, n - FORWARD_DAYS):
        while li < len(realtime_lows_full) and realtime_lows_full[li] <= i:
            lows_upto.append(realtime_lows_full[li])
            li += 1
        if len(lows_upto) < 2:
            continue
        pos2 = lows_upto[-1]
        candidates = [p for p in lows_upto if p != pos2 and MIN_GAP_DAYS <= (pos2 - p) <= MAX_GAP_DAYS]
        if not candidates:
            continue
        pos1 = min(candidates, key=lambda p: close_values[p])
        is_today_new_low = (pos2 == i)
        price2 = float(c.iloc[pos2])
        price_today = float(c.iloc[i])
        within_tol = bool(price_today >= price2 and (price_today - price2) / price2 <= TOLERANCE_PCT)
        if not (is_today_new_low or within_tol):
            continue

        price1 = float(c.iloc[pos1])
        rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
        if pd.isna(rsi1) or pd.isna(rsi2):
            continue
        if not (price2 < price1 and rsi2 > rsi1):
            continue  # pass 케이스만(정석 다이버전스) 대상으로 ADX 방향 검증

        adx_today = float(adx_series.iloc[i]) if not pd.isna(adx_series.iloc[i]) else None
        adx_5ago = float(adx_series.iloc[i - 5]) if i >= 5 and not pd.isna(adx_series.iloc[i - 5]) else None
        if adx_today is None or adx_5ago is None:
            continue
        adx_rising = adx_today > adx_5ago

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        rows.append({"adx": adx_today, "adx_rising": adx_rising, "fwd_return": fwd_return, "win": fwd_return > 0})
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

    print(f"\n=== 정석 다이버전스(PASS) 신호 총 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%\n")

    low_adx = [r for r in all_rows if r["adx"] < 22]
    print(f"=== ADX<22 구간(현재 fail 처리) 세부 분석: 표본 {len(low_adx)}건 ===")
    wr, avg = win_rate(low_adx)
    if wr is not None:
        print(f"전체: 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p\n")

    rising = [r for r in low_adx if r["adx_rising"]]
    falling = [r for r in low_adx if not r["adx_rising"]]
    wr_r, avg_r = win_rate(rising)
    wr_f, avg_f = win_rate(falling)
    if wr_r is not None:
        print(f"  ADX 회복중(5일전보다 상승): 표본 {len(rising)}건 | 승률 {wr_r:.1f}% | 기준대비 {wr_r-baseline_wr:+.1f}%p")
    if wr_f is not None:
        print(f"  ADX 계속하락(5일전보다 하락): 표본 {len(falling)}건 | 승률 {wr_f:.1f}% | 기준대비 {wr_f-baseline_wr:+.1f}%p")

    print(f"\n=== 참고: ADX>=22 구간(현재 pass/warn) ===")
    high_adx = [r for r in all_rows if r["adx"] >= 22]
    wr_h, avg_h = win_rate(high_adx)
    if wr_h is not None:
        print(f"전체: 표본 {len(high_adx)}건 | 승률 {wr_h:.1f}% | 기준대비 {wr_h-baseline_wr:+.1f}%p")


if __name__ == "__main__":
    main()
