"""
stage123_adx_test.py
사용법: python stage123_adx_test.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식 (1~3단계):
  게이트: 진짜 스윙저점 기반 불리쉬 다이버전스
  조건1: 저점 간격 3일 이상
  조건2: Z-score (종목 자신의 1년 RSI 평균/표준편차 기준, "개별 계산")
  조건3: ADX 20~25 구간 (다이버전스 시점)

10종목(대형5+중소형5)에 대해 위 게이트+조건1+조건2를 만족하는 신호를
찾고, 그중 조건3(ADX 20~25)까지 만족하는 경우와 아닌 경우의 승률을 비교.
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
MIN_GAP = 3  # 조건1


def find_realtime_lows(c, order=5):
    positions = []
    values = c.values
    n = len(values)
    for i in range(order, n):
        past_window = values[i-order:i]
        if values[i] < past_window.min():
            positions.append(i)
    return positions


def analyze_ticker(ticker):
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
        if gap > MAX_GAP or gap < MIN_GAP:  # 조건1
            continue

        price1, price2 = float(c.iloc[pos1]), float(c.iloc[pos2])
        rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
        if pd.isna(rsi1) or pd.isna(rsi2):
            continue

        is_divergence = bool(price2 < price1 and rsi2 > rsi1)  # 게이트
        if not is_divergence:
            continue

        # 조건2: Z-score, 종목 자신의 1년 RSI 평균/표준편차 (개별 계산)
        rsi_window = rsi.iloc[max(0, pos2-252):pos2+1].dropna()
        zscore = None
        if len(rsi_window) >= 60:
            rsi_mean = float(rsi_window.mean())
            rsi_std = float(rsi_window.std())
            if rsi_std > 0:
                zscore = (rsi2 - rsi_mean) / rsi_std

        if zscore is None or zscore > -1.0:
            continue  # Z-score 조건 미달이면 애초에 후보 아님

        # 조건3: ADX 20~25 구간 (참고용으로 같이 기록, 필터링은 안 하고 비교만)
        adx_val = float(adx_series.iloc[pos2]) if not pd.isna(adx_series.iloc[pos2]) else None
        adx_in_range = bool(adx_val is not None and 20 <= adx_val <= 25)

        if pos2 + FORWARD_DAYS < n:
            entry = float(c.iloc[pos2])
            fwd_return = float(c.iloc[pos2 + FORWARD_DAYS]) / entry - 1
            date2 = str(hist.index[pos2].date())
            rows.append({"ticker": ticker, "date": date2, "zscore": zscore,
                         "adx": adx_val, "adx_in_range": adx_in_range,
                         "fwd_return": fwd_return, "win": fwd_return > 0})
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_rows = []
    print(f"=== 1~2단계(게이트+간격3일+Z<=-1.0) 통과 신호, ADX(조건3) 비교 ===\n")
    for t in TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 ({e})")
            continue
        if rows is None:
            print(f"{t}: 데이터 없음")
            continue
        for r in rows:
            adx_str = f"{r['adx']:.1f}" if r["adx"] is not None else "N/A"
            print(f"  [{r['ticker']}] {r['date']}: Z={r['zscore']:.2f} | ADX={adx_str} "
                  f"({'범위내' if r['adx_in_range'] else '범위밖'}) | {r['fwd_return']*100:+.2f}% | "
                  f"{'승리' if r['win'] else '패배'}")
        all_rows.extend(rows)

    print(f"\n=== 종합 ===")
    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"1~2단계만(조건3 적용 전): 표본 {len(all_rows)}건 | 승률 {baseline_wr:.1f}%\n")

    in_range = [r for r in all_rows if r["adx_in_range"]]
    out_range = [r for r in all_rows if not r["adx_in_range"]]
    wr_in, avg_in = win_rate(in_range)
    wr_out, avg_out = win_rate(out_range)
    if wr_in is not None:
        print(f"조건3(ADX20~25) 충족: 표본 {len(in_range)}건 | 승률 {wr_in:.1f}% | 기준대비 {wr_in-baseline_wr:+.1f}%p")
    if wr_out is not None:
        print(f"조건3(ADX20~25) 미충족: 표본 {len(out_range)}건 | 승률 {wr_out:.1f}% | 기준대비 {wr_out-baseline_wr:+.1f}%p")


if __name__ == "__main__":
    main()
