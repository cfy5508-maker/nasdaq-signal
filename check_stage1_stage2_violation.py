"""
check_stage1_stage2_violation.py
사용법: python check_stage1_stage2_violation.py

동작: Z-score<=-1.0 다이버전스 매수 후 패배했던 18개 케이스에 대해,
      그 시점(신호 발생일) 기준으로
      1) 나스닥 지수(^IXIC)가 200일선 위/아래였는지 (1단계: 시장국면)
      2) 그 종목 자체가 200일선 위/아래였는지 (참고: 종목 자체 추세)
      를 확인해서, 1단계 위반이 패배의 공통 원인인지 검증한다.
"""
import yfinance as yf
import pandas as pd

CASES = [
    ("ORCL", "2025-04-03", -1.25, -5.94),
    ("ORCL", "2026-01-28", -1.08, -9.05),
    ("TXN", "2024-12-03", -1.13, -4.62),
    ("LOW", "2025-04-03", -1.14, -1.58),
    ("LOW", "2026-05-21", -1.06, -3.07),
    ("CHPT", "2025-01-28", -1.20, -27.19),
    ("CHPT", "2026-03-18", -1.00, -9.81),
    ("AFRM", "2026-02-05", -1.91, -14.02),
    ("AFRM", "2026-02-11", -1.80, -10.82),
    ("AFRM", "2026-02-27", -1.70, -0.21),
    ("AFRM", "2026-03-12", -1.26, -5.45),
    ("NIO", "2025-12-01", -2.01, -3.67),
    ("NIO", "2025-04-03", -1.22, -5.88),
    ("DKNG", "2026-02-12", -1.44, -5.25),
    ("DKNG", "2026-01-29", -1.18, -15.97),
    ("DKNG", "2025-03-18", -1.15, -10.08),
    ("PLUG", "2025-04-14", -1.70, -9.43),
    ("PLUG", "2026-06-24", -1.65, -8.81),
]


def get_sma200_status(hist, date_str):
    """해당 날짜에 종가가 200일선 위/아래인지 반환."""
    idx = hist.index.get_indexer([pd.Timestamp(date_str, tz=hist.index.tz)], method="nearest")[0]
    close = hist["Close"]
    sma200 = close.rolling(200).mean()
    if idx < 200 or pd.isna(sma200.iloc[idx]):
        return None
    return bool(close.iloc[idx] > sma200.iloc[idx])


def main():
    print("나스닥 지수(^IXIC) 데이터 로딩...")
    nasdaq_hist = yf.Ticker("^IXIC").history(period="3y")

    ticker_hist_cache = {}
    print(f"\n=== Z<=-1.0 패배 케이스 18건, 1단계(시장국면)/종목자체 200일선 위반 여부 ===\n")
    stage1_violation_count = 0
    stock_below_count = 0

    for ticker, date_str, zscore, fwd_return in CASES:
        if ticker not in ticker_hist_cache:
            ticker_hist_cache[ticker] = yf.Ticker(ticker).history(period="3y")
        stock_hist = ticker_hist_cache[ticker]

        nasdaq_above = get_sma200_status(nasdaq_hist, date_str)
        stock_above = get_sma200_status(stock_hist, date_str)

        nasdaq_str = "위" if nasdaq_above else ("아래" if nasdaq_above is not None else "N/A")
        stock_str = "위" if stock_above else ("아래" if stock_above is not None else "N/A")

        if nasdaq_above is False:
            stage1_violation_count += 1
        if stock_above is False:
            stock_below_count += 1

        print(f"  {ticker} {date_str} (Z={zscore:.2f}, 수익률{fwd_return:+.2f}%): "
              f"나스닥200일선 {nasdaq_str} | 종목200일선 {stock_str}")

    print(f"\n=== 종합 ===")
    print(f"총 18건 중 나스닥이 200일선 '아래'였던 경우(1단계 위반): {stage1_violation_count}건")
    print(f"총 18건 중 종목 자체가 200일선 '아래'였던 경우: {stock_below_count}건")


if __name__ == "__main__":
    main()
