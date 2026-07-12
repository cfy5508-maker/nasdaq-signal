"""
check_sma200_full43.py
사용법: python check_sma200_full43.py

동작: Z-score<=-1.0인 다이버전스 신호 43건(승리25+패배18) 전체에 대해,
      그 시점 종목 자체가 200일선 위/아래였는지 확인하고,
      "200일선 위" 필터를 걸었을 때 승률이 실제로 개선되는지 계산한다.
"""
import yfinance as yf
import pandas as pd

# (ticker, date, zscore, fwd_return, win)
CASES = [
    ("ORCL", "2025-11-20", -1.70, 3.27, True),
    ("ORCL", "2025-04-03", -1.25, -5.94, False),
    ("ORCL", "2026-01-28", -1.08, -9.05, False),
    ("ADBE", "2026-06-25", -1.41, 15.63, True),
    ("ADBE", "2025-07-16", -1.00, 0.62, True),
    ("TXN", "2025-07-31", -1.81, 6.99, True),
    ("TXN", "2026-03-12", -1.36, 1.77, True),
    ("TXN", "2026-03-19", -1.27, 3.49, True),
    ("TXN", "2025-11-20", -1.17, 19.05, True),
    ("TXN", "2024-12-03", -1.13, -4.62, False),
    ("TXN", "2026-03-30", -1.03, 17.41, True),
    ("LOW", "2026-03-18", -1.96, 2.12, True),
    ("LOW", "2025-01-07", -1.54, 6.45, True),
    ("LOW", "2026-06-01", -1.32, 6.01, True),
    ("LOW", "2025-04-03", -1.14, -1.58, False),
    ("LOW", "2026-05-21", -1.06, -3.07, False),
    ("CHPT", "2025-11-20", -2.11, 31.86, True),
    ("CHPT", "2025-03-03", -1.96, 17.06, True),
    ("CHPT", "2026-02-05", -1.49, 10.39, True),
    ("CHPT", "2025-01-28", -1.20, -27.19, False),
    ("CHPT", "2024-11-08", -1.02, 1.77, True),
    ("CHPT", "2026-03-18", -1.00, -9.81, False),
    ("AFRM", "2026-02-23", -2.23, 7.81, True),
    ("AFRM", "2026-02-05", -1.91, -14.02, False),
    ("AFRM", "2025-03-18", -1.80, 4.14, True),
    ("AFRM", "2026-02-11", -1.80, -10.82, False),
    ("AFRM", "2026-02-27", -1.70, -0.21, False),
    ("AFRM", "2025-10-10", -1.42, 5.97, True),
    ("AFRM", "2026-03-27", -1.41, 22.20, True),
    ("AFRM", "2026-03-12", -1.26, -5.45, False),
    ("NIO", "2025-12-01", -2.01, -3.67, False),
    ("NIO", "2025-04-03", -1.22, -5.88, False),
    ("NIO", "2026-01-14", -1.10, 4.61, True),
    ("NIO", "2024-11-26", -1.00, 8.35, True),
    ("DKNG", "2025-10-10", -2.16, 0.95, True),
    ("DKNG", "2026-02-12", -1.44, -5.25, False),
    ("DKNG", "2025-03-13", -1.32, 0.14, True),
    ("DKNG", "2026-01-29", -1.18, -15.97, False),
    ("DKNG", "2025-03-18", -1.15, -10.08, False),
    ("DKNG", "2024-12-23", -1.15, 0.03, True),
    ("PLUG", "2025-04-14", -1.70, -9.43, False),
    ("PLUG", "2026-06-24", -1.65, -8.81, False),
    ("PLUG", "2025-03-03", -1.26, 13.33, True),
]


def get_sma200_status(hist, date_str):
    idx = hist.index.get_indexer([pd.Timestamp(date_str, tz=hist.index.tz)], method="nearest")[0]
    close = hist["Close"]
    sma200 = close.rolling(200).mean()
    if idx < 200 or pd.isna(sma200.iloc[idx]):
        return None
    return bool(close.iloc[idx] > sma200.iloc[idx])


def main():
    print(f"=== Z<=-1.0 전체 {len(CASES)}건에 대해 종목 200일선 상태 확인 ===\n")
    ticker_cache = {}
    results = []
    for ticker, date_str, zscore, fwd_return, win in CASES:
        if ticker not in ticker_cache:
            ticker_cache[ticker] = yf.Ticker(ticker).history(period="3y")
        hist = ticker_cache[ticker]
        above = get_sma200_status(hist, date_str)
        results.append({"ticker": ticker, "date": date_str, "zscore": zscore,
                         "fwd_return": fwd_return, "win": win, "sma200_above": above})
        status = "위" if above else ("아래" if above is not None else "N/A")
        print(f"  {ticker} {date_str}: Z={zscore:.2f} | {fwd_return:+.2f}% | {'승리' if win else '패배'} | 종목200일선 {status}")

    print(f"\n=== 필터링 전후 비교 ===")
    total_wins = sum(1 for r in results if r["win"])
    print(f"필터 없음(Z<=-1.0 전체): {total_wins}/{len(results)} = {total_wins/len(results)*100:.1f}%")

    above_group = [r for r in results if r["sma200_above"] is True]
    below_group = [r for r in results if r["sma200_above"] is False]

    aw = sum(1 for r in above_group if r["win"])
    bw = sum(1 for r in below_group if r["win"])
    print(f"\n200일선 위 그룹: {aw}/{len(above_group)} = {aw/len(above_group)*100:.1f}%" if above_group else "200일선 위 그룹: 표본 없음")
    print(f"200일선 아래 그룹: {bw}/{len(below_group)} = {bw/len(below_group)*100:.1f}%" if below_group else "200일선 아래 그룹: 표본 없음")


if __name__ == "__main__":
    main()
