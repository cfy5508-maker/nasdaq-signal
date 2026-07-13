"""
cross_signal_backtest.py
사용법: python cross_signal_backtest.py [FORWARD_DAYS] [SHORT_MA] [LONG_MA]
기본값: FORWARD_DAYS=5, SHORT_MA=20, LONG_MA=60

동작: 단기이평선(SHORT_MA)이 장기이평선(LONG_MA)을 교차하는 지점(골든/데드크로스)을
      찾아서, "최근 10일 이내에 그 크로스가 발생했는지"에 따라 이후 지수 반응을 비교한다.
      QQQ+SPY+DIA, 5년치 데이터 사용.
"""
import sys
import numpy as np
import yfinance as yf

ETFS = ["QQQ", "SPY", "DIA"]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SHORT_MA = int(sys.argv[2]) if len(sys.argv) > 2 else 20
LONG_MA = int(sys.argv[3]) if len(sys.argv) > 3 else 60
CROSS_LOOKBACK = 10


def collect_rows(etf_symbol):
    hist = yf.Ticker(etf_symbol).history(period="5y")
    close = hist["Close"]
    n = len(hist)

    sma_short = close.rolling(SHORT_MA).mean()
    sma_long = close.rolling(LONG_MA).mean()
    above = sma_short > sma_long  # True: 단기>장기(정배열)

    rows = []
    start = LONG_MA + CROSS_LOOKBACK + 1
    for i in range(start, n - FORWARD_DAYS):
        # 최근 CROSS_LOOKBACK일 이내에 크로스(above 값이 바뀐 지점)가 있었는지
        golden_recent = False
        dead_recent = False
        for j in range(i - CROSS_LOOKBACK + 1, i + 1):
            if pd_isna_safe(above.iloc[j]) or pd_isna_safe(above.iloc[j-1]):
                continue
            if above.iloc[j] and not above.iloc[j-1]:
                golden_recent = True
            if not above.iloc[j] and above.iloc[j-1]:
                dead_recent = True

        fwd_return = float(close.iloc[i+FORWARD_DAYS]) / float(close.iloc[i]) - 1
        rows.append({"golden_recent": golden_recent, "dead_recent": dead_recent, "fwd_return": fwd_return})
    return rows


def pd_isna_safe(v):
    try:
        return v != v  # NaN != NaN is True
    except Exception:
        return False


def main():
    all_rows = []
    for etf in ETFS:
        print(f"{etf} 데이터 로딩...")
        rows = collect_rows(etf)
        all_rows.extend(rows)

    overall_avg = np.mean([r["fwd_return"] for r in all_rows]) * 100
    overall_up = np.mean([r["fwd_return"] > 0 for r in all_rows]) * 100
    print(f"\n[전체 기준선] {FORWARD_DAYS}일뒤 평균수익률 {overall_avg:+.3f}% | 상승비율 {overall_up:.1f}% | 표본 {len(all_rows)}건")
    print(f"(단기{SHORT_MA}일선 vs 장기{LONG_MA}일선, 최근{CROSS_LOOKBACK}일 이내 크로스 여부)\n")

    print("=== 골든크로스(최근10일 이내 발생) 여부 ===")
    for val, label in [(True, "최근10일내 골든크로스 있음"), (False, "없음")]:
        sub = [r for r in all_rows if r["golden_recent"] == val]
        returns = np.array([r["fwd_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  {label}: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")

    print("\n=== 데드크로스(최근10일 이내 발생) 여부 ===")
    for val, label in [(True, "최근10일내 데드크로스 있음"), (False, "없음")]:
        sub = [r for r in all_rows if r["dead_recent"] == val]
        returns = np.array([r["fwd_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  {label}: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")


if __name__ == "__main__":
    main()
