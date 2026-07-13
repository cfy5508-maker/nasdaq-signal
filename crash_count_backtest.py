"""
crash_count_backtest.py
사용법: python crash_count_backtest.py [N_PCT] [LOOKBACK_DAYS]
기본값: N_PCT=3(%), LOOKBACK_DAYS=20

동작: 최근 LOOKBACK_DAYS일 이내에 "하루 하락폭이 N_PCT% 이상"인 날이
      몇 번 있었는지 세고, 그 횟수별로 다음날 지수 반응(평균수익률, 상승비율)을
      비교한다. QQQ+SPY+DIA, 5년치 데이터 사용.
"""
import sys
import numpy as np
import yfinance as yf

ETFS = ["QQQ", "SPY", "DIA"]
N_PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
LOOKBACK_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def collect_rows(etf_symbol):
    hist = yf.Ticker(etf_symbol).history(period="5y")
    close = hist["Close"]
    n = len(hist)

    daily_ret = close.pct_change() * 100  # %

    rows = []
    for i in range(LOOKBACK_DAYS + 1, n - 1):
        window = daily_ret.iloc[i-LOOKBACK_DAYS:i]  # 오늘 제외, 최근 LOOKBACK_DAYS일
        crash_count = int((window <= -N_PCT).sum())

        next_day_return = float(close.iloc[i+1]) / float(close.iloc[i]) - 1
        rows.append({"crash_count": crash_count, "next_day_return": next_day_return})
    return rows


def main():
    all_rows = []
    for etf in ETFS:
        print(f"{etf} 데이터 로딩...")
        rows = collect_rows(etf)
        all_rows.extend(rows)

    overall_avg = np.mean([r["next_day_return"] for r in all_rows]) * 100
    overall_up = np.mean([r["next_day_return"] > 0 for r in all_rows]) * 100
    print(f"\n[전체 기준선] 평균 다음날수익률 {overall_avg:+.3f}% | 상승비율 {overall_up:.1f}% | 표본 {len(all_rows)}건")
    print(f"(하락캔들 기준: 하루 -{N_PCT}% 이상, 최근 {LOOKBACK_DAYS}일 이내 개수 세기)\n")

    max_count = max(r["crash_count"] for r in all_rows)
    print("=== 최근 N일 이내 하락캔들(N%+) 개수별 다음날 반응 ===\n")
    for cnt in range(0, min(max_count, 8) + 1):
        sub = [r for r in all_rows if r["crash_count"] == cnt]
        if not sub:
            continue
        returns = np.array([r["next_day_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  {cnt}회: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")

    if max_count > 8:
        sub = [r for r in all_rows if r["crash_count"] > 8]
        if sub:
            returns = np.array([r["next_day_return"] for r in sub])
            avg_ret = returns.mean() * 100
            up_ratio = (returns > 0).mean() * 100
            freq = len(sub) / len(all_rows) * 100
            print(f"  9회+: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")

    print("\n=== 참고: 0회 vs 1회이상 단순 이분 ===")
    for label, cond in [("0회(없음)", lambda r: r["crash_count"] == 0), ("1회이상", lambda r: r["crash_count"] >= 1)]:
        sub = [r for r in all_rows if cond(r)]
        returns = np.array([r["next_day_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  {label}: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")


if __name__ == "__main__":
    main()
