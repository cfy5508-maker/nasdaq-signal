"""
decline_start_check.py
사용법: python decline_start_check.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=5

동작: "하락 초입"을 판단하는 두 후보를 5일뒤 지수반응으로 검증한다.
  ① 연속음봉: 오늘까지 며칠 연속으로 하락했는지(1,2,3,4,5일+)
  ② 빈도: 최근10일 중 음봉이 몇 번이었는지(0~10회)
  QQQ+SPY+DIA, 5년치 데이터 사용.
"""
import sys
import numpy as np
import yfinance as yf

ETFS = ["QQQ", "SPY", "DIA"]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
FREQ_LOOKBACK = 10


def collect_rows(etf_symbol):
    hist = yf.Ticker(etf_symbol).history(period="5y")
    close = hist["Close"]
    n = len(hist)
    daily_ret = close.pct_change()

    rows = []
    for i in range(FREQ_LOOKBACK + 1, n - FORWARD_DAYS):
        # ① 연속 음봉 개수 (오늘 포함, 오늘부터 거슬러 올라가며 연속 하락일 카운트)
        consec = 0
        for j in range(i, -1, -1):
            if daily_ret.iloc[j] < 0:
                consec += 1
            else:
                break

        # ② 최근 10일(오늘포함) 중 음봉 개수
        window = daily_ret.iloc[i-FREQ_LOOKBACK+1:i+1]
        down_freq = int((window < 0).sum())

        fwd_return = float(close.iloc[i+FORWARD_DAYS]) / float(close.iloc[i]) - 1
        rows.append({"consec": consec, "down_freq": down_freq, "fwd_return": fwd_return})
    return rows


def main():
    all_rows = []
    for etf in ETFS:
        print(f"{etf} 데이터 로딩...")
        rows = collect_rows(etf)
        all_rows.extend(rows)

    overall_avg = np.mean([r["fwd_return"] for r in all_rows]) * 100
    overall_up = np.mean([r["fwd_return"] > 0 for r in all_rows]) * 100
    print(f"\n[전체 기준선] {FORWARD_DAYS}일뒤 평균수익률 {overall_avg:+.3f}% | 상승비율 {overall_up:.1f}% | 표본 {len(all_rows)}건\n")

    print(f"=== ① 연속 음봉 개수별 {FORWARD_DAYS}일뒤 반응 ===\n")
    for c in range(0, 7):
        sub = [r for r in all_rows if r["consec"] == c]
        if not sub:
            continue
        returns = np.array([r["fwd_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  연속{c}일: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")
    sub = [r for r in all_rows if r["consec"] >= 7]
    if sub:
        returns = np.array([r["fwd_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  연속7일+: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")

    print(f"\n=== ② 최근10일 음봉 빈도별 {FORWARD_DAYS}일뒤 반응 ===\n")
    for c in range(0, 11):
        sub = [r for r in all_rows if r["down_freq"] == c]
        if not sub:
            continue
        returns = np.array([r["fwd_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  {c}회: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")


if __name__ == "__main__":
    main()
