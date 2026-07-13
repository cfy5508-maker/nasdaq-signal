"""
warning_threshold_sweep.py
사용법: python warning_threshold_sweep.py

동작: "경고(높음)" 기준선을 20/25/30/40/50/70/100%로 스윕하면서,
      "경고(높음) + 추세하락중(N=3일)" 조합의 승률과 변동성이 어떻게 변하는지 확인한다.
      QQQ+SPY+DIA, 5년치 데이터 사용.
"""
import numpy as np
import yfinance as yf

ETFS = ["QQQ", "SPY", "DIA"]
VOL_LOOKBACK_3M = 63
TREND_N = 3
THRESHOLDS = [20, 25, 30, 40, 50, 70, 100]


def collect_rows(etf_symbol):
    hist = yf.Ticker(etf_symbol).history(period="5y")
    vol = hist["Volume"]
    close = hist["Close"]
    n = len(hist)

    rows = []
    for i in range(VOL_LOOKBACK_3M, n - 1):
        avg_vol_3m = float(vol.iloc[i-VOL_LOOKBACK_3M:i].mean())
        today_vol = float(vol.iloc[i])
        if avg_vol_3m <= 0:
            continue
        diff_pct = (today_vol / avg_vol_3m - 1) * 100

        if i < 2 * TREND_N:
            continue
        recentN_avg = float(vol.iloc[i-TREND_N+1:i+1].mean())
        priorN_avg = float(vol.iloc[i-2*TREND_N+1:i-TREND_N+1].mean())
        if priorN_avg <= 0:
            continue
        trend_up = recentN_avg > priorN_avg

        next_day_return = float(close.iloc[i+1]) / float(close.iloc[i]) - 1
        rows.append({"diff_pct": diff_pct, "trend_up": trend_up, "next_day_return": next_day_return})
    return rows


def main():
    all_rows = []
    for etf in ETFS:
        print(f"{etf} 데이터 로딩...")
        rows = collect_rows(etf)
        all_rows.extend(rows)

    overall_avg = np.mean([r["next_day_return"] for r in all_rows]) * 100
    overall_up = np.mean([r["next_day_return"] > 0 for r in all_rows]) * 100
    print(f"\n[전체 기준선] 평균 다음날수익률 {overall_avg:+.3f}% | 상승비율 {overall_up:.1f}% | 표본 {len(all_rows)}건\n")

    print("=== '경고(높음)+추세하락중' 조합의 임계값(threshold) 스윕 ===\n")
    for th in THRESHOLDS:
        sub = [r for r in all_rows if r["diff_pct"] >= th and not r["trend_up"]]
        if not sub:
            print(f"  threshold={th}%: 표본 없음")
            continue
        returns = np.array([r["next_day_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        std_ret = returns.std() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  threshold>={th}%: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p) | 변동성 {std_ret:.3f}%")

    print("\n=== 참고: '경고(높음)+추세상승중' 조합도 같이 스윕(비교용) ===\n")
    for th in THRESHOLDS:
        sub = [r for r in all_rows if r["diff_pct"] >= th and r["trend_up"]]
        if not sub:
            print(f"  threshold={th}%: 표본 없음")
            continue
        returns = np.array([r["next_day_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        std_ret = returns.std() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  threshold>={th}%: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p) | 변동성 {std_ret:.3f}%")


if __name__ == "__main__":
    main()
