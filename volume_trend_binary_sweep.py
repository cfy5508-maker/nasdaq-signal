"""
volume_trend_binary_sweep.py
사용법: python volume_trend_binary_sweep.py

동작: "최근N일 평균거래량 vs 직전N일 평균거래량"을 강도 구분 없이
      단순히 "상승중(recentN > priorN)" / "하락중(recentN < priorN)" 이분법으로만 나눠서,
      N=3,5,10,20일 각각 다음날 지수 반응(평균수익률, 상승비율)을 비교한다.
"""
import numpy as np
import yfinance as yf

PERIODS = [3, 5, 10, 20]


def main():
    print("QQQ 데이터 로딩...")
    qqq = yf.Ticker("QQQ").history(period="2y")
    vol = qqq["Volume"]
    close = qqq["Close"]
    n = len(qqq)

    overall_returns = []
    for i in range(60, n - 1):
        overall_returns.append(float(close.iloc[i+1]) / float(close.iloc[i]) - 1)
    overall_avg = np.mean(overall_returns) * 100
    overall_up = np.mean([r > 0 for r in overall_returns]) * 100
    print(f"[전체 기준선] 평균 다음날수익률 {overall_avg:+.3f}% | 상승비율 {overall_up:.1f}%\n")

    for N in PERIODS:
        rising, falling = [], []
        for i in range(60, n - 1):
            if i < 2 * N:
                continue
            recentN_avg = float(vol.iloc[i-N+1:i+1].mean())
            priorN_avg = float(vol.iloc[i-2*N+1:i-N+1].mean())
            if priorN_avg <= 0:
                continue
            next_day_return = float(close.iloc[i+1]) / float(close.iloc[i]) - 1
            if recentN_avg > priorN_avg:
                rising.append(next_day_return)
            else:
                falling.append(next_day_return)

        print(f"=== N={N}일 (최근{N}일 vs 직전{N}일 평균거래량) ===")
        for label, arr in [("상승중(거래량 늘어나는 중)", rising), ("하락중(거래량 줄어드는 중)", falling)]:
            arr = np.array(arr)
            avg_ret = arr.mean() * 100
            up_ratio = (arr > 0).mean() * 100
            std_ret = arr.std() * 100
            print(f"  {label}: 표본{len(arr)}건 | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p) | 변동성 {std_ret:.3f}%")
        print()


if __name__ == "__main__":
    main()
