"""
volume_trend_period_sweep.py
사용법: python volume_trend_period_sweep.py

동작: "최근N일 평균거래량 vs 직전N일 평균거래량" 추세 비교를
      N=3,5,10,20일로 각각 계산해서, "다음날(1거래일 뒤) 지수 자체"의
      반응(평균수익률, 변동성, 상승비율)을 비교한다.
      어느 N이 방향성이 가장 뚜렷하고 노이즈 없이 안정적인지 확인.
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
        rows = []
        for i in range(60, n - 1):
            if i < 2 * N:
                continue
            recentN_avg = float(vol.iloc[i-N+1:i+1].mean())
            priorN_avg = float(vol.iloc[i-2*N+1:i-N+1].mean())
            if priorN_avg <= 0:
                continue
            trend_pct = (recentN_avg / priorN_avg - 1) * 100
            next_day_return = float(close.iloc[i+1]) / float(close.iloc[i]) - 1
            rows.append({"trend_pct": trend_pct, "next_day_return": next_day_return})

        print(f"=== N={N}일 (최근{N}일 vs 직전{N}일 평균거래량) - 표본 {len(rows)}건 ===")

        bands = [
            (-999, -20, "가파르게 감소(-20%미만)"),
            (-20, -5, "완만히 감소(-20~-5%)"),
            (-5, 5, "보합(-5~5%)"),
            (5, 20, "완만히 증가(5~20%)"),
            (20, 999, "가파르게 증가(20%이상)"),
        ]
        for lo, hi, label in bands:
            sub = [r for r in rows if lo <= r["trend_pct"] < hi]
            if not sub:
                continue
            returns = np.array([r["next_day_return"] for r in sub])
            avg_ret = returns.mean() * 100
            std_ret = returns.std() * 100
            up_ratio = (returns > 0).mean() * 100
            freq = len(sub) / len(rows) * 100
            print(f"  {label}: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 변동성 {std_ret:.3f}% | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")
        print()


if __name__ == "__main__":
    main()
