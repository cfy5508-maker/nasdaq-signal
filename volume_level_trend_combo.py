"""
volume_level_trend_combo.py
사용법: python volume_level_trend_combo.py

동작: ①거래량 수준(오늘 vs 3개월평균, 5구간)과 ②거래량 추세(N=3일, 상승/하락 이분법)를
      조합해서, 각 조합별로 다음날 지수 반응(평균수익률, 상승비율)을 확인한다.
"""
import numpy as np
import yfinance as yf

VOL_LOOKBACK_3M = 63
TREND_N = 3


def main():
    print("QQQ 데이터 로딩...")
    qqq = yf.Ticker("QQQ").history(period="2y")
    vol = qqq["Volume"]
    close = qqq["Close"]
    n = len(qqq)

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

    overall_avg = np.mean([r["next_day_return"] for r in rows]) * 100
    overall_up = np.mean([r["next_day_return"] > 0 for r in rows]) * 100
    print(f"\n[전체 기준선] 평균 다음날수익률 {overall_avg:+.3f}% | 상승비율 {overall_up:.1f}% | 표본 {len(rows)}건\n")

    level_bands = [
        (-999, -30, "수준:경고(낮음)"),
        (-30, -10, "수준:주의"),
        (-10, 10, "수준:정상"),
        (10, 30, "수준:관망"),
        (30, 999, "수준:경고(높음)"),
    ]

    print("=== ①수준 x ②추세(N=3일) 조합별 다음날 반응 ===\n")
    for lo, hi, level_label in level_bands:
        for trend_val, trend_label in [(True, "추세:상승중"), (False, "추세:하락중")]:
            sub = [r for r in rows if lo <= r["diff_pct"] < hi and r["trend_up"] == trend_val]
            if not sub:
                continue
            returns = np.array([r["next_day_return"] for r in sub])
            avg_ret = returns.mean() * 100
            up_ratio = (returns > 0).mean() * 100
            std_ret = returns.std() * 100
            freq = len(sub) / len(rows) * 100
            print(f"  [{level_label} + {trend_label}] 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p) | 변동성 {std_ret:.3f}%")
        print()


if __name__ == "__main__":
    main()
