"""
volume_level_market_impact.py
사용법: python volume_level_market_impact.py

동작: QQQ 거래량이 3개월평균 대비 특정 구간(수준)일 때,
      "다음날(1거래일 뒤) 지수 자체"가 얼마나 변동하는지(평균수익률, 변동성)와
      상승/하락 비율을 확인한다. 개별 종목 승률이 아니라 시장 자체의 반응을 본다.
"""
import numpy as np
import yfinance as yf

VOL_LOOKBACK_3M = 63


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
        diff_pct = (today_vol / avg_vol_3m - 1) * 100 if avg_vol_3m > 0 else None
        if diff_pct is None:
            continue
        next_day_return = float(close.iloc[i+1]) / float(close.iloc[i]) - 1
        rows.append({"diff_pct": diff_pct, "next_day_return": next_day_return})

    print(f"\n=== 표본 {len(rows)}건 - 거래량수준 구간별 '다음날 지수' 반응 ===\n")

    bands = [
        (-999, -30, "~-30% (경고, 극단적으로 낮음)"),
        (-30, -10, "-30~-10% (주의)"),
        (-10, 10, "-10~10% (정상)"),
        (10, 30, "10~30% (관망)"),
        (30, 999, "30%~ (경고, 극단적으로 높음)"),
    ]

    overall_avg = np.mean([r["next_day_return"] for r in rows]) * 100
    overall_up_ratio = np.mean([r["next_day_return"] > 0 for r in rows]) * 100
    print(f"[전체 기준선] 평균 다음날수익률 {overall_avg:+.3f}% | 상승비율 {overall_up_ratio:.1f}%\n")

    for lo, hi, label in bands:
        sub = [r for r in rows if lo <= r["diff_pct"] < hi]
        if not sub:
            continue
        returns = np.array([r["next_day_return"] for r in sub])
        avg_ret = returns.mean() * 100
        std_ret = returns.std() * 100
        up_ratio = (returns > 0).mean() * 100
        print(f"[{label}] 표본 {len(sub)}건({len(sub)/len(rows)*100:.1f}%)")
        print(f"  다음날 평균수익률: {avg_ret:+.3f}% (기준대비 {avg_ret-overall_avg:+.3f}%p)")
        print(f"  다음날 변동성(표준편차): {std_ret:.3f}%")
        print(f"  다음날 상승비율: {up_ratio:.1f}% (기준대비 {up_ratio-overall_up_ratio:+.1f}%p)")
        print()


if __name__ == "__main__":
    main()
