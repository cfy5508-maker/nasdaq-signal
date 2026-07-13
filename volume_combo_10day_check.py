"""
volume_combo_10day_check.py
사용법: python volume_combo_10day_check.py

동작: 거래량 수준x추세 조합별로, "다음날"과 "10일후" 두 기준을 나란히 비교한다.
      QQQ+SPY+DIA, 5년치 데이터 사용 (지수 자체의 반응 기준).
"""
import numpy as np
import yfinance as yf

ETFS = ["QQQ", "SPY", "DIA"]
VOL_LOOKBACK_3M = 63
TREND_N = 3
FORWARD_DAYS_LONG = 10


def collect_rows(etf_symbol):
    hist = yf.Ticker(etf_symbol).history(period="5y")
    vol = hist["Volume"]
    close = hist["Close"]
    n = len(hist)

    rows = []
    for i in range(VOL_LOOKBACK_3M, n - FORWARD_DAYS_LONG):
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
        day10_return = float(close.iloc[i+FORWARD_DAYS_LONG]) / float(close.iloc[i]) - 1
        rows.append({"diff_pct": diff_pct, "trend_up": trend_up,
                     "next_day_return": next_day_return, "day10_return": day10_return})
    return rows


def main():
    all_rows = []
    for etf in ETFS:
        print(f"{etf} 데이터 로딩...")
        rows = collect_rows(etf)
        all_rows.extend(rows)

    def stats(key):
        arr = np.array([r[key] for r in all_rows])
        return arr.mean() * 100, (arr > 0).mean() * 100

    overall_avg_1d, overall_up_1d = stats("next_day_return")
    overall_avg_10d, overall_up_10d = stats("day10_return")
    print(f"\n[전체 기준선] 다음날: 평균{overall_avg_1d:+.3f}% 상승비율{overall_up_1d:.1f}% | 10일후: 평균{overall_avg_10d:+.3f}% 상승비율{overall_up_10d:.1f}% | 표본{len(all_rows)}건\n")

    level_bands = [
        (-999, -30, "수준:경고(낮음)"),
        (-30, -10, "수준:주의"),
        (-10, 10, "수준:정상"),
        (10, 30, "수준:관망"),
        (30, 999, "수준:경고(높음)"),
    ]

    print("=== ①수준 x ②추세(N=3일) - 다음날 vs 10일후 비교 ===\n")
    for lo, hi, level_label in level_bands:
        for trend_val, trend_label in [(True, "상승중"), (False, "하락중")]:
            sub = [r for r in all_rows if lo <= r["diff_pct"] < hi and r["trend_up"] == trend_val]
            if not sub:
                continue
            arr1 = np.array([r["next_day_return"] for r in sub])
            arr10 = np.array([r["day10_return"] for r in sub])
            avg1, up1 = arr1.mean()*100, (arr1>0).mean()*100
            avg10, up10 = arr10.mean()*100, (arr10>0).mean()*100
            freq = len(sub) / len(all_rows) * 100
            print(f"  [{level_label}+{trend_label}] 표본{len(sub)}건({freq:.1f}%)")
            print(f"    다음날 : 평균{avg1:+.3f}%(기준대비{avg1-overall_avg_1d:+.3f}%p) | 상승비율{up1:.1f}%(기준대비{up1-overall_up_1d:+.1f}%p)")
            print(f"    10일후 : 평균{avg10:+.3f}%(기준대비{avg10-overall_avg_10d:+.3f}%p) | 상승비율{up10:.1f}%(기준대비{up10-overall_up_10d:+.1f}%p)")
        print()


if __name__ == "__main__":
    main()
