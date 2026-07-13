"""
volume_recovery_axis_test.py
사용법: python volume_recovery_axis_test.py [N]
기본값: N=10

동작: 새 축2 - 최근N일 동안 "상승일 평균거래량 vs 하락일 평균거래량"을 비교해서
      "회복중"(상승일 거래량이 더 큼) / "미회복"(하락일 거래량이 더 큼)으로 나누고,
      다음날 지수 반응(평균수익률, 상승비율)을 확인한다.
      기존 축1(거래량수준)과도 조합해서 확인.
      QQQ+SPY+DIA, 5년치 데이터 사용.
"""
import sys
import numpy as np
import yfinance as yf

ETFS = ["QQQ", "SPY", "DIA"]
VOL_LOOKBACK_3M = 63
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10


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

        if i < N + 1:
            continue
        # 최근 N일(오늘 포함) 동안, 상승일/하락일 거래량 분리
        up_vols, down_vols = [], []
        for j in range(i - N + 1, i + 1):
            day_ret = float(close.iloc[j]) - float(close.iloc[j-1])
            if day_ret > 0:
                up_vols.append(float(vol.iloc[j]))
            elif day_ret < 0:
                down_vols.append(float(vol.iloc[j]))
        if not up_vols or not down_vols:
            continue
        up_avg = np.mean(up_vols)
        down_avg = np.mean(down_vols)
        recovering = up_avg > down_avg

        next_day_return = float(close.iloc[i+1]) / float(close.iloc[i]) - 1
        rows.append({"diff_pct": diff_pct, "recovering": recovering, "next_day_return": next_day_return})
    return rows


def main():
    all_rows = []
    for etf in ETFS:
        print(f"{etf} 데이터 로딩...")
        rows = collect_rows(etf)
        all_rows.extend(rows)

    overall_avg = np.mean([r["next_day_return"] for r in all_rows]) * 100
    overall_up = np.mean([r["next_day_return"] > 0 for r in all_rows]) * 100
    print(f"\n[전체 기준선] 평균 다음날수익률 {overall_avg:+.3f}% | 상승비율 {overall_up:.1f}% | 표본 {len(all_rows)}건 (N={N}일)\n")

    print("=== 축2 단독: 회복중 vs 미회복 ===")
    for val, label in [(True, "회복중(상승일거래량>하락일거래량)"), (False, "미회복(하락일거래량>=상승일거래량)")]:
        sub = [r for r in all_rows if r["recovering"] == val]
        returns = np.array([r["next_day_return"] for r in sub])
        avg_ret = returns.mean() * 100
        up_ratio = (returns > 0).mean() * 100
        std_ret = returns.std() * 100
        freq = len(sub) / len(all_rows) * 100
        print(f"  {label}: 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p) | 변동성 {std_ret:.3f}%")

    print("\n=== 축1(수준) x 축2(회복여부) 조합 ===\n")
    level_bands = [
        (-999, -30, "수준:낮음"),
        (-30, -10, "수준:낮음약함"),
        (-10, 10, "수준:정상"),
        (10, 40, "수준:높음약함"),
        (40, 999, "수준:높음"),
    ]
    for lo, hi, level_label in level_bands:
        for val, label in [(True, "회복중"), (False, "미회복")]:
            sub = [r for r in all_rows if lo <= r["diff_pct"] < hi and r["recovering"] == val]
            if not sub:
                continue
            returns = np.array([r["next_day_return"] for r in sub])
            avg_ret = returns.mean() * 100
            up_ratio = (returns > 0).mean() * 100
            freq = len(sub) / len(all_rows) * 100
            print(f"  [{level_label}+{label}] 표본{len(sub)}건({freq:.1f}%) | 평균수익률 {avg_ret:+.3f}%(기준대비{avg_ret-overall_avg:+.3f}%p) | 상승비율 {up_ratio:.1f}%(기준대비{up_ratio-overall_up:+.1f}%p)")


if __name__ == "__main__":
    main()
