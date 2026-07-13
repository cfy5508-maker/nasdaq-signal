"""
market_health_backtest.py
사용법: python market_health_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 나스닥 대표 ETF(QQQ)의 "거래량 3개월평균 대비 %"를 세밀한 구간으로 나누고,
      "그날 지수가 상승했는지 하락했는지"와 교차해서, 그 이후 나스닥 상장
      대형주들의 평균 승률이 어떻게 달라지는지 확인한다.
      (1단계 시장건전성 지표를 점수화하기 전 사전 검증용)
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "META", "AMD", "TSLA", "NFLX", "CSCO",
    "MSFT", "GOOGL", "AMZN", "ADBE", "CRM", "TXN", "AVGO", "QCOM",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
VOL_LOOKBACK_3M = 63


def main():
    print("QQQ(나스닥100 ETF) 데이터 로딩...")
    qqq = yf.Ticker("QQQ").history(period="2y")
    qqq_vol = qqq["Volume"]
    qqq_close = qqq["Close"]

    # 매일의 거래량%, 지수방향 계산
    n = len(qqq)
    market_state = {}  # date -> (vol_diff_pct, price_up)
    for i in range(VOL_LOOKBACK_3M, n):
        avg_vol = float(qqq_vol.iloc[i-VOL_LOOKBACK_3M:i].mean())
        today_vol = float(qqq_vol.iloc[i])
        vol_diff_pct = (today_vol / avg_vol - 1) * 100 if avg_vol > 0 else None
        price_up = bool(qqq_close.iloc[i] > qqq_close.iloc[i-1])
        market_state[qqq.index[i].date()] = (vol_diff_pct, price_up)

    all_rows = []
    for t in LARGECAP_TICKERS:
        try:
            hist = yf.Ticker(t).history(period="2y")
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if hist.empty or len(hist) < 100:
            continue
        c = hist["Close"]
        m = len(hist)
        for i in range(VOL_LOOKBACK_3M, m - FORWARD_DAYS):
            d = hist.index[i].date()
            if d not in market_state:
                continue
            vol_diff_pct, price_up = market_state[d]
            if vol_diff_pct is None:
                continue
            entry = float(c.iloc[i])
            fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1
            all_rows.append({"vol_diff_pct": vol_diff_pct, "price_up": price_up,
                             "fwd_return": fwd_return, "win": fwd_return > 0})

    print(f"\n=== 전체 표본 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    def win_rate(recs):
        if not recs:
            return None, None
        arr = np.array([r["fwd_return"] for r in recs])
        return float((arr > 0).mean() * 100), float(arr.mean() * 100)

    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"기준선: 승률 {baseline_wr:.1f}% | 평균수익률 {baseline_avg:+.2f}%\n")

    bands = [(-999, -50), (-50, -30), (-30, -10), (-10, 0), (0, 10), (10, 30), (30, 50), (50, 999)]

    print("=== 거래량 구간별 승률 (지수방향 무관) ===")
    for lo, hi in bands:
        sub = [r for r in all_rows if lo <= r["vol_diff_pct"] < hi]
        wr, avg = win_rate(sub)
        lo_l = str(lo) if lo > -999 else "~"
        hi_l = str(hi) if hi < 999 else "이상"
        if wr is not None:
            print(f"  {lo_l}~{hi_l}%: 표본 {len(sub)}건 | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n=== 거래량 구간 x 지수방향(상승/하락) 교차분석 ===")
    for lo, hi in bands:
        for up_val, up_label in [(True, "지수상승"), (False, "지수하락")]:
            sub = [r for r in all_rows if lo <= r["vol_diff_pct"] < hi and r["price_up"] == up_val]
            wr, avg = win_rate(sub)
            lo_l = str(lo) if lo > -999 else "~"
            hi_l = str(hi) if hi < 999 else "이상"
            if wr is not None:
                print(f"  {lo_l}~{hi_l}% + {up_label}: 표본 {len(sub)}건 | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")


if __name__ == "__main__":
    main()
