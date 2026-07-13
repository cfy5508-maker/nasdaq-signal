"""
market_health_v2_backtest.py
사용법: python market_health_v2_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 고정 %(30%,-30%) 기준선 대신, 두 축을 조합해서 거래량 상태를 분류한다.
  축1(수준): 오늘 거래량이 3개월 평균보다 높은지/낮은지
  축2(추세): 최근5일 평균거래량이 직전5일 평균거래량보다 늘고있는지/줄고있는지

  조합:
    수준높음+추세상승 = 거래량 급증
    수준높음+추세하락 = 고점에서꺾임
    수준낮음+추세상승 = 바닥에서반등
    수준낮음+추세하락 = 거래량 급감
"""
import sys
import numpy as np
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
    n = len(qqq)

    market_state = {}  # date -> category
    for i in range(VOL_LOOKBACK_3M, n):
        avg_vol_3m = float(qqq_vol.iloc[i-VOL_LOOKBACK_3M:i].mean())
        today_vol = float(qqq_vol.iloc[i])
        level_high = today_vol > avg_vol_3m

        if i < 10:
            continue
        recent5_avg = float(qqq_vol.iloc[i-4:i+1].mean())
        prior5_avg = float(qqq_vol.iloc[i-9:i-4].mean())
        trend_rising = recent5_avg > prior5_avg

        if level_high and trend_rising:
            category = "거래량급증"
        elif level_high and not trend_rising:
            category = "고점에서꺾임"
        elif not level_high and trend_rising:
            category = "바닥에서반등"
        else:
            category = "거래량급감"

        market_state[qqq.index[i].date()] = category

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
            category = market_state[d]
            entry = float(c.iloc[i])
            fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1
            all_rows.append({"category": category, "fwd_return": fwd_return, "win": fwd_return > 0})

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

    print("=== 4개 범주별 승률 (수준x추세 조합) ===")
    for cat in ["거래량급증", "고점에서꺾임", "바닥에서반등", "거래량급감"]:
        sub = [r for r in all_rows if r["category"] == cat]
        wr, avg = win_rate(sub)
        if wr is not None:
            freq = len(sub) / len(all_rows) * 100
            print(f"  {cat}: 표본 {len(sub)}건({freq:.1f}%) | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p | 평균수익률 {avg:+.2f}%")


if __name__ == "__main__":
    main()
