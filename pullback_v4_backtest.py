"""
pullback_v4_backtest.py
사용법: python pullback_v4_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

제미나이 제안 3가지를 각각 독립적으로(조합 아님) 검증:
  1. 셋업/트리거 분리: 셋업일(이평선근접+RSI50+) 당일 고가를, 그 다음날~3일내 종가가 돌파하는지
  2. ATR 기반 이격도: 고정 3% 대신 0.5*ATR로 근접 여부 재정의, 기존 3%기준과 승률 비교
  3. 손절선(직전 저점 이탈): 진입 후 forward_days 안에 직전 스윙저점을 종가기준 이탈한 비율과,
     이탈했을 때/안했을때 최종 수익률 비교
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SWING_ORDER = 5
MA_TOLERANCE_PCT = 0.03


def find_swing_lows(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n - order):
        window = close_values[i-order:i+order+1]
        if close_values[i] == window.min() and (window == close_values[i]).sum() == 1:
            positions.append(i)
    return positions


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 300:
        return None
    o, h, l, c = hist["Open"], hist["High"], hist["Low"], hist["Close"]
    close_values = c.values
    sma20 = c.rolling(20).mean()
    sma60 = c.rolling(60).mean()
    rsi = RSIIndicator(c, window=14).rsi()
    atr = AverageTrueRange(h, l, c, window=14).average_true_range()
    n = len(hist)

    swing_lows_full = find_swing_lows(close_values, order=SWING_ORDER)

    setup_rows = []       # ①번 검증용: 셋업 발생일 전부
    ma3pct_rows = []      # ②번 비교용: 3% 고정 근접 신호
    ma_atr_rows = []      # ②번 비교용: ATR 기반 근접 신호
    stop_rows = []        # ③번 검증용: 기존 게이트(신저점확정+3%근접+RSI50+) 신호에 손절 통계

    for i in range(280, n - FORWARD_DAYS - 5):
        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        sma60_now = float(sma60.iloc[i]) if not pd.isna(sma60.iloc[i]) else None
        rsi_now = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else None
        atr_now = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else None
        price_now = close_values[i]
        if sma20_now is None or sma60_now is None or rsi_now is None:
            continue
        rsi_alive = rsi_now >= 50

        # --- 근접 판정: 3%고정 vs ATR기반(0.5*ATR) ---
        near_3pct = (sma20_now and abs(price_now - sma20_now) / sma20_now <= MA_TOLERANCE_PCT) or \
                    (sma60_now and abs(price_now - sma60_now) / sma60_now <= MA_TOLERANCE_PCT)
        near_atr = False
        if atr_now:
            near_atr = (abs(price_now - sma20_now) <= 0.5 * atr_now) or (abs(price_now - sma60_now) <= 0.5 * atr_now)

        entry = price_now
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        if near_3pct and rsi_alive:
            ma3pct_rows.append({"fwd_return": fwd_return, "win": fwd_return > 0})
        if near_atr and rsi_alive:
            ma_atr_rows.append({"fwd_return": fwd_return, "win": fwd_return > 0})

        # --- ①번: 셋업/트리거 분리 ---
        # 셋업(오늘) 조건 만족 시, 셋업 당일 고가를 저장하고 이후 1~3일 내 돌파 여부 확인
        if near_3pct and rsi_alive:
            setup_high = float(h.iloc[i])
            triggered = False
            trigger_lag = None
            for lag in range(1, 4):
                if i + lag >= n:
                    break
                if float(c.iloc[i + lag]) > setup_high:
                    triggered = True
                    trigger_lag = lag
                    break
            if triggered:
                entry_t = float(c.iloc[i + trigger_lag])
                if i + trigger_lag + FORWARD_DAYS < n:
                    fwd_t = float(c.iloc[i + trigger_lag + FORWARD_DAYS]) / entry_t - 1
                    setup_rows.append({"fwd_return": fwd_t, "win": fwd_t > 0, "lag": trigger_lag})

        # --- ③번: 기존 게이트(신저점확정 시점) 방식으로 손절 통계 ---
        lows_upto = [p for p in swing_lows_full if p + SWING_ORDER <= i]
        if len(lows_upto) >= 1:
            recent_low = lows_upto[-1]
            is_freshly_confirmed = (i == recent_low + SWING_ORDER)
            if is_freshly_confirmed and near_3pct and rsi_alive:
                recent_low_price = close_values[recent_low]
                future_closes = c.iloc[i+1:i+1+FORWARD_DAYS]
                stopped_out = bool((future_closes < recent_low_price).any())
                stop_rows.append({"fwd_return": fwd_return, "win": fwd_return > 0, "stopped_out": stopped_out})

    return setup_rows, ma3pct_rows, ma_atr_rows, stop_rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_setup, all_ma3, all_maatr, all_stop = [], [], [], []
    for t in LARGECAP_TICKERS:
        try:
            s, m3, matr, st = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        all_setup.extend(s); all_ma3.extend(m3); all_maatr.extend(matr); all_stop.extend(st)

    print(f"\n=== ① 셋업/트리거 분리 검증 ({FORWARD_DAYS}일 뒤 기준) ===")
    wr, avg = win_rate(all_setup)
    if wr is not None:
        print(f"표본 {len(all_setup)}건 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}%")
        for lag_val in [1, 2, 3]:
            sub = [r for r in all_setup if r["lag"] == lag_val]
            wr_l, avg_l = win_rate(sub)
            if wr_l is not None:
                print(f"  {lag_val}일만에 돌파: 표본 {len(sub)} | 승률 {wr_l:.1f}%")
    else:
        print("표본 없음")

    print(f"\n=== ② ATR기반 vs 고정3% 이격도 비교 (근접+RSI50이상, 매일 스캔 기준) ===")
    wr3, avg3 = win_rate(all_ma3)
    wratr, avgatr = win_rate(all_maatr)
    if wr3 is not None:
        print(f"고정 3% 기준: 표본 {len(all_ma3)} | 승률 {wr3:.1f}% | 평균수익률 {avg3:+.2f}%")
    if wratr is not None:
        print(f"ATR(0.5x) 기준: 표본 {len(all_maatr)} | 승률 {wratr:.1f}% | 평균수익률 {avgatr:+.2f}%")

    print(f"\n=== ③ 손절선(직전 스윙저점 이탈) 검증 ===")
    wr_s, avg_s = win_rate(all_stop)
    if wr_s is not None:
        print(f"전체(기존게이트, 신저점확정+3%근접+RSI50+): 표본 {len(all_stop)} | 승률 {wr_s:.1f}% | 평균수익률 {avg_s:+.2f}%")
        stopped = [r for r in all_stop if r["stopped_out"]]
        not_stopped = [r for r in all_stop if not r["stopped_out"]]
        wr_st, avg_st = win_rate(stopped)
        wr_ns, avg_ns = win_rate(not_stopped)
        stop_rate = len(stopped) / len(all_stop) * 100
        print(f"  손절선(직전저점) 터치 비율: {stop_rate:.1f}% ({len(stopped)}/{len(all_stop)})")
        if wr_st is not None:
            print(f"  손절터치 그룹의 {FORWARD_DAYS}일뒤 결과: 승률 {wr_st:.1f}% | 평균수익률 {avg_st:+.2f}% (참고: 실제로는 손절로 먼저 청산됐을 것)")
        if wr_ns is not None:
            print(f"  손절 안터치 그룹: 승률 {wr_ns:.1f}% | 평균수익률 {avg_ns:+.2f}%")
    else:
        print("표본 없음")


if __name__ == "__main__":
    main()
