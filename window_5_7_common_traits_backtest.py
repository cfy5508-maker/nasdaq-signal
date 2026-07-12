"""
window_5_7_common_traits_backtest.py
사용법: python window_5_7_common_traits_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: "다이버전스 확정 후 5~7일 지난" 신호만 골라서(중소형주 최고 구간),
      그 안에서 결과적으로 이긴 케이스와 진 케이스를 나눠
      여러 특성(가격하락폭, RSI기울기, 저점유지여부, RSI절대값,
      거래량 패턴, 20일선 근접도, 선행반등폭 등)의 평균/분포를 비교한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

SMALLCAP_TICKERS = [
    "RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
    "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ",
    "FUBO", "PATH", "ASAN", "BBAI", "SOUN", "RUN", "ENVX", "JOBY", "ACHR", "LAZR",
    "QS", "FSLY", "PLUG", "NIO", "XPEV", "BYND", "WOLF", "GPRO", "DNA", "HOOD",
    "AI", "PTON", "SNAP", "LYFT", "DASH", "PINS", "ETSY", "BILL", "TTD", "ROKU",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = 250
MIN_HISTORY = 260


def find_signals(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    c, h, l, v = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    rsi = RSIIndicator(c, window=14).rsi()
    macd_hist = MACD(c).macd_diff()
    sma20 = c.rolling(20).mean()

    n = len(hist)
    start = max(MIN_HISTORY, n - LOOKBACK_DAYS - FORWARD_DAYS)
    end = n - FORWARD_DAYS

    results = []
    for i in range(start, end):
        recent = c.iloc[max(0, i-90):i+1]
        if len(recent) < 90:
            continue

        low1_idx = recent.iloc[:45].idxmin()
        low2_idx = recent.iloc[45:].idxmin()
        bullish_divergence = bool(c[low2_idx] < c[low1_idx] and rsi[low2_idx] > rsi[low1_idx])

        mh = macd_hist.iloc[max(0, i-90):i+1]
        if len(mh) < 90:
            continue
        macd_rising = bool(mh.iloc[-1] > mh.iloc[:-1].min())

        if not (bullish_divergence and macd_rising):
            continue

        low2_pos = hist.index.get_loc(low2_idx)
        days_since_low2 = i - low2_pos

        if not (5 <= days_since_low2 < 7):
            continue

        rsi_slope = float(rsi[low2_idx] - rsi[low1_idx])
        price_drop_ratio = float((c[low1_idx] - c[low2_idx]) / c[low1_idx] * 100)

        low2_price = float(c[low2_idx])
        subsequent_lows = l.iloc[low2_pos+1:i+1]
        held_the_low = bool((subsequent_lows >= low2_price).all()) if len(subsequent_lows) > 0 else True

        # 저점을 깼다면 얼마나 더 깼는지(%), 안 깼으면 0
        min_subsequent_low = float(subsequent_lows.min()) if len(subsequent_lows) > 0 else low2_price
        new_low_break_pct = max(0.0, (low2_price - min_subsequent_low) / low2_price * 100)

        rsi_today = float(rsi.iloc[i])

        vol_recent5 = float(v.iloc[i-5:i].mean()) if i >= 5 else float("nan")
        vol_prior5 = float(v.iloc[i-10:i-5].mean()) if i >= 10 else float("nan")
        vol_ratio = vol_recent5 / vol_prior5 if vol_prior5 and vol_prior5 > 0 else float("nan")

        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        dist_from_sma20 = (float(c.iloc[i]) - sma20_now) / sma20_now * 100 if sma20_now else float("nan")

        pre_rebound = (float(c.iloc[i]) - low2_price) / low2_price * 100

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "ticker": ticker,
            "rsi_slope": rsi_slope,
            "price_drop_ratio": price_drop_ratio,
            "held_the_low": held_the_low,
            "new_low_break_pct": new_low_break_pct,
            "rsi_today": rsi_today,
            "vol_ratio": vol_ratio,
            "dist_from_sma20": dist_from_sma20,
            "pre_rebound": pre_rebound,
            "days_since_low2": days_since_low2,
            "fwd_return": fwd_return,
            "win": fwd_return > 0,
        })
    return results


def summarize(label, vals):
    arr = np.array([v for v in vals if not (isinstance(v, float) and np.isnan(v))])
    if len(arr) == 0:
        print(f"    {label}: 데이터 없음")
        return
    print(f"    {label}: 평균 {arr.mean():.2f} | 중앙값 {np.median(arr):.2f} | 표본 {len(arr)}")


def main():
    all_signals = []
    for t in SMALLCAP_TICKERS:
        recs = find_signals(t)
        if recs is None:
            continue
        all_signals.extend(recs)

    print(f"=== 5~7일 대기 구간 신호 {len(all_signals)}건 (중소형주) ===\n")
    if not all_signals:
        print("신호 없음")
        return

    wins = [s for s in all_signals if s["win"]]
    losses = [s for s in all_signals if not s["win"]]
    print(f"승리: {len(wins)}건 | 패배: {len(losses)}건\n")

    features = [
        ("rsi_slope", "RSI 기울기"),
        ("price_drop_ratio", "가격 하락폭(%)"),
        ("rsi_today", "오늘 RSI 절대값"),
        ("vol_ratio", "최근5일/이전5일 거래량 비율"),
        ("dist_from_sma20", "20일선 대비 이격도(%)"),
        ("pre_rebound", "저점 이후 선행 반등폭(%)"),
        ("new_low_break_pct", "저점 갱신폭(더 깬 정도, %)"),
    ]

    print("[특성별 승리 vs 패배 그룹 비교]")
    for key, label in features:
        print(f"  {label}")
        summarize("승리", [s[key] for s in wins])
        summarize("패배", [s[key] for s in losses])
        print()

    win_held = sum(1 for s in wins if s["held_the_low"])
    loss_held = sum(1 for s in losses if s["held_the_low"])
    print(f"  저점유지 비율: 승리그룹 {win_held}/{len(wins)}건({win_held/len(wins)*100:.0f}%) | "
          f"패배그룹 {loss_held}/{len(losses)}건({loss_held/len(losses)*100 if losses else 0:.0f}%)")


if __name__ == "__main__":
    main()
