"""
all_signals_common_traits_backtest.py
사용법: python all_signals_common_traits_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 대형주/중소형주 구분 없이, 다이버전스(RSI+MACD) 신호가 뜬 모든 날을
      모아서 10영업일 뒤 승리/패배로 나누고, 여러 특성의 평균을 비교해서
      승패를 가르는 공통점을 찾는다. 표본을 쪼개지 않고 전체로 본다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
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
    sma50 = c.rolling(50).mean()

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

        rsi_slope = float(rsi[low2_idx] - rsi[low1_idx])
        price_drop_ratio = float((c[low1_idx] - c[low2_idx]) / c[low1_idx] * 100)

        low2_price = float(c[low2_idx])
        subsequent_lows = l.iloc[low2_pos+1:i+1]
        held_the_low = bool((subsequent_lows >= low2_price).all()) if len(subsequent_lows) > 0 else True

        rsi_today = float(rsi.iloc[i])

        vol_recent5 = float(v.iloc[i-5:i].mean()) if i >= 5 else float("nan")
        vol_prior20 = float(v.iloc[i-25:i-5].mean()) if i >= 25 else float("nan")
        vol_ratio = vol_recent5 / vol_prior20 if vol_prior20 and vol_prior20 > 0 else float("nan")

        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        dist_from_sma20 = (float(c.iloc[i]) - sma20_now) / sma20_now * 100 if sma20_now else float("nan")

        sma50_now = float(sma50.iloc[i]) if not pd.isna(sma50.iloc[i]) else None
        above_sma50 = bool(float(c.iloc[i]) > sma50_now) if sma50_now else None

        pre_rebound = (float(c.iloc[i]) - low2_price) / low2_price * 100

        rsi_series = rsi.iloc[max(0, i-252):i+1].dropna()
        rsi_pctile = float((rsi_series < rsi_today).mean() * 100) if len(rsi_series) > 0 else 50.0

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "ticker": ticker,
            "rsi_slope": rsi_slope,
            "price_drop_ratio": price_drop_ratio,
            "held_the_low": held_the_low,
            "rsi_today": rsi_today,
            "rsi_pctile": rsi_pctile,
            "vol_ratio": vol_ratio,
            "dist_from_sma20": dist_from_sma20,
            "above_sma50": above_sma50,
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
    for t in TICKERS:
        recs = find_signals(t)
        if recs is None:
            print(f"{t}: 데이터 없음")
            continue
        all_signals.extend(recs)
        print(f"{t}: 다이버전스 신호 {len(recs)}건")

    print(f"\n=== 전체 다이버전스 신호 {len(all_signals)}건 (시가총액 구분 없음) ===\n")
    if not all_signals:
        print("신호 없음")
        return

    wins = [s for s in all_signals if s["win"]]
    losses = [s for s in all_signals if not s["win"]]
    baseline_wr = len(wins) / len(all_signals) * 100
    print(f"전체 승률: {baseline_wr:.1f}% | 승리: {len(wins)}건 | 패배: {len(losses)}건\n")

    features = [
        ("rsi_slope", "RSI 기울기"),
        ("price_drop_ratio", "가격 하락폭(%)"),
        ("rsi_today", "오늘 RSI 절대값"),
        ("rsi_pctile", "RSI 52주 백분위"),
        ("vol_ratio", "최근5일/이전20일 거래량 비율"),
        ("dist_from_sma20", "20일선 대비 이격도(%)"),
        ("pre_rebound", "저점 이후 선행 반등폭(%)"),
        ("days_since_low2", "저점 확정 후 경과일"),
    ]

    print("[특성별 승리 vs 패배 그룹 비교 - 차이가 클수록 결정적 요인]")
    diffs = []
    for key, label in features:
        win_vals = [s[key] for s in wins if not (isinstance(s[key], float) and np.isnan(s[key]))]
        loss_vals = [s[key] for s in losses if not (isinstance(s[key], float) and np.isnan(s[key]))]
        if not win_vals or not loss_vals:
            continue
        win_mean = np.mean(win_vals)
        loss_mean = np.mean(loss_vals)
        diff = win_mean - loss_mean
        diffs.append((label, win_mean, loss_mean, diff))

    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    for label, win_mean, loss_mean, diff in diffs:
        print(f"  {label}: 승리평균 {win_mean:.2f} vs 패배평균 {loss_mean:.2f} (차이 {diff:+.2f})")

    win_held = sum(1 for s in wins if s["held_the_low"])
    loss_held = sum(1 for s in losses if s["held_the_low"])
    print(f"\n  저점유지 비율: 승리그룹 {win_held/len(wins)*100:.0f}% | 패배그룹 {loss_held/len(losses)*100:.0f}%")

    win_above50 = sum(1 for s in wins if s["above_sma50"])
    loss_above50 = sum(1 for s in losses if s["above_sma50"])
    print(f"  50일선 위 비율: 승리그룹 {win_above50/len(wins)*100:.0f}% | 패배그룹 {loss_above50/len(losses)*100:.0f}%")

    print("\n해석: '차이'가 큰 특성일수록 승패를 가르는 결정적 조건일 가능성이 높음.")


if __name__ == "__main__":
    main()
