"""
zone_20_30_downtrend_traits_backtest.py
사용법: python zone_20_30_downtrend_traits_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: "다이버전스 + RSI백분위 20~30% + 궤적하락중" 조건에 해당하는
      신호들만 골라서, 10일 뒤 승리/패배로 나누고 다음 5가지 특성의
      평균 차이를 비교한다.
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
    "W", "CHWY", "PENN", "ABNB", "DOCU", "ZM", "BROS", "CAVA", "SG", "APP",
    "SMR", "OKLO", "VKTX", "RXRX", "ARQQ", "DAVE", "NU", "CELH", "ELF", "FIGS",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = 250
MIN_HISTORY = 260


def rolling_pctile_at(rsi, pos, window=252):
    start = max(0, pos - window)
    w = rsi.iloc[start:pos+1].dropna()
    if len(w) == 0:
        return None
    val = rsi.iloc[pos]
    return float((w < val).mean() * 100)


def find_signals(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    rsi = RSIIndicator(c, window=14).rsi()
    macd_hist = MACD(c).macd_diff()
    sma20 = c.rolling(20).mean()

    n = len(hist)
    start = max(MIN_HISTORY, n - LOOKBACK_DAYS - FORWARD_DAYS)
    end = n - FORWARD_DAYS

    all_div_positions = []

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

        is_divergence_day = bullish_divergence and macd_rising
        if is_divergence_day:
            all_div_positions.append(i)

        if not is_divergence_day:
            continue

        low2_pos = hist.index.get_loc(low2_idx)
        if low2_pos < 5:
            continue

        pctile_now = rolling_pctile_at(rsi, low2_pos)
        pctile_5ago = rolling_pctile_at(rsi, low2_pos - 5)
        if pctile_now is None or pctile_5ago is None:
            continue

        if not (20 <= pctile_now < 30):
            continue
        if not (pctile_now < pctile_5ago):
            continue

        vol_recent5 = float(v.iloc[max(0, i-5):i].mean()) if i >= 5 else float("nan")
        vol_prior20 = float(v.iloc[max(0, i-25):max(0, i-5)].mean()) if i >= 25 else float("nan")
        vol_spike_ratio = vol_recent5 / vol_prior20 if vol_prior20 and vol_prior20 > 0 else float("nan")

        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        dist_from_sma20 = (float(c.iloc[i]) - sma20_now) / sma20_now * 100 if sma20_now else float("nan")

        repeat_count = sum(1 for p in all_div_positions if i - 20 <= p < i)

        today_candle_pct = (float(c.iloc[i]) - float(o.iloc[i])) / float(o.iloc[i]) * 100

        decline_speed = pctile_5ago - pctile_now

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "vol_spike_ratio": vol_spike_ratio,
            "dist_from_sma20": dist_from_sma20,
            "repeat_count": repeat_count,
            "today_candle_pct": today_candle_pct,
            "decline_speed": decline_speed,
            "fwd_return": fwd_return,
            "win": fwd_return > 0,
        })
    return results


def summarize(vals):
    arr = np.array([v for v in vals if not (isinstance(v, float) and np.isnan(v))])
    if len(arr) == 0:
        return None
    return arr.mean(), np.median(arr), len(arr)


def main():
    all_signals = []
    for t in SMALLCAP_TICKERS:
        recs = find_signals(t)
        if recs:
            all_signals.extend(recs)

    print(f"=== '다이버전스+RSI백분위20~30%+궤적하락중' 신호 {len(all_signals)}건 ===\n")
    if not all_signals:
        print("신호 없음")
        return

    wins = [s for s in all_signals if s["win"]]
    losses = [s for s in all_signals if not s["win"]]
    baseline_wr = len(wins) / len(all_signals) * 100
    print(f"전체 승률: {baseline_wr:.1f}% | 승리: {len(wins)}건 | 패배: {len(losses)}건\n")

    features = [
        ("vol_spike_ratio", "투매성 거래량 스파이크(최근5일/이전20일)"),
        ("dist_from_sma20", "20일선 대비 이격도(%)"),
        ("repeat_count", "최근20일 내 다이버전스 반복횟수"),
        ("today_candle_pct", "신호당일 캔들 방향(종가-시가 %)"),
        ("decline_speed", "5일간 RSI백분위 하락속도"),
    ]

    print("[특성별 승리 vs 패배 그룹 비교 - 차이가 클수록 결정적 요인]")
    diffs = []
    for key, label in features:
        win_stat = summarize([s[key] for s in wins])
        loss_stat = summarize([s[key] for s in losses])
        if win_stat is None or loss_stat is None:
            continue
        win_mean, win_med, win_n = win_stat
        loss_mean, loss_med, loss_n = loss_stat
        diff = win_mean - loss_mean
        diffs.append((label, win_mean, win_n, loss_mean, loss_n, diff))

    diffs.sort(key=lambda x: abs(x[5]), reverse=True)
    for label, win_mean, win_n, loss_mean, loss_n, diff in diffs:
        print(f"  {label}")
        print(f"    승리평균 {win_mean:.2f}(n={win_n}) vs 패배평균 {loss_mean:.2f}(n={loss_n}) | 차이 {diff:+.2f}")

    print("\n해석: 차이가 크고 표본이 충분한 특성이 승패를 가르는 실질적 조건일 가능성이 높음.")


if __name__ == "__main__":
    main()
