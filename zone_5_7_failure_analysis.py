"""
full_stage_analysis_largecap.py
사용법: python full_stage_analysis_largecap.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 대형주에서 다이버전스가 발생한 날들을 모아서, 모든 세부 지표를
      계산하고 각각의 승률을 비교한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator

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


def detect_bullish_engulfing(o, h, l, c):
    return c.iloc[-1] > o.iloc[-2] and o.iloc[-1] < c.iloc[-2] and c.iloc[-1] > c.iloc[-2] and o.iloc[-2] > c.iloc[-2]


def detect_hammer(o, h, l, c):
    body = abs(c.iloc[-1] - o.iloc[-1])
    lower_wick = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
    upper_wick = h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])
    return body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5


def detect_morning_star(o, h, l, c):
    if len(c) < 3:
        return False
    d1_bear = c.iloc[-3] < o.iloc[-3]
    d2_small = abs(c.iloc[-2] - o.iloc[-2]) < abs(c.iloc[-3] - o.iloc[-3]) * 0.4
    d3_bull = c.iloc[-1] > o.iloc[-1] and c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2
    return d1_bear and d2_small and d3_bull


def trigger_with_breakout(o, h, l, c):
    n = len(c)
    for lookback in range(1, 4):
        idx = n - 1 - lookback
        if idx < 3:
            continue
        sub_o, sub_h, sub_l, sub_c = o.iloc[:idx+1], h.iloc[:idx+1], l.iloc[:idx+1], c.iloc[:idx+1]
        is_hammer = detect_hammer(sub_o, sub_h, sub_l, sub_c)
        is_engulf = detect_bullish_engulfing(sub_o, sub_h, sub_l, sub_c)
        is_star = detect_morning_star(sub_o, sub_h, sub_l, sub_c)
        if (is_hammer or is_engulf or is_star) and c.iloc[-1] > h.iloc[idx]:
            return True
    return False


def rolling_pctile_series(rsi, window=252):
    result = pd.Series(index=rsi.index, dtype=float)
    for j in range(window, len(rsi)):
        w = rsi.iloc[j-window:j]
        result.iloc[j] = (w < rsi.iloc[j]).mean() * 100
    return result


def find_signals(ticker, benchmark_close):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < MIN_HISTORY + 20:
        return None
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    rsi = RSIIndicator(c, window=14).rsi()
    macd_hist = MACD(c).macd_diff()
    adx_ind = ADXIndicator(h, l, c, window=14)
    adx_series = adx_ind.adx()
    sma20 = c.rolling(20).mean()
    pctile_series = rolling_pctile_series(rsi)
    weekly = hist.resample("W").agg({"Close": "last"}).dropna()
    weekly_rsi = RSIIndicator(weekly["Close"], window=14).rsi()
    bench_aligned = benchmark_close.reindex(c.index, method="ffill")

    n = len(hist)
    start = max(MIN_HISTORY, n - LOOKBACK_DAYS - FORWARD_DAYS)
    end = n - FORWARD_DAYS

    div_positions = []
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
        div_positions.append(i)

        third = recent.iloc[:30].idxmin(), recent.iloc[30:60].idxmin(), recent.iloc[60:].idxmin()
        triple_divergence = bool(rsi[third[0]] < rsi[third[1]] < rsi[third[2]] and c[third[0]] > c[third[1]] > c[third[2]])

        rsi_slope = float(rsi[low2_idx] - rsi[low1_idx])
        price_drop_ratio = float((c[low1_idx] - c[low2_idx]) / c[low1_idx] * 100)

        low2_pos = hist.index.get_loc(low2_idx)
        days_since_low2 = i - low2_pos
        low2_price = float(c[low2_idx])
        subsequent_lows = l.iloc[low2_pos+1:i+1]
        held_the_low = bool((subsequent_lows >= low2_price).all()) if len(subsequent_lows) > 0 else True

        pctile_now = pctile_series.iloc[i] if not pd.isna(pctile_series.iloc[i]) else None
        pctile_5ago = pctile_series.iloc[i-5] if i >= 5 and not pd.isna(pctile_series.iloc[i-5]) else None
        trending_down = bool(pctile_now is not None and pctile_5ago is not None and pctile_now < pctile_5ago)
        decline_speed = (pctile_5ago - pctile_now) if (pctile_now is not None and pctile_5ago is not None) else None

        repeat_count = sum(1 for p in div_positions if i - 20 <= p < i)

        vol1 = float(v[low1_idx]); vol2 = float(v[low2_idx])
        vol_divergence = bool(vol2 < vol1)
        vol_recent5 = float(v.iloc[max(0,i-5):i].mean()) if i >= 5 else float("nan")
        vol_prior20 = float(v.iloc[max(0,i-25):max(0,i-5)].mean()) if i >= 25 else float("nan")
        vol_spike = vol_recent5/vol_prior20 if vol_prior20 and vol_prior20>0 else float("nan")

        adx_last = float(adx_series.iloc[i]) if not pd.isna(adx_series.iloc[i]) else None
        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        dist_sma20 = (float(c.iloc[i])-sma20_now)/sma20_now*100 if sma20_now else float("nan")

        daily_rsi_ok = bool(rsi.iloc[i] < 45)
        w_pos = weekly.index.get_indexer([hist.index[i]], method="ffill")[0]
        weekly_rsi_ok = bool(w_pos>=0 and w_pos < len(weekly_rsi) and not pd.isna(weekly_rsi.iloc[w_pos]) and weekly_rsi.iloc[w_pos] < 55)

        sub_o, sub_h, sub_l, sub_c = o.iloc[:i+1], h.iloc[:i+1], l.iloc[:i+1], c.iloc[:i+1]
        has_trigger = trigger_with_breakout(sub_o, sub_h, sub_l, sub_c)

        rs_ok = None
        if i>=20 and not pd.isna(bench_aligned.iloc[i]) and not pd.isna(bench_aligned.iloc[i-20]):
            stock_ret = float(c.iloc[i])/float(c.iloc[i-20])-1
            bench_ret = float(bench_aligned.iloc[i])/float(bench_aligned.iloc[i-20])-1
            rs_ok = bool(stock_ret > bench_ret)

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i+FORWARD_DAYS])/entry - 1

        results.append({
            "triple_divergence": triple_divergence, "rsi_slope": rsi_slope, "price_drop_ratio": price_drop_ratio,
            "days_since_low2": days_since_low2, "held_the_low": held_the_low,
            "trending_down": trending_down, "decline_speed": decline_speed,
            "repeat_count": repeat_count, "vol_divergence": vol_divergence, "vol_spike": vol_spike,
            "adx_last": adx_last, "dist_sma20": dist_sma20,
            "mtf_ok": daily_rsi_ok and weekly_rsi_ok, "has_trigger": has_trigger, "rs_ok": rs_ok,
            "fwd_return": fwd_return,
        })
    return results


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr>0).mean()*100), float(arr.mean()*100)


def summarize(vals):
    arr = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if len(arr) == 0:
        return None
    return float(arr.mean()), float(np.median(arr)), len(arr)


def main():
    print("벤치마크(IWM) 로딩 중...")
    bench = yf.Ticker("IWM").history(period="2y")["Close"]

    all_signals = []
    for t in SMALLCAP_TICKERS:
        try:
            recs = find_signals(t, bench)
        except Exception as e:
            print(f"{t}: 에러로 스킵 ({e})")
            continue
        if recs:
            all_signals.extend(recs)
        print(f"{t}: {len(recs) if recs else 0}건")

    zone_5_7 = [s for s in all_signals if 5 <= s["days_since_low2"] < 7]
    print(f"\n=== 5~7일 구간 신호 {len(zone_5_7)}건 ===\n")
    if not zone_5_7:
        print("표본 없음")
        return

    wins = [s for s in zone_5_7 if s["fwd_return"] > 0]
    losses = [s for s in zone_5_7 if s["fwd_return"] <= 0]
    baseline_wr = len(wins) / len(zone_5_7) * 100
    print(f"5~7일 구간 승률: {baseline_wr:.1f}% | 승리 {len(wins)}건 | 패배 {len(losses)}건\n")

    features = [
        ("triple_divergence", "삼중 다이버전스(0/1)"),
        ("rsi_slope", "RSI 기울기"),
        ("price_drop_ratio", "가격 하락폭(%)"),
        ("repeat_count", "다이버전스 반복횟수"),
        ("adx_last", "ADX"),
        ("dist_sma20", "20일선 대비 이격도(%)"),
        ("vol_spike", "거래량스파이크(최근5일/이전20일)"),
        ("decline_speed", "5일간 RSI백분위 하락속도"),
    ]

    print("[특성별 승리 vs 패배 평균 비교 - 차이 큰 순]")
    diffs = []
    for key, label in features:
        win_stat = summarize([s[key] for s in wins])
        loss_stat = summarize([s[key] for s in losses])
        if win_stat is None or loss_stat is None:
            continue
        win_mean, win_med, win_n = win_stat
        loss_mean, loss_med, loss_n = loss_stat
        diffs.append((label, win_mean, win_n, loss_mean, loss_n, win_mean - loss_mean))
    diffs.sort(key=lambda x: abs(x[5]), reverse=True)
    for label, win_mean, win_n, loss_mean, loss_n, diff in diffs:
        print(f"  {label}: 승리 {win_mean:.2f}(n={win_n}) vs 패배 {loss_mean:.2f}(n={loss_n}) | 차이 {diff:+.2f}")

    print("\n[불리언 특성 - 승리/패배 그룹 내 비율]")
    bool_features = [
        ("held_the_low", "저점유지(안 깸)"),
        ("trending_down", "궤적 하락중"),
        ("vol_divergence", "거래량 다이버전스(저점간)"),
        ("mtf_ok", "멀티타임프레임 정배열"),
        ("has_trigger", "반전캔들+돌파"),
        ("rs_ok", "상대강도 우위"),
    ]
    for key, label in bool_features:
        win_true = sum(1 for s in wins if s[key] is True)
        loss_true = sum(1 for s in losses if s[key] is True)
        win_pct = win_true / len(wins) * 100 if wins else 0
        loss_pct = loss_true / len(losses) * 100 if losses else 0
        print(f"  {label}: 승리그룹 {win_pct:.0f}%({win_true}/{len(wins)}) vs 패배그룹 {loss_pct:.0f}%({loss_true}/{len(losses)})")

    print("\n해석: 차이가 크고 표본이 충분한 항목이 5~7일 구간 내 승패를 가르는 실질적 원인.")


if __name__ == "__main__":
    main()
