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

    print(f"\n=== 중소형주 다이버전스 신호 {len(all_signals)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    baseline_wr, baseline_avg = win_rate(all_signals)
    print(f"기준선: 승률 {baseline_wr:.1f}%\n")

    bool_factors = [
        ("triple_divergence", "삼중 다이버전스"),
        ("held_the_low", "저점유지(안 깸)"),
        ("trending_down", "궤적 하락중"),
        ("vol_divergence", "거래량 다이버전스(저점간)"),
        ("mtf_ok", "멀티타임프레임 정배열"),
        ("has_trigger", "반전캔들+돌파"),
        ("rs_ok", "상대강도(vs SPY) 우위"),
    ]
    for key, label in bool_factors:
        yes = [s for s in all_signals if s[key] is True]
        no = [s for s in all_signals if s[key] is False]
        wr_y, avg_y = win_rate(yes)
        wr_n, avg_n = win_rate(no)
        print(f"[{label}]")
        if wr_y is not None:
            print(f"  충족: 표본 {len(yes)} | 승률 {wr_y:.1f}% | 기준대비 {wr_y-baseline_wr:+.1f}%p")
        if wr_n is not None:
            print(f"  미충족: 표본 {len(no)} | 승률 {wr_n:.1f}% | 기준대비 {wr_n-baseline_wr:+.1f}%p")

    print("\n[반복횟수 구간별]")
    for lo, hi in [(0,3),(3,6),(6,10),(10,999)]:
        sub = [s for s in all_signals if lo <= s["repeat_count"] < hi]
        wr, avg = win_rate(sub)
        if wr: print(f"  {lo}~{hi if hi<999 else '+'}회: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[경과일수 구간별]")
    for lo, hi in [(0,3),(3,5),(5,7),(7,10),(10,15),(15,999)]:
        sub = [s for s in all_signals if lo <= s["days_since_low2"] < hi]
        wr, avg = win_rate(sub)
        if wr: print(f"  {lo}~{hi if hi<999 else '+'}일: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[ADX 구간별]")
    for lo, hi in [(0,15),(15,20),(20,25),(25,30),(30,999)]:
        sub = [s for s in all_signals if s["adx_last"] is not None and lo <= s["adx_last"] < hi]
        wr, avg = win_rate(sub)
        if wr: print(f"  {lo}~{hi if hi<999 else '+'}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n[거래량스파이크 구간별]")
    for lo, hi in [(0,0.8),(0.8,1.0),(1.0,1.3),(1.3,1.7),(1.7,999)]:
        sub = [s for s in all_signals if not (isinstance(s["vol_spike"],float) and np.isnan(s["vol_spike"])) and lo <= s["vol_spike"] < hi]
        wr, avg = win_rate(sub)
        if wr: print(f"  {lo}~{hi if hi<999 else '+'}배: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n해석: 기준대비 차이가 크고 표본이 충분한 항목이 실제로 유의미한 조건.")


if __name__ == "__main__":
    main()
