"""
divergence_only_backtest.py
사용법: python divergence_only_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: RSI/MACD 다이버전스가 발생한 날들만 골라서(게이트 조건),
      그 안에서 나머지 4개 바닥매수 근거
      (반전캔들, RSI백분위, 볼린저하단, 다이버전스 거래량확인)
      각각이 승률에 미치는 영향을 검증한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
]
SMALLCAP_TICKERS = [
    "RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
    "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ",
    "FUBO", "PATH", "ASAN", "BBAI", "SOUN", "RUN", "ENVX", "JOBY", "ACHR", "LAZR",
    "QS", "FSLY", "PLUG", "NIO", "XPEV", "BYND", "WOLF", "GPRO", "DNA", "HOOD",
    "AI", "PTON", "SNAP", "LYFT", "DASH", "PINS", "ETSY", "BILL", "TTD", "ROKU",
]
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LARGECAP_SET = set(LARGECAP_TICKERS)
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


def find_divergence_signals(ticker, benchmark_close):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    rsi = RSIIndicator(c, window=14).rsi()
    bb = BollingerBands(c, window=20, window_dev=2)
    bb_low = bb.bollinger_lband()
    macd_hist = MACD(c).macd_diff()

    # 벤치마크(IWM)를 이 종목 날짜 인덱스에 맞춰 정렬
    bench_aligned = benchmark_close.reindex(c.index, method="ffill")

    # RSI 롤링 백분위 시계열 미리 계산 (시차 결합용)
    rsi_pctile_series = pd.Series(index=rsi.index, dtype=float)
    for j in range(252, len(rsi)):
        window = rsi.iloc[j-252:j].dropna()
        if len(window) > 0:
            rsi_pctile_series.iloc[j] = (window < rsi.iloc[j]).mean() * 100

    n = len(hist)
    start = max(MIN_HISTORY, n - LOOKBACK_DAYS - FORWARD_DAYS)
    end = n - FORWARD_DAYS

    results = []
    for i in range(start, end):
        recent = c.iloc[max(0, i-90):i+1]
        recent_rsi = rsi.iloc[max(0, i-90):i+1]
        if len(recent) < 90:
            continue

        low1_idx = recent.iloc[:45].idxmin()
        low2_idx = recent.iloc[45:].idxmin()
        bullish_divergence = bool(c[low2_idx] < c[low1_idx] and rsi[low2_idx] > rsi[low1_idx])

        mh = macd_hist.iloc[max(0, i-90):i+1]
        if len(mh) < 90:
            continue
        macd_rising = bool(mh.iloc[-1] > mh.iloc[:-1].min())

        if not bullish_divergence:
            continue

        sub_o, sub_h, sub_l, sub_c = o.iloc[:i+1], h.iloc[:i+1], l.iloc[:i+1], c.iloc[:i+1]
        has_trigger_candle = trigger_with_breakout(sub_o, sub_h, sub_l, sub_c)

        rsi_series = rsi.iloc[max(0, i-252):i+1].dropna()
        rsi_last = float(rsi.iloc[i])
        rsi_pctile = float((rsi_series < rsi_last).mean() * 100) if len(rsi_series) > 0 else 50.0
        rsi_oversold = rsi_pctile <= 15  # 기존 이진 판정도 유지(참고용 출력에 사용)
        rsi_strong = rsi_pctile <= 10
        rsi_weak = 10 < rsi_pctile <= 15

        # 시차 결합: 최근 10일(오늘 포함) 안에 rsi_strong이 한 번이라도 있었는지
        window_pctile = rsi_pctile_series.iloc[max(0, i-9):i+1]
        recent_rsi_strong = bool((window_pctile <= 10).any())

        bb_low_now = float(bb_low.iloc[i])
        near_bb = float(c.iloc[i]) <= bb_low_now * 1.05

        vol1 = float(v[low1_idx])
        vol2 = float(v[low2_idx])
        vol_confirmed = bool(vol2 < vol1)

        # 상대강도(RS): 최근 20일 이 종목 수익률 - 벤치마크(IWM) 수익률
        rs_positive = False
        if i >= 20 and not pd.isna(bench_aligned.iloc[i]) and not pd.isna(bench_aligned.iloc[i-20]):
            stock_ret_20d = float(c.iloc[i]) / float(c.iloc[i-20]) - 1
            bench_ret_20d = float(bench_aligned.iloc[i]) / float(bench_aligned.iloc[i-20]) - 1
            rs_positive = bool(stock_ret_20d > bench_ret_20d)

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        results.append({
            "ticker": ticker,
            "is_large": ticker in LARGECAP_SET,
            "macd_rising": macd_rising,
            "trigger_candle": has_trigger_candle,
            "rsi_oversold": rsi_oversold,
            "rsi_strong": rsi_strong,
            "rsi_weak": rsi_weak,
            "recent_rsi_strong": recent_rsi_strong,
            "recent_rsi_strong_and_candle": bool(recent_rsi_strong and has_trigger_candle),
            "rsi_and_candle_same_day": bool(rsi_oversold and has_trigger_candle),
            "rsi_and_bb_same_day": bool(rsi_oversold and near_bb),
            "rsi_and_candle_and_bb": bool(rsi_oversold and has_trigger_candle and near_bb),
            "near_bb": near_bb,
            "vol_confirmed": vol_confirmed,
            "rs_positive": rs_positive,
            "fwd_return": fwd_return,
        })
    return results


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def analyze_group(signals, group_label, exclude_factor=None):
    baseline_wr, baseline_avg = win_rate(signals)
    if baseline_wr is None:
        print(f"=== {group_label}: 표본 없음 ===\n")
        return
    print(f"=== {group_label} ({len(signals)}건, {FORWARD_DAYS}영업일 뒤 기준) ===")
    print(f"기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    factors = [
        ("macd_rising", "MACD 히스토그램 상승"),
        ("trigger_candle", "반전캔들+돌파 확정"),
        ("rsi_oversold", "RSI 백분위 15% 이하"),
        ("near_bb", "볼린저 하단 근접"),
        ("vol_confirmed", "다이버전스 거래량 확인(2번째저점 거래량<1번째)"),
        ("rs_positive", "상대강도(RS, 최근20일 IWM 대비 초과수익)"),
        ("recent_rsi_strong_and_candle", "시차결합(최근10일내 RSI10%이하 + 오늘 반전캔들)"),
    ]

    for key, label in factors:
        if key == exclude_factor:
            print(f"  [{label}] -- 이 그룹에서 제외됨 --\n")
            continue
        yes = [s for s in signals if s[key]]
        no = [s for s in signals if not s[key]]
        wr_yes, avg_yes = win_rate(yes)
        wr_no, avg_no = win_rate(no)
        print(f"  [{label}]")
        if wr_yes is not None:
            gap = wr_yes - baseline_wr
            print(f"    충족: 표본 {len(yes)}일 | 승률 {wr_yes:.1f}% | 평균수익률 {avg_yes:+.2f}% | 기준선 대비 {gap:+.1f}%p")
        if wr_no is not None:
            gap = wr_no - baseline_wr
            print(f"    미충족: 표본 {len(no)}일 | 승률 {wr_no:.1f}% | 평균수익률 {avg_no:+.2f}% | 기준선 대비 {gap:+.1f}%p")
    print()

    # 우선순위 기반 가중치 점수식으로 실제 테스트
    if exclude_factor == "near_bb":  # 대형주
        weights = {"vol_confirmed": 2.5, "rsi_oversold": 2.0, "trigger_candle": 1.0}
        total_w = sum(weights.values())
        for s in signals:
            raw = sum(w for k, w in weights.items() if s[k])
            s["combo_score"] = round(raw / total_w * 100)
    elif exclude_factor == "vol_confirmed":  # 중소형주 - 같은날 RSI+캔들/볼린저 결합 검증
        n_rsi_candle = sum(1 for s in signals if s["rsi_and_candle_same_day"])
        n_rsi_bb = sum(1 for s in signals if s["rsi_and_bb_same_day"])
        n_all3 = sum(1 for s in signals if s["rsi_and_candle_and_bb"])
        print(f"  [진단] RSI+캔들 동시(같은날): {n_rsi_candle}건, RSI+볼린저 동시: {n_rsi_bb}건, 셋다: {n_all3}건")
        for key in ["rsi_and_candle_same_day", "rsi_and_bb_same_day", "rsi_and_candle_and_bb"]:
            sub = [s for s in signals if s[key]]
            wr, avg = win_rate(sub)
            if wr is not None:
                gap = wr - baseline_wr
                print(f"    [{key}] 표본 {len(sub)}일 | 승률 {wr:.1f}% | 기준선 대비 {gap:+.1f}%p")
        print()
        weights = {"rsi_and_candle_same_day": 3.0, "rsi_oversold": 1.0, "near_bb": 0.5}
        total_w = sum(weights.values())
        for s in signals:
            raw = sum(w for k, w in weights.items() if s[k])
            s["combo_score"] = round(raw / total_w * 100)
    else:
        weights = {"vol_confirmed": 2.5, "rsi_oversold": 2.0, "trigger_candle": 1.0, "near_bb": 0.5}
        total_w = sum(weights.values())
        for s in signals:
            raw = sum(w for k, w in weights.items() if s[k])
            s["combo_score"] = round(raw / total_w * 100)

    if exclude_factor == "vol_confirmed":  # 중소형주는 85점 구간 추가로 세분화
        very_high = [s for s in signals if s["combo_score"] >= 85]
        high = [s for s in signals if 65 <= s["combo_score"] < 85]
        mid = [s for s in signals if 40 <= s["combo_score"] < 65]
        low = [s for s in signals if s["combo_score"] < 40]
        bands = [("85점 이상", very_high), ("65~84점", high), ("40~64점", mid), ("40점 미만", low)]
    else:
        high = [s for s in signals if s["combo_score"] >= 65]
        mid = [s for s in signals if 40 <= s["combo_score"] < 65]
        low = [s for s in signals if s["combo_score"] < 40]
        bands = [("65점 이상", high), ("40~64점", mid), ("40점 미만", low)]

    print(f"  [가중치 점수식: {weights}]")
    for band_label, recs in bands:
        wr, avg = win_rate(recs)
        if wr is None:
            print(f"    {band_label}: 표본 없음")
            continue
        gap = wr - baseline_wr
        print(f"    {band_label}: 표본 {len(recs)}일 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}% | 기준선 대비 {gap:+.1f}%p")
    print()


def main():
    print("벤치마크(IWM, 러셀2000 ETF) 데이터 로딩 중...")
    benchmark_close = yf.Ticker("IWM").history(period="2y")["Close"]

    all_signals = []
    for t in TICKERS:
        recs = find_divergence_signals(t, benchmark_close)
        if recs is None:
            print(f"{t}: 데이터 없음")
            continue
        all_signals.extend(recs)
        print(f"{t}: 다이버전스 신호 {len(recs)}건")

    if not all_signals:
        print("신호 없음")
        return

    large_signals = [s for s in all_signals if s["is_large"]]
    small_signals = [s for s in all_signals if not s["is_large"]]

    print()
    analyze_group(all_signals, "전체(대형주+중소형주 통합)")
    analyze_group(large_signals, "대형주만 (볼린저하단 제외)", exclude_factor="near_bb")
    analyze_group(small_signals, "중소형주만 (거래량확인 제외)", exclude_factor="vol_confirmed")

    print("해석: 다이버전스만으로 이미 필터링된 상태에서, 각 근거가 '충족'일 때")
    print("승률이 유의미하게 더 높으면 그 근거를 추가 조건으로 결합할 가치가 있음.")


if __name__ == "__main__":
    main()
