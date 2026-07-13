"""
pullback_v3_backtest.py
사용법: python pullback_v3_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식(눌림목 v3 - 상승채널 + OBV):
  게이트(전부 충족):
    1. 최근스윙고점 > 직전스윙고점 (Higher High)
    2. 최근스윙저점 > 직전스윙저점 (Higher Low)
    3. 직전스윙고점 -> 최근스윙저점 구간에 하루 -5%+ 급락 캔들 없음
    4. 종가 > 40일선
    5. OBV 최근스윙저점 > OBV 직전스윙저점 (매수세 추세 안 꺾임)
  참고: 트리거캔들(해머/엔걸핑/모닝스타+돌파, 또는 긴아래꼬리) 여부
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import RSIIndicator

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SWING_ORDER = 5
CRASH_THRESHOLD = -0.05  # 하루 -5% 이상 급락


def find_swing_lows(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n - order):
        window = close_values[i-order:i+order+1]
        if close_values[i] == window.min() and (window == close_values[i]).sum() == 1:
            positions.append(i)
    return positions


def find_swing_highs(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n - order):
        window = close_values[i-order:i+order+1]
        if close_values[i] == window.max() and (window == close_values[i]).sum() == 1:
            positions.append(i)
    return positions


def detect_hammer(o, h, l, c, i):
    body = abs(c.iloc[i] - o.iloc[i])
    lower_wick = min(o.iloc[i], c.iloc[i]) - l.iloc[i]
    upper_wick = h.iloc[i] - max(o.iloc[i], c.iloc[i])
    return bool(body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5)


def detect_bullish_engulfing(o, h, l, c, i):
    if i < 1:
        return False
    return bool(c.iloc[i] > o.iloc[i-1] and o.iloc[i] < c.iloc[i-1] and c.iloc[i] > c.iloc[i-1] and o.iloc[i-1] > c.iloc[i-1])


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 300:
        return None
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    close_values = c.values
    daily_ret = c.pct_change()
    sma40 = c.rolling(40).mean()
    sma20 = c.rolling(20).mean()
    sma60 = c.rolling(60).mean()
    rsi = RSIIndicator(c, window=14).rsi()
    obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    obv_values = obv.values
    n = len(hist)

    swing_lows_full = find_swing_lows(close_values, order=SWING_ORDER)
    swing_highs_full = find_swing_highs(close_values, order=SWING_ORDER)
    swing_lows_obv_full = find_swing_lows(obv_values, order=SWING_ORDER)

    rows = []
    start = 300
    end = n - FORWARD_DAYS
    for i in range(start, end):
        lows_upto = [p for p in swing_lows_full if p + SWING_ORDER <= i]
        highs_upto = [p for p in swing_highs_full if p + SWING_ORDER <= i]
        obv_lows_upto = [p for p in swing_lows_obv_full if p + SWING_ORDER <= i]

        if len(lows_upto) < 2 or len(highs_upto) < 2 or len(obv_lows_upto) < 2:
            continue

        recent_low, prior_low = lows_upto[-1], lows_upto[-2]
        recent_high, prior_high = highs_upto[-1], highs_upto[-2]

        # 신저점이 "오늘 막 확정된 첫날"만 신호로 인정 (필수 게이트 - 진입시점 자체를 정의)
        is_freshly_confirmed = (i == recent_low + SWING_ORDER)
        if not is_freshly_confirmed:
            continue

        cond1_higher_high = bool(close_values[recent_high] > close_values[prior_high])
        cond2_higher_low = bool(close_values[recent_low] > close_values[prior_low])

        cond3_no_crash = None
        if prior_high < recent_low:
            segment_ret = daily_ret.iloc[prior_high+1:recent_low+1]
            cond3_no_crash = bool((segment_ret >= CRASH_THRESHOLD).all()) if len(segment_ret) > 0 else True

        sma40_now = float(sma40.iloc[i]) if not pd.isna(sma40.iloc[i]) else None
        cond4_above_sma40 = bool(sma40_now is not None and close_values[i] > sma40_now)
        dist_sma40_pct = ((close_values[i] - sma40_now) / sma40_now * 100) if sma40_now else None

        sma40_20ago = float(sma40.iloc[i-20]) if i >= 20 and not pd.isna(sma40.iloc[i-20]) else None
        sma40_uptrend = bool(sma40_now is not None and sma40_20ago is not None and sma40_now > sma40_20ago)

        # 20일선/60일선 근접 여부(±3% 이내) + 그 시점 RSI
        MA_TOLERANCE = 0.03
        sma20_now = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else None
        sma60_now = float(sma60.iloc[i]) if not pd.isna(sma60.iloc[i]) else None
        near_sma20 = bool(sma20_now is not None and abs(close_values[i] - sma20_now) / sma20_now <= MA_TOLERANCE)
        near_sma60 = bool(sma60_now is not None and abs(close_values[i] - sma60_now) / sma60_now <= MA_TOLERANCE)
        near_any_ma = near_sma20 or near_sma60
        rsi_now = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else None
        rsi_above_50 = bool(rsi_now is not None and rsi_now >= 50)

        recent_obv_low, prior_obv_low = obv_lows_upto[-1], obv_lows_upto[-2]
        cond5_obv_rising = bool(obv_values[recent_obv_low] > obv_values[prior_obv_low])

        # 최근 N일간 OBV 기울기(연속값) - 스윙비교가 아니라 단순 최근 추세
        obv_slope_10 = None
        obv_slope_20 = None
        if i >= 10 and not np.isnan(obv_values[i-10]):
            obv_slope_10 = obv_values[i] - obv_values[i-10]
        if i >= 20 and not np.isnan(obv_values[i-20]):
            obv_slope_20 = obv_values[i] - obv_values[i-20]
        # 정규화: OBV 절대값 스케일이 종목마다 다르므로, 최근 60일 OBV 표준편차로 나눠서 비교 가능하게 함
        obv_window = obv_values[max(0, i-60):i+1]
        obv_std_60 = float(np.std(obv_window)) if len(obv_window) > 10 else None
        obv_slope_10_norm = (obv_slope_10 / obv_std_60) if (obv_slope_10 is not None and obv_std_60) else None

        # 트리거캔들: 신저점 확정일 기준 최근 1~3일 소급 확인(신규진입과 동일 방식)
        has_trigger = False
        for lb in range(0, 4):
            idx = i - lb
            if idx < 3:
                continue
            if detect_hammer(o, h, l, c, idx) or detect_bullish_engulfing(o, h, l, c, idx):
                has_trigger = True
                break

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        rows.append({
            "cond1_higher_high": cond1_higher_high,
            "cond2_higher_low": cond2_higher_low,
            "cond3_no_crash": cond3_no_crash,
            "cond4_above_sma40": cond4_above_sma40,
            "dist_sma40_pct": dist_sma40_pct,
            "sma40_uptrend": sma40_uptrend,
            "near_any_ma": near_any_ma, "near_sma20": near_sma20, "near_sma60": near_sma60,
            "rsi_now": rsi_now, "rsi_above_50": rsi_above_50,
            "cond5_obv_rising": cond5_obv_rising,
            "obv_slope_10_norm": obv_slope_10_norm,
            "has_trigger": has_trigger,
            "fwd_return": fwd_return, "win": fwd_return > 0,
        })
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_rows = []
    for t in LARGECAP_TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)

    print(f"\n=== 신저점 확정 시점(기본 게이트만) 총 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"기준선(조건 미적용): 승률 {baseline_wr:.1f}% | 평균수익률 {baseline_avg:+.2f}%\n")

    conditions = [
        ("cond1_higher_high", "1. 고점 우상향(Higher High)"),
        ("cond2_higher_low", "2. 저점 우상향(Higher Low)"),
        ("cond3_no_crash", "3. 급락캔들(-5%+) 없음"),
        ("cond4_above_sma40", "4. 40일선 위"),
        ("cond5_obv_rising", "5. OBV 저점 우상향"),
        ("has_trigger", "6. 트리거캔들(최근1~3일)"),
    ]
    diffs = []
    for key, label in conditions:
        yes = [r for r in all_rows if r[key] is True]
        no = [r for r in all_rows if r[key] is False]
        wr_y, avg_y = win_rate(yes)
        wr_n, avg_n = win_rate(no)
        print(f"[{label}]")
        if wr_y is not None:
            print(f"  충족: 표본 {len(yes)} | 승률 {wr_y:.1f}% | 기준대비 {wr_y-baseline_wr:+.1f}%p")
        if wr_n is not None:
            print(f"  미충족: 표본 {len(no)} | 승률 {wr_n:.1f}% | 기준대비 {wr_n-baseline_wr:+.1f}%p")
        if wr_y is not None:
            diffs.append((label, wr_y - baseline_wr, len(yes)))

    print("\n=== 40일선 이격도(%) 구간별 발생빈도 + 승률 (전체구간) ===")
    dist_rows = [r for r in all_rows if r["dist_sma40_pct"] is not None]
    total_n = len(dist_rows)
    for lo, hi in [(-999, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 999)]:
        sub = [r for r in dist_rows if lo <= r["dist_sma40_pct"] < hi]
        wr, avg = win_rate(sub)
        freq_pct = len(sub) / total_n * 100 if total_n else 0
        lo_l = str(lo) if lo > -999 else "~"
        hi_l = str(hi) if hi < 999 else "이상"
        if wr is not None:
            print(f"  {lo_l}~{hi_l}%: 표본 {len(sub)}건({freq_pct:.1f}%) | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n=== 40일선 이격도(%) 구간별 x 트리거캔들 유무 교차분석 ===")
    for lo, hi in [(-999, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 999)]:
        lo_l = str(lo) if lo > -999 else "~"
        hi_l = str(hi) if hi < 999 else "이상"
        for trig_val, trig_label in [(True, "트리거O"), (False, "트리거X")]:
            sub = [r for r in dist_rows if lo <= r["dist_sma40_pct"] < hi and r["has_trigger"] == trig_val]
            wr, avg = win_rate(sub)
            freq_pct = len(sub) / total_n * 100 if total_n else 0
            if wr is not None:
                print(f"  {lo_l}~{hi_l}% + {trig_label}: 표본 {len(sub)}건({freq_pct:.1f}%) | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print("\n=== 40일선 자체 기울기(20일전 대비 우상향인지) - 진짜 상승추세 확인 ===")
    up_slope = [r for r in all_rows if r["sma40_uptrend"] is True]
    down_slope = [r for r in all_rows if r["sma40_uptrend"] is False]
    wr_u, avg_u = win_rate(up_slope)
    wr_d, avg_d = win_rate(down_slope)
    if wr_u is not None:
        print(f"  40일선 우상향: 표본 {len(up_slope)} | 승률 {wr_u:.1f}% | 기준대비 {wr_u-baseline_wr:+.1f}%p")
    if wr_d is not None:
        print(f"  40일선 우하향/횡보: 표본 {len(down_slope)} | 승률 {wr_d:.1f}% | 기준대비 {wr_d-baseline_wr:+.1f}%p")

    print("\n=== OBV 최근 10일 기울기(정규화, 60일표준편차 단위) 구간별 승률 ===")
    obv_rows = [r for r in all_rows if r["obv_slope_10_norm"] is not None]
    obv_total = len(obv_rows)
    for lo, hi in [(-999, -1.0), (-1.0, -0.3), (-0.3, 0), (0, 0.3), (0.3, 1.0), (1.0, 999)]:
        sub = [r for r in obv_rows if lo <= r["obv_slope_10_norm"] < hi]
        wr, avg = win_rate(sub)
        freq_pct = len(sub) / obv_total * 100 if obv_total else 0
        lo_l = str(lo) if lo > -999 else "~"
        hi_l = str(hi) if hi < 999 else "이상"
        if wr is not None:
            print(f"  {lo_l}~{hi_l}: 표본 {len(sub)}건({freq_pct:.1f}%) | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p | 평균수익률 {avg:+.2f}%")

    print("\n=== OBV기울기(단순 양수/음수)만으로 이분 ===")
    obv_pos = [r for r in obv_rows if r["obv_slope_10_norm"] > 0]
    obv_neg = [r for r in obv_rows if r["obv_slope_10_norm"] <= 0]
    wr_p, avg_p = win_rate(obv_pos)
    wr_ng, avg_ng = win_rate(obv_neg)
    if wr_p is not None:
        print(f"  양수(상승중): 표본 {len(obv_pos)} | 승률 {wr_p:.1f}% | 기준대비 {wr_p-baseline_wr:+.1f}%p")
    if wr_ng is not None:
        print(f"  음수(하락중): 표본 {len(obv_neg)} | 승률 {wr_ng:.1f}% | 기준대비 {wr_ng-baseline_wr:+.1f}%p")

    print("\n=== 20일선/60일선 근접(±3%) 여부 ===")
    near = [r for r in all_rows if r["near_any_ma"]]
    far = [r for r in all_rows if not r["near_any_ma"]]
    wr_near, avg_near = win_rate(near)
    wr_far, avg_far = win_rate(far)
    if wr_near is not None:
        print(f"  근접: 표본 {len(near)} | 승률 {wr_near:.1f}% | 기준대비 {wr_near-baseline_wr:+.1f}%p")
    if wr_far is not None:
        print(f"  비근접: 표본 {len(far)} | 승률 {wr_far:.1f}% | 기준대비 {wr_far-baseline_wr:+.1f}%p")

    print("\n=== 핵심 검증: 20/60일선 근접 + RSI50이상 조합 ===")
    combos = [
        ("근접+RSI50이상", lambda r: r["near_any_ma"] and r["rsi_above_50"]),
        ("근접+RSI50미만", lambda r: r["near_any_ma"] and not r["rsi_above_50"]),
        ("비근접+RSI50이상", lambda r: not r["near_any_ma"] and r["rsi_above_50"]),
        ("비근접+RSI50미만", lambda r: not r["near_any_ma"] and not r["rsi_above_50"]),
    ]
    for label, cond in combos:
        sub = [r for r in all_rows if r["rsi_now"] is not None and cond(r)]
        wr, avg = win_rate(sub)
        if wr is not None:
            print(f"  {label}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p | 평균수익률 {avg:+.2f}%")

    print("\n=== 근접+RSI50이상 구간에서 트리거캔들 유무 ===")
    target = [r for r in all_rows if r["near_any_ma"] and r["rsi_above_50"]]
    t_yes = [r for r in target if r["has_trigger"]]
    t_no = [r for r in target if not r["has_trigger"]]
    wr_ty, avg_ty = win_rate(t_yes)
    wr_tn, avg_tn = win_rate(t_no)
    if wr_ty is not None:
        print(f"  트리거O: 표본 {len(t_yes)} | 승률 {wr_ty:.1f}% | 기준대비 {wr_ty-baseline_wr:+.1f}%p")
    if wr_tn is not None:
        print(f"  트리거X: 표본 {len(t_no)} | 승률 {wr_tn:.1f}% | 기준대비 {wr_tn-baseline_wr:+.1f}%p")

    print("\n=== 기여도(충족시 승률 개선폭) 순위 - 1~6번 조건 ===")
    diffs.sort(key=lambda x: x[1], reverse=True)
    for label, diff, n_sample in diffs:
        print(f"  {label}: {diff:+.1f}%p (표본 {n_sample})")


if __name__ == "__main__":
    main()
