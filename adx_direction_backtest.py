"""
adx_direction_backtest.py
사용법: python adx_direction_backtest.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

계산식(2~4단계 게이트 통과 신호에 대해):
  게이트: 진짜 스윙저점 다이버전스(간격3~30일, 오차범위 포함)
  ADX 절대값과 별개로, "5일 전 대비 ADX가 오르는 중인지"를 추가로 확인.
  특히 ADX<22(현재 fail 구간)를 "회복중"과 "계속하락"으로 나눠
  회복중일 때 승률이 실제로 더 좋은지 검증.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

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
]
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MIN_GAP_DAYS = 3
MAX_GAP_DAYS = 30
TOLERANCE_PCT = 0.03


def find_realtime_lows(close_values, order=5):
    positions = []
    n = len(close_values)
    for i in range(order, n):
        past_window = close_values[i - order:i]
        if close_values[i] < past_window.min():
            positions.append(i)
    return positions


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 280:
        return None
    h, l, c = hist["High"], hist["Low"], hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    adx_ind = ADXIndicator(h, l, c, window=14)
    adx_series = adx_ind.adx()
    plus_di_series = adx_ind.adx_pos()
    minus_di_series = adx_ind.adx_neg()
    close_values = c.values
    n = len(hist)

    realtime_lows_full = find_realtime_lows(close_values, order=5)

    rows = []
    # 매일(i) 시점에서, 그 시점까지의 데이터만으로 저점을 판정(과거 재현)
    lows_upto = []
    li = 0
    for i in range(260, n - FORWARD_DAYS):
        while li < len(realtime_lows_full) and realtime_lows_full[li] <= i:
            lows_upto.append(realtime_lows_full[li])
            li += 1
        if len(lows_upto) < 2:
            continue
        pos2 = lows_upto[-1]
        candidates = [p for p in lows_upto if p != pos2 and MIN_GAP_DAYS <= (pos2 - p) <= MAX_GAP_DAYS]
        if not candidates:
            continue
        pos1 = min(candidates, key=lambda p: close_values[p])
        is_today_new_low = (pos2 == i)
        price2 = float(c.iloc[pos2])
        price_today = float(c.iloc[i])
        within_tol = bool(price_today >= price2 and (price_today - price2) / price2 <= TOLERANCE_PCT)
        if not (is_today_new_low or within_tol):
            continue

        price1 = float(c.iloc[pos1])
        rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
        if pd.isna(rsi1) or pd.isna(rsi2):
            continue
        if not (price2 < price1 and rsi2 > rsi1):
            continue  # pass 케이스만(정석 다이버전스) 대상으로 ADX 방향 검증

        adx_today = float(adx_series.iloc[i]) if not pd.isna(adx_series.iloc[i]) else None
        adx1 = float(adx_series.iloc[pos1]) if not pd.isna(adx_series.iloc[pos1]) else None
        adx2 = float(adx_series.iloc[pos2]) if not pd.isna(adx_series.iloc[pos2]) else None
        if adx_today is None or adx1 is None or adx2 is None:
            continue
        # 다이버전스 구간(저점1~저점2) 내부에서만 ADX 변화를 봄
        adx_declining_within = adx2 < adx1  # 정석 이론: RSI개선+ADX하락이 추세소진 신호
        adx_change_within = adx2 - adx1

        # ADX 종목별 개별 Z-score (참고용으로 계속 유지)
        adx_window = adx_series.iloc[max(0, i - 252):i + 1].dropna()
        adx_zscore = None
        if len(adx_window) >= 60:
            adx_mean = float(adx_window.mean())
            adx_std = float(adx_window.std())
            if adx_std > 0:
                adx_zscore = (adx_today - adx_mean) / adx_std

        # 가격 하락폭 대비 RSI 개선폭의 비율 - "기울기 균형" 가설 검증용
        price_drop_pct = (price1 - price2) / price1 * 100
        rsi_rise = rsi2 - rsi1
        slope_ratio = (rsi_rise / price_drop_pct) if price_drop_pct > 0 else None

        # +DI/-DI 방향 지표 (오늘 시점, 그리고 구간 내 변화)
        plus_di_today = float(plus_di_series.iloc[i]) if not pd.isna(plus_di_series.iloc[i]) else None
        minus_di_today = float(minus_di_series.iloc[i]) if not pd.isna(minus_di_series.iloc[i]) else None
        plus_di_1 = float(plus_di_series.iloc[pos1]) if not pd.isna(plus_di_series.iloc[pos1]) else None
        minus_di_1 = float(minus_di_series.iloc[pos1]) if not pd.isna(minus_di_series.iloc[pos1]) else None
        plus_di_2 = float(plus_di_series.iloc[pos2]) if not pd.isna(plus_di_series.iloc[pos2]) else None
        minus_di_2 = float(minus_di_series.iloc[pos2]) if not pd.isna(minus_di_series.iloc[pos2]) else None

        di_bullish_today = None
        di_gap_improving = None
        di_just_crossed_up = None
        if plus_di_today is not None and minus_di_today is not None:
            di_bullish_today = plus_di_today > minus_di_today
        if None not in (plus_di_1, minus_di_1, plus_di_2, minus_di_2):
            di_gap_1 = plus_di_1 - minus_di_1
            di_gap_2 = plus_di_2 - minus_di_2
            di_gap_improving = di_gap_2 > di_gap_1
            di_just_crossed_up = bool(di_gap_1 <= 0 and di_gap_2 > 0)

        entry = float(c.iloc[i])
        fwd_return = float(c.iloc[i + FORWARD_DAYS]) / entry - 1

        rows.append({"adx": adx_today, "adx_zscore": adx_zscore,
                     "adx_declining_within": adx_declining_within, "adx_change_within": adx_change_within,
                     "price_drop_pct": price_drop_pct, "rsi_rise": rsi_rise, "slope_ratio": slope_ratio,
                     "di_bullish_today": di_bullish_today, "di_gap_improving": di_gap_improving,
                     "di_just_crossed_up": di_just_crossed_up,
                     "fwd_return": fwd_return, "win": fwd_return > 0})
    return rows


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_rows = []
    for t in TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 스킵 ({e})")
            continue
        if rows:
            all_rows.extend(rows)

    print(f"\n=== 정석 다이버전스(PASS) 신호 총 {len(all_rows)}건, {FORWARD_DAYS}일 뒤 기준 ===\n")
    if not all_rows:
        print("표본 없음")
        return

    baseline_wr, baseline_avg = win_rate(all_rows)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%\n")

    print(f"=== 핵심 검증: 다이버전스 구간(저점1~저점2) 내부에서 ADX 변화 방향 ===")
    print("(정석 이론: RSI는 개선되는데 ADX는 하락 = 추세소진 신호 = 좋은 신호일 것)")
    declining = [r for r in all_rows if r["adx_declining_within"]]
    rising = [r for r in all_rows if not r["adx_declining_within"]]
    wr_d, avg_d = win_rate(declining)
    wr_r, avg_r = win_rate(rising)
    if wr_d is not None:
        print(f"  구간내 ADX 하락(저점2<저점1, 추세소진 이론과 일치): 표본 {len(declining)}건 | 승률 {wr_d:.1f}% | 기준대비 {wr_d-baseline_wr:+.1f}%p | 평균수익률 {avg_d:+.2f}%")
    if wr_r is not None:
        print(f"  구간내 ADX 상승(저점2>=저점1, 추세소진 이론과 불일치): 표본 {len(rising)}건 | 승률 {wr_r:.1f}% | 기준대비 {wr_r-baseline_wr:+.1f}%p | 평균수익률 {avg_r:+.2f}%")

    print(f"\n=== 참고: ADX>=22 구간(현재 pass/warn) ===")
    high_adx = [r for r in all_rows if r["adx"] >= 22]
    wr_h, avg_h = win_rate(high_adx)
    if wr_h is not None:
        print(f"전체: 표본 {len(high_adx)}건 | 승률 {wr_h:.1f}% | 기준대비 {wr_h-baseline_wr:+.1f}%p")

    print(f"\n=== +DI/-DI 검증 1: 오늘(저점2) 시점에 +DI가 -DI보다 큰지 (매수압력 우세) ===")
    di_rows = [r for r in all_rows if r["di_bullish_today"] is not None]
    bullish = [r for r in di_rows if r["di_bullish_today"]]
    bearish = [r for r in di_rows if not r["di_bullish_today"]]
    wr_b, avg_b = win_rate(bullish)
    wr_be, avg_be = win_rate(bearish)
    if wr_b is not None:
        print(f"  +DI>-DI(매수압력 우세): 표본 {len(bullish)}건 | 승률 {wr_b:.1f}% | 기준대비 {wr_b-baseline_wr:+.1f}%p")
    if wr_be is not None:
        print(f"  +DI<=-DI(매도압력 우세): 표본 {len(bearish)}건 | 승률 {wr_be:.1f}% | 기준대비 {wr_be-baseline_wr:+.1f}%p")

    print(f"\n=== +DI/-DI 검증 2: 다이버전스 구간(저점1~저점2) 동안 (+DI-(-DI)) 격차가 개선됐는지 ===")
    gap_rows = [r for r in all_rows if r["di_gap_improving"] is not None]
    improving = [r for r in gap_rows if r["di_gap_improving"]]
    worsening = [r for r in gap_rows if not r["di_gap_improving"]]
    wr_i, avg_i = win_rate(improving)
    wr_w, avg_w = win_rate(worsening)
    if wr_i is not None:
        print(f"  격차 개선(매수압력 강해지는중): 표본 {len(improving)}건 | 승률 {wr_i:.1f}% | 기준대비 {wr_i-baseline_wr:+.1f}%p")
    if wr_w is not None:
        print(f"  격차 악화(매수압력 약해지는중): 표본 {len(worsening)}건 | 승률 {wr_w:.1f}% | 기준대비 {wr_w-baseline_wr:+.1f}%p")

    print(f"\n=== +DI/-DI 검증 3: 다이버전스 구간 내에서 -DI우세→+DI우세로 골든크로스 발생 여부 ===")
    cross_rows = [r for r in all_rows if r["di_just_crossed_up"] is not None]
    crossed = [r for r in cross_rows if r["di_just_crossed_up"]]
    not_crossed = [r for r in cross_rows if not r["di_just_crossed_up"]]
    wr_c, avg_c = win_rate(crossed)
    wr_n, avg_n = win_rate(not_crossed)
    if wr_c is not None:
        print(f"  골든크로스 발생: 표본 {len(crossed)}건 | 승률 {wr_c:.1f}% | 기준대비 {wr_c-baseline_wr:+.1f}%p")
    if wr_n is not None:
        print(f"  골든크로스 없음: 표본 {len(not_crossed)}건 | 승률 {wr_n:.1f}% | 기준대비 {wr_n-baseline_wr:+.1f}%p")

    print(f"\n=== ADX Z-score(종목별 개별계산) 구간별 승률 ===")
    z_rows = [r for r in all_rows if r["adx_zscore"] is not None]
    for lo, hi in [(-999, -1.0), (-1.0, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 999)]:
        sub = [r for r in z_rows if lo <= r["adx_zscore"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            lo_l = str(lo) if lo > -999 else "~"
            hi_l = str(hi) if hi < 999 else "이상"
            print(f"  {lo_l}~{hi_l}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")

    print(f"\n=== 가격하락폭 대비 RSI개선폭 비율(slope_ratio) 구간별 승률 ===")
    print("(비율이 낮음=가격만 많이 빠지고 RSI는 조금 개선 / 비율이 높음=가격은 조금 빠졌는데 RSI는 많이 개선)")
    sr_rows = [r for r in all_rows if r["slope_ratio"] is not None]
    ratios = np.array([r["slope_ratio"] for r in sr_rows])
    print(f"분포: 평균 {ratios.mean():.2f} | 중앙값 {np.median(ratios):.2f} | 5~95% 범위 {np.percentile(ratios,5):.2f}~{np.percentile(ratios,95):.2f}\n")
    for lo, hi in [(-999, 1), (1, 2), (2, 4), (4, 7), (7, 12), (12, 999)]:
        sub = [r for r in sr_rows if lo <= r["slope_ratio"] < hi]
        wr, avg = win_rate(sub)
        if wr is not None:
            lo_l = str(lo) if lo > -999 else "~"
            hi_l = str(hi) if hi < 999 else "이상"
            print(f"  {lo_l}~{hi_l}: 표본 {len(sub)} | 승률 {wr:.1f}% | 기준대비 {wr-baseline_wr:+.1f}%p")


if __name__ == "__main__":
    main()
