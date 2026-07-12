"""
score_win_rate_backtest.py
사용법: python score_win_rate_backtest.py [FORWARD_DAYS] [LOOKBACK_DAYS]
기본값: FORWARD_DAYS=10, LOOKBACK_DAYS=150

동작: 10개 종목 각각에 대해, 최근 LOOKBACK_DAYS 거래일 매일 그 시점까지의
      데이터만으로 신규진입 점수/추매 점수를 재구성한다. 그 점수가 65 이상
      (통과 구간)이었던 날 이후 FORWARD_DAYS 거래일 뒤 수익률을 계산해서
      승률/평균수익률을 산출하고, 전체 기준선과 비교한다.

한계: 2단계(펀더멘털)는 과거 애널리스트 목표가를 구할 수 없어 'warn' 고정.
      기술적 단계(3~8단계)만으로 재구성한 근사 검증.
"""
import sys, os
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_indicators import (
    detect_bullish_engulfing, detect_hammer, detect_morning_star,
    trigger_with_breakout, WEIGHTS, ADDON_WEIGHTS, STATUS_SCORE
)

TICKERS = ["NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
           "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH"]
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LOOKBACK_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
MIN_HISTORY = 260


def score_at(hist_full, cutoff_idx):
    hist = hist_full.iloc[:cutoff_idx+1]
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]

    rsi = RSIIndicator(c, window=14).rsi()
    bb = BollingerBands(c, window=20, window_dev=2)
    macd = MACD(c)
    adx_ind = ADXIndicator(h, l, c, window=14)
    obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()

    last_close = float(c.iloc[-1])
    bb_low = float(bb.bollinger_lband().iloc[-1])
    rsi_last = float(rsi.iloc[-1])

    rsi_series_52w = rsi.dropna()
    rsi_percentile = float((rsi_series_52w < rsi_last).mean() * 100) if len(rsi_series_52w) > 0 else 50.0
    rsi_low = rsi_percentile <= 15
    rsi_mid = 15 < rsi_percentile <= 30

    stage3 = "pass" if (last_close <= bb_low * 1.02 and rsi_low) else \
             "warn" if (last_close <= bb_low * 1.05 or rsi_mid) else "fail"

    recent = c.iloc[-90:]
    if len(recent) >= 90:
        low1_idx, low2_idx = recent.iloc[:45].idxmin(), recent.iloc[45:].idxmin()
        bullish_divergence = bool(c[low2_idx] < c[low1_idx] and rsi[low2_idx] > rsi[low1_idx])
    else:
        bullish_divergence = False
    macd_hist = macd.macd_diff()
    macd_hist_rising = bool(macd_hist.iloc[-1] > macd_hist.iloc[-90:-1].min() and macd_hist.iloc[-1] < 0) if len(macd_hist) > 90 else False
    stage4 = "pass" if (bullish_divergence and macd_hist_rising) else \
             "warn" if (bullish_divergence or macd_hist_rising) else "fail"

    stage5 = "pass" if bool(obv.iloc[-1] > obv.iloc[-20] and c.iloc[-1] < c.iloc[-20]) else "warn"

    adx_last, adx_prev = float(adx_ind.adx().iloc[-1]), float(adx_ind.adx().iloc[-6])
    trend_exhaustion = "pass" if (adx_prev >= 25 and adx_last < adx_prev) else "unknown"

    weekly = hist.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    weekly_rsi = RSIIndicator(weekly["Close"], window=14).rsi()
    mtf_aligned = bool(rsi_last < 45 and weekly_rsi.iloc[-1] < 55) if len(weekly_rsi.dropna()) > 0 else False
    stage7 = "pass" if mtf_aligned else "warn"

    breakout_ok, _, _ = trigger_with_breakout(o, h, l, c)
    stage8 = "pass" if breakout_ok else "fail"

    entry_stages = {
        "2_fundamentals": "warn", "3_technical_position": stage3, "4_divergence_momentum": stage4,
        "5_volume_flow": stage5, "6_trend_exhaustion": trend_exhaustion,
        "7_multi_timeframe": stage7, "8_trigger_candle": stage8,
    }
    known = {k: WEIGHTS[k] for k in WEIGHTS if entry_stages[k] != "unknown"}
    entry_score = round(sum(known[k] * STATUS_SCORE[entry_stages[k]] for k in known) / sum(known.values()) * 100) if known else 0

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    plus_di = adx_ind.adx_pos()
    minus_di = adx_ind.adx_neg()
    near_20 = abs(last_close - sma20.iloc[-1]) / sma20.iloc[-1] <= 0.02
    near_50 = abs(last_close - sma50.iloc[-1]) / sma50.iloc[-1] <= 0.02
    addon_stage3 = "pass" if (rsi_low and (near_20 or near_50)) else \
                   "warn" if (rsi_low or (rsi_mid and (near_20 or near_50))) else "fail"

    recent20_high = float(h.iloc[-20:].max())
    pullback_pct = (recent20_high - last_close) / recent20_high * 100
    # 뒤집음: 깊은 조정(>10%)을 pass로, 얕은 조정(<3%)을 fail로
    addon_stage4 = "pass" if pullback_pct > 10 else "warn" if 3 <= pullback_pct <= 10 else "fail"

    vol_recent5 = float(v.iloc[-5:].mean())
    vol_prior5 = float(v.iloc[-10:-5].mean())
    addon_stage5 = "pass" if vol_recent5 > vol_prior5 else "warn"

    recent5_high = float(h.iloc[-6:-1].max())
    # 뒤집음: 돌파 "안 한" 상태(아직 고점 아래, 눌린 채로 대기)를 pass로
    addon_stage8 = "fail" if bool(c.iloc[-1] > recent5_high and c.iloc[-1] > o.iloc[-1]) else "pass"

    addon_stages = {
        "3_pullback_position": addon_stage3, "4_pullback_depth": addon_stage4,
        "5_volume_flow": addon_stage5, "8_trigger_breakout": addon_stage8,
    }
    known_a = {k: ADDON_WEIGHTS[k] for k in ADDON_WEIGHTS if k in addon_stages and addon_stages[k] != "unknown"}
    addon_score = round(sum(known_a[k] * STATUS_SCORE[addon_stages[k]] for k in known_a) / sum(known_a.values()) * 100) if known_a else 0

    return entry_score, addon_score, addon_stages


def build_records(ticker):
    hist_full = yf.Ticker(ticker).history(period="2y")
    if hist_full.empty:
        return None
    n = len(hist_full)
    records = []
    start = max(MIN_HISTORY, n - LOOKBACK_DAYS - FORWARD_DAYS)
    end = n - FORWARD_DAYS
    for cutoff_idx in range(start, end):
        entry_score, addon_score, addon_stages = score_at(hist_full, cutoff_idx)
        close_now = float(hist_full["Close"].iloc[cutoff_idx])
        close_fwd = float(hist_full["Close"].iloc[cutoff_idx + FORWARD_DAYS])
        fwd_return = close_fwd / close_now - 1
        records.append({"ticker": ticker, "entry_score": entry_score, "addon_score": addon_score,
                         "addon_stages": addon_stages, "fwd_return": fwd_return})
    return records


def win_rate(recs):
    if not recs:
        return None, None
    arr = np.array([r["fwd_return"] for r in recs])
    return float((arr > 0).mean() * 100), float(arr.mean() * 100)


def main():
    all_records = []
    for t in TICKERS:
        recs = build_records(t)
        if recs is None:
            print(f"{t}: 데이터 없음")
            continue
        all_records.extend(recs)
        print(f"{t}: {len(recs)}일 재구성 완료")

    if not all_records:
        print("표본 없음")
        return

    print(f"\n=== 전체 {len(all_records)}개 표본, {FORWARD_DAYS}영업일 뒤 수익률 기준 ===\n")

    baseline_wr, baseline_avg = win_rate(all_records)
    print(f"전체 기준선: 승률 {baseline_wr:.1f}%, 평균수익률 {baseline_avg:+.2f}%\n")

    for label, key in [("추매 점수", "addon_score")]:
        high = [r for r in all_records if r[key] >= 65]
        mid = [r for r in all_records if 40 <= r[key] < 65]
        low = [r for r in all_records if r[key] < 40]

        print(f"--- {label} (합산 점수 기준) ---")
        for band_label, recs in [("65점 이상(pass)", high), ("40~64점(warn)", mid), ("40점 미만(fail)", low)]:
            wr, avg = win_rate(recs)
            if wr is None:
                print(f"  {band_label}: 표본 없음")
                continue
            gap = wr - baseline_wr
            print(f"  {band_label}: 표본 {len(recs)}일 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}% | 기준선 대비 {gap:+.1f}%p")
        print()

    # ── 단계별(2~9) 개별 pass/warn/fail 승률 ──
    print("=== 추매 단계별(2~9) pass/warn/fail 승률 — 어느 단계가 실제로 유효한지 확인 ===\n")
    stage_keys = ["3_pullback_position", "4_pullback_depth", "5_volume_flow", "8_trigger_breakout"]
    for sk in stage_keys:
        print(f"[{sk}]")
        for status in ["pass", "warn", "fail"]:
            recs = [r for r in all_records if r["addon_stages"].get(sk) == status]
            wr, avg = win_rate(recs)
            if wr is None:
                print(f"  {status}: 표본 없음")
                continue
            gap = wr - baseline_wr
            print(f"  {status}: 표본 {len(recs)}일 | 승률 {wr:.1f}% | 평균수익률 {avg:+.2f}% | 기준선 대비 {gap:+.1f}%p")
        print()


if __name__ == "__main__":
    main()
