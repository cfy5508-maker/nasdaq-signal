"""
check_nvda_score_history.py
사용법: python check_nvda_score_history.py

동작: NVDA에 대해 지난 5거래일 각각의 시점으로 돌아가서, 그 시점까지의
      데이터만으로 기술적 단계(3~8단계, 추매 3~8단계)를 재계산하고
      점수 변화를 출력한다.

한계: 2단계(펀더멘털 - 애널리스트 목표가, PEG 등)는 과거 시점 값을 구할
      방법이 없어 현재 값으로 고정한다. 순수 기술적 지표 검증용.
"""
import sys, os
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# fetch_indicators.py와 동일한 함수 재사용
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_indicators import (
    detect_bullish_engulfing, detect_hammer, detect_morning_star,
    trigger_with_breakout, WEIGHTS, ADDON_WEIGHTS, STATUS_SCORE
)

TICKER = "NVDA"
N_DAYS = 5


def analyze_at(hist_full, cutoff_idx):
    """hist_full의 0~cutoff_idx까지만 사용해서 그 시점 기준 점수를 계산."""
    hist = hist_full.iloc[:cutoff_idx+1]
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]

    rsi = RSIIndicator(c, window=14).rsi()
    bb = BollingerBands(c, window=20, window_dev=2)
    macd = MACD(c)
    adx_ind = ADXIndicator(h, l, c, window=14)
    obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    atr = AverageTrueRange(h, l, c, window=14).average_true_range()

    last_close = float(c.iloc[-1])
    bb_low = float(bb.bollinger_lband().iloc[-1])
    rsi_last = float(rsi.iloc[-1])

    rsi_series_52w = rsi.dropna()
    rsi_percentile = float((rsi_series_52w < rsi_last).mean() * 100) if len(rsi_series_52w) > 0 else 50.0
    rsi_low = rsi_percentile <= 15
    rsi_mid = 15 < rsi_percentile <= 30

    # 신규진입 3단계
    stage3 = "pass" if (last_close <= bb_low * 1.02 and rsi_low) else \
             "warn" if (last_close <= bb_low * 1.05 or rsi_mid) else "fail"

    # 4단계 다이버전스 (90일)
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

    # 2단계는 현재값 고정 (한계)
    stage2 = "warn"

    entry_stages = {
        "2_fundamentals": stage2, "3_technical_position": stage3, "4_divergence_momentum": stage4,
        "5_volume_flow": stage5, "6_trend_exhaustion": trend_exhaustion,
        "7_multi_timeframe": stage7, "8_trigger_candle": stage8,
    }
    known = {k: WEIGHTS[k] for k in WEIGHTS if entry_stages[k] != "unknown"}
    entry_score = round(sum(known[k] * STATUS_SCORE[entry_stages[k]] for k in known) / sum(known.values()) * 100) if known else 0

    # ── 추매 ──
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
    addon_stage4 = "pass" if 3 <= pullback_pct <= 10 else "warn" if pullback_pct < 3 else "fail"

    vol_recent5 = float(v.iloc[-5:].mean())
    vol_prior5 = float(v.iloc[-10:-5].mean())
    addon_stage5 = "pass" if vol_recent5 > vol_prior5 else "warn"

    addon_stage6 = "pass" if bool(adx_last >= 20 and plus_di.iloc[-1] > minus_di.iloc[-1]) else "fail"

    weekly_sma20 = weekly["Close"].rolling(20).mean()
    daily_above_sma20 = bool(last_close > sma20.iloc[-1])
    weekly_above_sma20 = bool(len(weekly_sma20.dropna()) > 0 and weekly["Close"].iloc[-1] > weekly_sma20.iloc[-1])
    addon_stage7 = "pass" if (daily_above_sma20 and weekly_above_sma20) else "warn" if daily_above_sma20 else "fail"

    recent5_high = float(h.iloc[-6:-1].max())
    addon_stage8 = "pass" if bool(c.iloc[-1] > recent5_high and c.iloc[-1] > o.iloc[-1]) else "fail"

    addon_stages = {
        "2_fundamentals": stage2, "3_pullback_position": addon_stage3, "4_pullback_depth": addon_stage4,
        "5_volume_flow": addon_stage5, "6_trend_continuation": addon_stage6,
        "7_multi_timeframe": addon_stage7, "8_trigger_breakout": addon_stage8,
    }
    known_a = {k: ADDON_WEIGHTS[k] for k in ADDON_WEIGHTS if addon_stages[k] != "unknown"}
    addon_score = round(sum(known_a[k] * STATUS_SCORE[addon_stages[k]] for k in known_a) / sum(known_a.values()) * 100) if known_a else 0

    return {
        "date": hist.index[-1].strftime("%Y-%m-%d"),
        "close": round(last_close, 2),
        "rsi": round(rsi_last, 1),
        "rsi_percentile": round(rsi_percentile, 1),
        "entry_score": entry_score,
        "entry_stages": entry_stages,
        "addon_score": addon_score,
        "addon_stages": addon_stages,
    }


def main():
    hist_full = yf.Ticker(TICKER).history(period="2y")
    if hist_full.empty:
        print("데이터 없음")
        return

    n = len(hist_full)
    print(f"=== {TICKER} 지난 {N_DAYS}거래일 점수 재구성 ===")
    print("(2단계 펀더멘털은 과거값 조회 불가로 'warn' 고정 — 기술적 지표 검증용)\n")

    for i in range(N_DAYS, 0, -1):
        cutoff_idx = n - i
        if cutoff_idx < 260:  # 최소 1년치는 있어야 지표 계산 안정적
            continue
        r = analyze_at(hist_full, cutoff_idx)
        print(f"[{r['date']}] 종가 {r['close']} | RSI {r['rsi']} (백분위 {r['rsi_percentile']}%)")
        print(f"  신규진입 점수: {r['entry_score']:>3}  | " + ", ".join(f"{k.split('_',1)[1]}:{v}" for k, v in r['entry_stages'].items() if k != '2_fundamentals'))
        print(f"  추매 점수:    {r['addon_score']:>3}  | " + ", ".join(f"{k.split('_',1)[1]}:{v}" for k, v in r['addon_stages'].items() if k != '2_fundamentals'))
        print()


if __name__ == "__main__":
    main()
