"""
fetch_indicators.py
사용법: python fetch_indicators.py TICKER
결과: data/TICKER.json 생성 (2~10단계 판정값)
추가로 python fetch_indicators.py --macro 실행 시 data/_macro.json 생성 (1단계용)
"""
import sys, os, json
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

SECTOR_ETF = {
    "Healthcare": "XLV", "Technology": "XLK", "Financial Services": "XLF",
    "Energy": "XLE", "Industrials": "XLI", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Utilities": "XLU", "Real Estate": "XLRE",
    "Basic Materials": "XLB", "Communication Services": "XLC",
}


def rs_line(ticker_close, bench_close):
    ratio = (ticker_close / bench_close).dropna()
    return ratio.iloc[-1] > ratio.iloc[-63]  # 약 3개월 전 대비 상대강도 상승


def detect_bullish_engulfing(o, h, l, c):
    return c.iloc[-1] > o.iloc[-2] and o.iloc[-1] < c.iloc[-2] and c.iloc[-1] > c.iloc[-2] and o.iloc[-2] > c.iloc[-2]


def detect_hammer(o, h, l, c):
    body = abs(c.iloc[-1] - o.iloc[-1])
    lower_wick = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
    upper_wick = h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])
    return body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5


def detect_morning_star(o, h, l, c):
    d1_bear = c.iloc[-3] < o.iloc[-3]
    d2_small = abs(c.iloc[-2] - o.iloc[-2]) < abs(c.iloc[-3] - o.iloc[-3]) * 0.4
    d3_bull = c.iloc[-1] > o.iloc[-1] and c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2
    return d1_bear and d2_small and d3_bull


def fetch_macro():
    spx = yf.Ticker("^GSPC").history(period="1y")["Close"]
    ndx = yf.Ticker("^IXIC").history(period="1y")["Close"]
    vix = yf.Ticker("^VIX").history(period="3mo")["Close"]

    spx_above_200sma = spx.iloc[-1] > spx.rolling(200).mean().iloc[-1]
    ndx_above_200sma = ndx.iloc[-1] > ndx.rolling(200).mean().iloc[-1]
    vix_last = vix.iloc[-1]
    vix_stable = vix_last < 20 and vix_last < vix.rolling(10).mean().iloc[-1]

    result = {
        "updated": pd.Timestamp.utcnow().isoformat(),
        "spx_above_200sma": bool(spx_above_200sma),
        "ndx_above_200sma": bool(ndx_above_200sma),
        "vix": round(float(vix_last), 2),
        "vix_stable": bool(vix_stable),
        "macro_status": "pass" if (spx_above_200sma and ndx_above_200sma and vix_stable) else "warn"
                        if (spx_above_200sma or ndx_above_200sma) else "fail",
    }
    os.makedirs("data", exist_ok=True)
    with open("data/_macro.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def analyze(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y")
    if hist.empty:
        raise ValueError(f"no price history for {ticker}")

    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]

    rsi = RSIIndicator(c, window=14).rsi()
    bb = BollingerBands(c, window=20, window_dev=2)
    macd = MACD(c)
    adx_ind = ADXIndicator(h, l, c, window=14)
    obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    atr = AverageTrueRange(h, l, c, window=14).average_true_range()

    last_close = float(c.iloc[-1])
    bb_low, bb_high = float(bb.bollinger_lband().iloc[-1]), float(bb.bollinger_hband().iloc[-1])
    rsi_last = float(rsi.iloc[-1])

    # 40일 구간에서 두 개의 저점 비교 -> RSI 불리시 다이버전스 근사 판정
    recent = c.iloc[-40:]
    low1_idx, low2_idx = recent.iloc[:20].idxmin(), recent.iloc[20:].idxmin()
    price_lower_low = c[low2_idx] < c[low1_idx]
    rsi_higher_low = rsi[low2_idx] > rsi[low1_idx]
    bullish_divergence = bool(price_lower_low and rsi_higher_low)

    macd_hist = macd.macd_diff()
    macd_hist_rising = bool(macd_hist.iloc[-1] > macd_hist.iloc[-40:-1].min() and macd_hist.iloc[-1] < 0)

    adx_last, adx_prev = float(adx_ind.adx().iloc[-1]), float(adx_ind.adx().iloc[-6])
    trend_exhaustion = "pass" if (adx_prev >= 25 and adx_last < adx_prev) else "unknown"

    obv_up_while_price_down = bool(obv.iloc[-1] > obv.iloc[-20] and c.iloc[-1] < c.iloc[-20])

    weekly = hist.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    weekly_rsi = RSIIndicator(weekly["Close"], window=14).rsi()
    mtf_aligned = bool(rsi_last < 45 and weekly_rsi.iloc[-1] < 55)

    hammer = detect_hammer(o, h, l, c)
    engulfing = detect_bullish_engulfing(o, h, l, c)
    morning_star = detect_morning_star(o, h, l, c) if len(c) >= 3 else False
    trigger_candle = bool(hammer or engulfing or morning_star)

    info = tk.info
    target_mean = info.get("targetMeanPrice")
    forward_pe = info.get("forwardPE")
    peg = info.get("pegRatio")
    sector = info.get("sector")
    upside_pct = round((target_mean / last_close - 1) * 100, 1) if target_mean else None

    sector_rs = None
    if sector in SECTOR_ETF:
        etf_close = yf.Ticker(SECTOR_ETF[sector]).history(period="1y")["Close"]
        sector_rs = bool(rs_line(c, etf_close))

    stage3 = "pass" if (last_close <= bb_low * 1.02 and rsi_last <= 35) else \
             "warn" if (last_close <= bb_low * 1.05 or rsi_last <= 40) else "fail"
    stage4 = "pass" if (bullish_divergence and macd_hist_rising) else \
             "warn" if (bullish_divergence or macd_hist_rising) else "fail"
    stage5 = "pass" if obv_up_while_price_down else "warn"
    stage7 = "pass" if mtf_aligned else "warn"
    stage8 = "pass" if trigger_candle else "fail"

    stop_pct = round(float(atr.iloc[-1]) * 1.5 / last_close * 100, 1)

    stage2_flags = {
        "upside_ge_15": (upside_pct or 0) >= 15,
        "peg_reasonable": (peg is not None and peg <= 2.5),
        "forward_pe_present": forward_pe is not None,
    }
    stage2 = "pass" if all(stage2_flags.values()) else \
             "fail" if not stage2_flags["upside_ge_15"] else "warn"

    result = {
        "ticker": ticker.upper(),
        "price": round(last_close, 2),
        "updated": pd.Timestamp.utcnow().isoformat(),
        "stages": {
            "2_fundamentals": {"status": stage2, "upside_pct": upside_pct, "forward_pe": forward_pe, "peg": peg,
                                "note": "발행주식수 증감/최근 다운그레이드 이력은 야후 정보만으로는 부족 - 수동 확인 권장"},
            "3_technical_position": {"status": stage3, "rsi": round(rsi_last, 1), "bb_low": round(bb_low, 2), "bb_high": round(bb_high, 2)},
            "4_divergence_momentum": {"status": stage4, "bullish_divergence": bullish_divergence, "macd_hist_rising": macd_hist_rising},
            "5_volume_flow": {"status": stage5, "obv_bullish_divergence": obv_up_while_price_down},
            "6_trend_exhaustion": {"status": trend_exhaustion, "adx": round(adx_last, 1)},
            "7_multi_timeframe": {"status": stage7, "daily_rsi": round(rsi_last, 1), "weekly_rsi": round(float(weekly_rsi.iloc[-1]), 1)},
            "8_trigger_candle": {"status": stage8, "hammer": bool(hammer), "bullish_engulfing": bool(engulfing), "morning_star": bool(morning_star)},
            "9_position_sizing": {"atr_stop_pct": stop_pct},
            "10_target": {"target_mean_price": target_mean, "upside_pct": upside_pct},
        },
        "sector": sector,
        "sector_relative_strength_up": sector_rs,
    }
    os.makedirs("data", exist_ok=True)
    with open(f"data/{ticker.upper()}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python fetch_indicators.py TICKER | --macro")
        sys.exit(1)
    if sys.argv[1] == "--macro":
        fetch_macro()
    else:
        analyze(sys.argv[1])
