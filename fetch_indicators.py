"""
fetch_indicators.py
사용법: python fetch_indicators.py
동작: data/watchlist.json에 등록된 모든 티커를 순회하며
      2~10단계 체크리스트를 계산해 data/<TICKER>.json에 저장하고,
      점수를 매겨 data/rankings.json(순위표)를 생성한다.

watchlist.json 형식: ["GILD", "JOBY", "AMSC", ...]
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

WATCHLIST_PATH = "data/watchlist.json"
OUT_DIR = "data"

SECTOR_ETF = {
    "Healthcare": "XLV", "Technology": "XLK", "Financial Services": "XLF",
    "Energy": "XLE", "Industrials": "XLI", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Utilities": "XLU", "Real Estate": "XLRE",
    "Basic Materials": "XLB", "Communication Services": "XLC",
}

DOW30 = {
    "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
    "GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK",
    "MSFT","NKE","NVDA","PG","SHW","TRV","UNH","V","VZ","WMT"
}

_INDEX_CACHE = {}
_SP500_SET = None


def get_sp500_set():
    global _SP500_SET
    if _SP500_SET is not None:
        return _SP500_SET
    cache_path = os.path.join(OUT_DIR, "sp500_constituents.json")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        symbols = set(tables[0]["Symbol"].str.replace(".", "-", regex=False))
        with open(cache_path, "w") as f:
            json.dump(sorted(symbols), f)
        _SP500_SET = symbols
    except Exception:
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                _SP500_SET = set(json.load(f))
        else:
            _SP500_SET = set()
    return _SP500_SET


def determine_index(ticker, exchange):
    if exchange in ("NMS", "NGM", "NCM"):
        return ("나스닥", "^IXIC")
    if ticker in get_sp500_set():
        return ("S&P500", "^GSPC")
    if ticker in DOW30:
        return ("다우30", "^DJI")
    return (None, None)


def index_macro_status(index_symbol):
    if index_symbol in _INDEX_CACHE:
        return _INDEX_CACHE[index_symbol]
    try:
        hist = yf.Ticker(index_symbol).history(period="1y")["Close"]
        above_200sma = bool(hist.iloc[-1] > hist.rolling(200).mean().iloc[-1])
    except Exception:
        above_200sma = None
    result = "pass" if above_200sma else ("fail" if above_200sma is False else "unknown")
    _INDEX_CACHE[index_symbol] = result
    return result

WEIGHTS = {
    "2_fundamentals": 2.0,
    "3_technical_position": 1.5,
    "4_divergence_momentum": 1.5,
    "5_volume_flow": 1.0,
    "6_trend_exhaustion": 1.0,
    "7_multi_timeframe": 1.0,
    "8_trigger_candle": 1.5,
}
STATUS_SCORE = {"pass": 1.0, "warn": 0.5, "fail": 0.0, "unknown": 0.25}
MAX_SCORE = sum(WEIGHTS.values())


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


def trigger_with_breakout(o, h, l, c):
    """최근 1~3일 안에 반전캔들이 확정되고, 그 이후 종가가 그 캔들의 고점을 돌파했는지 확인."""
    n = len(c)
    for lookback in range(1, 4):
        idx = n - 1 - lookback
        if idx < 3:
            continue
        sub_o, sub_h, sub_l, sub_c = o.iloc[:idx+1], h.iloc[:idx+1], l.iloc[:idx+1], c.iloc[:idx+1]
        is_hammer = detect_hammer(sub_o, sub_h, sub_l, sub_c)
        is_engulf = detect_bullish_engulfing(sub_o, sub_h, sub_l, sub_c)
        is_star = detect_morning_star(sub_o, sub_h, sub_l, sub_c) if idx + 1 >= 3 else False
        if (is_hammer or is_engulf or is_star) and c.iloc[-1] > h.iloc[idx]:
            pattern_name = "해머" if is_hammer else ("불리시엔걸핑" if is_engulf else "모닝스타")
            return True, pattern_name, lookback
    return False, None, None


def rs_line(ticker_close, bench_close):
    ratio = (ticker_close / bench_close).dropna()
    if len(ratio) < 64:
        return None
    return bool(ratio.iloc[-1] > ratio.iloc[-63])


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

    recent = c.iloc[-90:]
    low1_idx, low2_idx = recent.iloc[:45].idxmin(), recent.iloc[45:].idxmin()
    price_lower_low = c[low2_idx] < c[low1_idx]
    rsi_higher_low = rsi[low2_idx] > rsi[low1_idx]
    bullish_divergence = bool(price_lower_low and rsi_higher_low)

    macd_hist = macd.macd_diff()
    macd_hist_rising = bool(macd_hist.iloc[-1] > macd_hist.iloc[-90:-1].min() and macd_hist.iloc[-1] < 0)

    adx_last, adx_prev = float(adx_ind.adx().iloc[-1]), float(adx_ind.adx().iloc[-6])
    trend_exhaustion = "pass" if (adx_prev >= 25 and adx_last < adx_prev) else "unknown"

    obv_up_while_price_down = bool(obv.iloc[-1] > obv.iloc[-20] and c.iloc[-1] < c.iloc[-20])

    weekly = hist.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    weekly_rsi = RSIIndicator(weekly["Close"], window=14).rsi()
    mtf_aligned = bool(rsi_last < 45 and weekly_rsi.iloc[-1] < 55)

    breakout_ok, breakout_pattern, breakout_days_ago = trigger_with_breakout(o, h, l, c)
    hammer = detect_hammer(o, h, l, c)
    engulfing = detect_bullish_engulfing(o, h, l, c)
    morning_star = detect_morning_star(o, h, l, c) if len(c) >= 3 else False

    info = {}
    try:
        info = tk.info
    except Exception:
        pass

    exchange = info.get("exchange")
    index_name, index_symbol = determine_index(ticker.upper(), exchange)
    stage1_status = index_macro_status(index_symbol) if index_symbol else "unknown"

    target_mean = info.get("targetMeanPrice")
    forward_pe = info.get("forwardPE")
    peg = info.get("pegRatio")
    sector = info.get("sector")
    upside_pct = round((target_mean / last_close - 1) * 100, 1) if target_mean else None

    sector_rs = None
    if sector in SECTOR_ETF:
        try:
            etf_close = yf.Ticker(SECTOR_ETF[sector]).history(period="1y")["Close"]
            sector_rs = rs_line(c, etf_close)
        except Exception:
            pass

    stage3 = "pass" if (last_close <= bb_low * 1.02 and rsi_last <= 35) else \
             "warn" if (last_close <= bb_low * 1.05 or rsi_last <= 40) else "fail"
    stage4 = "pass" if (bullish_divergence and macd_hist_rising) else \
             "warn" if (bullish_divergence or macd_hist_rising) else "fail"
    stage5 = "pass" if obv_up_while_price_down else "warn"
    stage7 = "pass" if mtf_aligned else "warn"
    stage8 = "pass" if breakout_ok else "fail"

    stop_pct = round(float(atr.iloc[-1]) * 1.5 / last_close * 100, 1)

    stage2_flags = {
        "upside_ge_15": (upside_pct or 0) >= 15,
        "peg_reasonable": (peg is not None and peg <= 2.5),
        "forward_pe_present": forward_pe is not None,
    }
    stage2 = "pass" if all(stage2_flags.values()) else \
             "fail" if not stage2_flags["upside_ge_15"] else "warn"

    stages = {
        "1_macro": {"status": stage1_status, "index_name": index_name, "index_symbol": index_symbol},
        "2_fundamentals": {"status": stage2, "upside_pct": upside_pct, "forward_pe": forward_pe, "peg": peg},
        "3_technical_position": {"status": stage3, "rsi": round(rsi_last, 1), "bb_low": round(bb_low, 2), "bb_high": round(bb_high, 2)},
        "4_divergence_momentum": {"status": stage4, "bullish_divergence": bullish_divergence, "macd_hist_rising": macd_hist_rising},
        "5_volume_flow": {"status": stage5, "obv_bullish_divergence": obv_up_while_price_down},
        "6_trend_exhaustion": {"status": trend_exhaustion, "adx": round(adx_last, 1)},
        "7_multi_timeframe": {"status": stage7, "daily_rsi": round(rsi_last, 1), "weekly_rsi": round(float(weekly_rsi.iloc[-1]), 1)},
        "8_trigger_candle": {"status": stage8, "breakout_confirmed": breakout_ok, "pattern": breakout_pattern, "days_ago": breakout_days_ago, "hammer": bool(hammer), "bullish_engulfing": bool(engulfing), "morning_star": bool(morning_star)},
        "9_position_sizing": {"atr_stop_pct": stop_pct},
        "10_target": {"target_mean_price": target_mean, "upside_pct": upside_pct},
    }

    known_weights = {k: WEIGHTS[k] for k in WEIGHTS if stages[k]["status"] != "unknown"}
    if known_weights:
        raw_score = sum(known_weights[k] * STATUS_SCORE[stages[k]["status"]] for k in known_weights)
        score = round(raw_score / sum(known_weights.values()) * 100)
    else:
        score = 0

    if stage1_status == "fail":
        score = min(score, 40)
    elif stage1_status == "unknown":
        score = round(score * 0.85)

    result = {
        "ticker": ticker.upper(),
        "price": round(last_close, 2),
        "updated": pd.Timestamp.utcnow().isoformat(),
        "score": score,
        "stages": stages,
        "sector": sector,
        "sector_relative_strength_up": sector_rs,
    }
    with open(f"{OUT_DIR}/{ticker.upper()}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    if not os.path.exists(WATCHLIST_PATH):
        print(f"watchlist not found: {WATCHLIST_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(WATCHLIST_PATH) as f:
        tickers = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for t in tickers:
        try:
            r = analyze(t)
            results.append(r)
            print(f"  {t}: score={r['score']}")
        except Exception as e:
            print(f"  {t}: FAILED ({e})", file=sys.stderr)
        time.sleep(0.6)

    rankings = sorted(
        [{"ticker": r["ticker"], "score": r["score"], "price": r["price"],
          "upside_pct": r["stages"]["2_fundamentals"]["upside_pct"],
          "rsi": r["stages"]["3_technical_position"]["rsi"],
          "divergence": r["stages"]["4_divergence_momentum"]["status"],
          "trigger": r["stages"]["8_trigger_candle"]["status"]}
         for r in results],
        key=lambda x: x["score"], reverse=True
    )
    with open(f"{OUT_DIR}/rankings.json", "w") as f:
        json.dump({"updated": pd.Timestamp.utcnow().isoformat(), "rankings": rankings}, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(results)}/{len(tickers)}개 성공, rankings.json 저장")


if __name__ == "__main__":
    main()
