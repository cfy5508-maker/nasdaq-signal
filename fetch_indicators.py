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
HISTORY_PATH = "data/score_history.jsonl"

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
    return ("S&P500", "^GSPC")  # 아무 데도 안 걸리면 S&P500을 기본값으로


INDEX_ETF_MAP = {
    "^IXIC": "QQQ",
    "^GSPC": "SPY",
    "^DJI": "DIA",
}
_VOLUME_CACHE = {}


def market_volume_health(index_symbol, months=3):
    """종목이 속한 지수의 대표 ETF 거래량이, 최근 N개월 평균 거래량 대비
    몇 % 높은지/낮은지, 그리고 가장 최근 마감일 가격이 올랐는지 내렸는지를 함께 보여준다.
    ※ '오늘'이 아니라 yfinance가 제공하는 '가장 최근에 확정된(마감된) 거래일' 기준이다.
    (거래량 비율만으로는 방향을 알 수 없어서, 가격 방향과 같이 봐야 해석이 됨)
    참고용이며 점수 계산에는 반영하지 않는다."""
    etf_symbol = INDEX_ETF_MAP.get(index_symbol)
    if etf_symbol is None:
        return {"etf": None, "latest_volume": None, "avg_volume_3m": None, "diff_pct": None, "price_up_latest": None}
    if etf_symbol in _VOLUME_CACHE:
        return _VOLUME_CACHE[etf_symbol]
    try:
        hist = yf.Ticker(etf_symbol).history(period="6mo")
        vol = hist["Volume"]
        close = hist["Close"]
        latest_vol = float(vol.iloc[-1])  # 가장 최근 마감된 거래일의 거래량
        avg_vol_3m = float(vol.iloc[-63:-1].mean())  # 최근 3개월(약 63거래일) 평균, 최신일 제외
        diff_pct = round((latest_vol / avg_vol_3m - 1) * 100, 1) if avg_vol_3m > 0 else None
        price_up_latest = bool(close.iloc[-1] > close.iloc[-2]) if len(close) >= 2 else None
        result = {"etf": etf_symbol, "latest_volume": int(latest_vol),
                   "avg_volume_3m": int(avg_vol_3m), "diff_pct": diff_pct, "price_up_latest": price_up_latest}
    except Exception:
        result = {"etf": etf_symbol, "latest_volume": None, "avg_volume_3m": None, "diff_pct": None, "price_up_latest": None}
    _VOLUME_CACHE[etf_symbol] = result
    return result


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
    # 1_market_health(시장건전성)는 참고용이라 여기 없음 - 점수 계산에 반영 안 됨
    "2_fundamentals": 2.0,
    "3_divergence_gate": 2.0,   # 다이버전스 게이트 (스윙저점+간격3일 이상)
    "4_zscore": 2.5,            # Z-score (종목별 개별 계산)
    "5_trigger_candle": 1.0,    # 반전캔들+돌파
}
STATUS_SCORE = {"pass": 1.0, "warn": 0.5, "fail": 0.0, "unknown": 0.25, "neutral": 0.25}
MAX_SCORE = sum(WEIGHTS.values())

ADDON_WEIGHTS = {
    "1_fundamentals": 3.0,          # 눌림목은 업사이드 여유가 핵심이라 신규진입(2.0)보다 상향
    "2_pullback_gate": 2.5,         # 20/40일선 근접 - 가장 강한 근거(대규모 검증)
    "3_rsi_zone": 1.0,              # RSI 50~60구간 - 약한 보조 신호
    "4_trigger_confirmed": 2.0,     # 셋업당일 고가 1~3일내 돌파 확정
}


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


def detect_long_lower_wick(o, h, l, c):
    """긴 아래꼬리: 해머보다 완화된 기준(위꼬리 제한 없음).
    백테스트 검증: 아래꼬리 있음 51.9%(+2.6%p) vs 없음 48.9%(-0.4%p), 표본158건 - 약한 신호."""
    body = abs(c.iloc[-1] - o.iloc[-1])
    lower_wick = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
    total_range = h.iloc[-1] - l.iloc[-1]
    if total_range <= 0:
        return False
    return (lower_wick / total_range) >= 0.5


def long_lower_wick_recent(o, h, l, c, lookback_days=3):
    """오늘뿐 아니라 최근 lookback_days일 안에 긴 아래꼬리가 있었는지 확인.
    (trigger_with_breakout처럼 돌파 확정까지는 요구하지 않음 - 약한 warn 신호이므로)"""
    n = len(c)
    for days_ago in range(0, lookback_days + 1):
        idx = n - 1 - days_ago
        if idx < 0:
            break
        sub_o, sub_h, sub_l, sub_c = o.iloc[:idx+1], h.iloc[:idx+1], l.iloc[:idx+1], c.iloc[:idx+1]
        if detect_long_lower_wick(sub_o, sub_h, sub_l, sub_c):
            return True, days_ago
    return False, None


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


def find_realtime_lows(close_values, order=5):
    """미래 데이터를 보지 않고, '오늘 종가가 최근 order일 종가보다 낮은지'만으로
    저점 후보를 표시한다. 이 방식으로 찾은 마지막 저점이 오늘(n-1)과 일치하면
    '오늘 막 새 저점을 확정했다'는 뜻이라 실시간 신호로 쓸 수 있다.
    (백테스트로 검증된 방식: 스윙저점 두 개 연속 비교 + 간격 3일 이상 + Z-score + ADX)
    """
    positions = []
    n = len(close_values)
    for i in range(order, n):
        past_window = close_values[i - order:i]
        if close_values[i] < past_window.min():
            positions.append(i)
    return positions


def analyze(ticker, trim_days=0, write_file=True):
    """trim_days: 0이면 오늘 기준. 1이면 마지막 1거래일을 잘라내고
    '그 전날 기준'으로 계산한다(소급 계산용, score_history 백필에 사용).
    write_file: False면 data/<TICKER>.json을 덮어쓰지 않는다(백필 시 오늘자 파일 보호)."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y")
    if hist.empty:
        raise ValueError(f"no price history for {ticker}")
    if trim_days > 0:
        if len(hist) <= trim_days:
            raise ValueError(f"not enough history to trim {trim_days} days for {ticker}")
        hist = hist.iloc[:-trim_days]

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
    bullish_divergence_legacy = bool(price_lower_low and rsi_higher_low)  # 참고용(구방식), 점수 미반영

    # ── 검증된 새 다이버전스 게이트: 진짜 스윙저점(전후 5일보다 낮은 지점) 기반 ──
    # 백테스트 검증: 68건 표본에서 Z-score 낮을수록 승률 66.7%→52.4%→26.1% 단조감소 확인
    # 추가: 오늘이 정확히 새 저점을 찍은 날이 아니어도, 마지막 저점가 대비 +3% 이내면
    # 신호가 아직 유효(저점 근처에서 다지는 중)한 것으로 간주한다.
    TOLERANCE_PCT = 0.03
    MIN_GAP_DAYS = 3
    MAX_GAP_DAYS = 30  # 30일(약 1.5개월) 넘게 떨어진 저점은 다른 하락 사이클로 간주해 배제
    close_values = c.values
    realtime_lows = find_realtime_lows(close_values, order=5)
    bullish_divergence = False     # pass인 경우만 True (점수 캡 판정용)
    divergence_present = False     # pass 또는 warn (약한 신호 포함) - 캡 완화용
    stage_divergence = "fail"
    gap_days = None
    signal_fresh = None  # True=오늘 신규 확정, False=오차범위 내 유지, None=신호없음
    if len(realtime_lows) >= 2:
        pos2 = realtime_lows[-1]

        # 저점1: "그냥 직전 저점"이 아니라, 저점2 이전 MIN~MAX_GAP일 구간 안에서
        # 실제로 가장 낮았던(가장 의미있는) 저점을 찾는다. 이래야 PEP처럼 저점이
        # 매일 조금씩 갱신되는 종목에서, 진짜 깊었던 저점(예: 10일 전 최저가)을
        # 놓치지 않고 비교할 수 있다.
        candidates = [p for p in realtime_lows
                      if p != pos2 and MIN_GAP_DAYS <= (pos2 - p) <= MAX_GAP_DAYS]
        pos1 = min(candidates, key=lambda p: close_values[p]) if candidates else None

        gap_days = (pos2 - pos1) if pos1 is not None else None
        is_today_new_low = (pos2 == len(close_values) - 1)

        # 오늘이 새 저점이 아니면, 마지막 저점가 대비 오늘 가격이 +3% 이내인지 확인
        price2_raw = float(c.iloc[pos2])
        price_today = float(c.iloc[-1])
        within_tolerance = bool(price_today >= price2_raw and (price_today - price2_raw) / price2_raw <= TOLERANCE_PCT)

        if pos1 is not None and (is_today_new_low or within_tolerance):
            price1, price2 = float(c.iloc[pos1]), price2_raw
            rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
            if not (pd.isna(rsi1) or pd.isna(rsi2)):
                price_lower_low_gate = price2 < price1
                rsi_improved = rsi2 > rsi1
                if price_lower_low_gate and rsi_improved:
                    stage_divergence = "pass"      # 정석 다이버전스
                    bullish_divergence = True
                    divergence_present = True
                    signal_fresh = is_today_new_low
                elif rsi_improved:
                    stage_divergence = "warn"      # 가격은 안 낮아졌지만 RSI는 개선(이중바닥 성격)
                    divergence_present = True
                    signal_fresh = is_today_new_low
                else:
                    stage_divergence = "fail"      # RSI도 개선 안 됨

    # 상승추세 진입 신호: 30일 이내에 정석 다이버전스가 있었지만, 그 이후 가격이
    # 오차범위(3%)를 넘어서 확실히 반등한 경우. 신규진입 점수(캡 30/60점)와는 별개로,
    # "이미 반등이 확인된 상태"를 알려주는 참고용 신호. 점수 계산에는 반영하지 않는다.
    uptrend_entry_signal = None
    if len(realtime_lows) >= 2:
        pos2_u = realtime_lows[-1]
        candidates_u = [p for p in realtime_lows
                        if p != pos2_u and MIN_GAP_DAYS <= (pos2_u - p) <= MAX_GAP_DAYS]
        pos1_u = min(candidates_u, key=lambda p: close_values[p]) if candidates_u else None
        if pos1_u is not None:
            price1_u, price2_u = float(c.iloc[pos1_u]), float(c.iloc[pos2_u])
            rsi1_u, rsi2_u = float(rsi.iloc[pos1_u]), float(rsi.iloc[pos2_u])
            price_today_u = float(c.iloc[-1])
            was_real_divergence = bool(price2_u < price1_u and rsi2_u > rsi1_u and
                                        not pd.isna(rsi1_u) and not pd.isna(rsi2_u))
            gain_since_low = (price_today_u - price2_u) / price2_u if price2_u > 0 else 0
            already_broke_tolerance = gain_since_low > TOLERANCE_PCT
            if was_real_divergence and already_broke_tolerance:
                uptrend_entry_signal = {
                    "active": True,
                    "days_since_divergence": (len(close_values) - 1) - pos2_u,
                    "gain_since_low_pct": round(gain_since_low * 100, 1),
                    "divergence_date_index": pos2_u,
                }

    # 다이버전스 무효화 신호: 과거(최근 30일 내)에 정석 다이버전스(pass)가 있었는데,
    # 그 이후 가격이 그 저점보다 더 뚫렸고(신저점 갱신) + RSI도 같이 더 낮아진 경우.
    # "반등 기대가 깨지고 계속 하락 중"이라는 뜻. 점수 계산에는 반영하지 않는 참고용.
    # 단, 현재 가장 최근 저점쌍이 이미 새로운 다이버전스(pass/warn)를 형성했다면,
    # 그건 이미 "새 사이클이 시작된 것"이므로 더 오래된 무효화 이력은 무시한다.
    divergence_invalidated_signal = None
    if len(realtime_lows) >= 3 and not divergence_present:
        pos_latest = realtime_lows[-1]  # 가장 최근 저점(=현재 진행 중인 하락의 끝)
        price_latest = float(c.iloc[pos_latest])
        rsi_latest_low = float(rsi.iloc[pos_latest])
        # 가장 최근 저점보다 이전에, "정석 다이버전스가 확정됐었던 저점"이 있었는지
        # 최근 것부터 역순으로, 30일 이내 범위에서 탐색
        for idx in range(len(realtime_lows) - 2, -1, -1):
            candidate_pos2 = realtime_lows[idx]
            if candidate_pos2 >= pos_latest:
                continue
            if pos_latest - candidate_pos2 > MAX_GAP_DAYS:
                break  # 30일 넘게 과거로 가면 더 볼 필요 없음
            # 이 candidate_pos2 시점에 정석 다이버전스가 있었는지, 그 자신의 pos1을 구해 확인
            own_candidates = [p for p in realtime_lows
                               if p != candidate_pos2 and MIN_GAP_DAYS <= (candidate_pos2 - p) <= MAX_GAP_DAYS]
            own_pos1 = min(own_candidates, key=lambda p: close_values[p]) if own_candidates else None
            if own_pos1 is None:
                continue
            own_price1, own_price2 = float(c.iloc[own_pos1]), float(c.iloc[candidate_pos2])
            own_rsi1, own_rsi2 = float(rsi.iloc[own_pos1]), float(rsi.iloc[candidate_pos2])
            if pd.isna(own_rsi1) or pd.isna(own_rsi2):
                continue
            had_pass_divergence = bool(own_price2 < own_price1 and own_rsi2 > own_rsi1)
            if not had_pass_divergence:
                continue
            # 정석 다이버전스가 있었던 저점을 찾음 -> 이후 가격/RSI가 더 무너졌는지 확인
            price_broke_down = price_latest < own_price2
            rsi_broke_down = rsi_latest_low < own_rsi2
            if price_broke_down and rsi_broke_down:
                divergence_invalidated_signal = {
                    "active": True,
                    "prior_divergence_date_index": candidate_pos2,
                    "days_since_prior_divergence": pos_latest - candidate_pos2,
                    "price_drop_since_pct": round((own_price2 - price_latest) / own_price2 * 100, 1),
                }
            break  # 가장 가까운 과거 다이버전스 하나만 확인하고 종료

    # Z-score: 다이버전스 유무와 무관하게, "지금 이 순간" RSI가 이 종목 평균 대비
    # 얼마나 이례적인 위치인지를 항상 계산한다 (종목별 개별 계산).
    # 상장 초기 등으로 1년치 데이터가 없으면, 있는 만큼(최소 30일)으로 계산한다.
    rsi_zscore = None
    rsi_window_1y = rsi.dropna()
    if len(rsi_window_1y) >= 30:
        rsi_mean_1y = float(rsi_window_1y.mean())
        rsi_std_1y = float(rsi_window_1y.std())
        if rsi_std_1y > 0:
            rsi_zscore = (rsi_last - rsi_mean_1y) / rsi_std_1y

    macd_hist = macd.macd_diff()
    macd_hist_rising = bool(macd_hist.iloc[-1] > macd_hist.iloc[-90:-1].min()) if len(macd_hist) > 90 else False

    adx_last = float(adx_ind.adx().iloc[-1])

    weekly = hist.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    weekly_rsi = RSIIndicator(weekly["Close"], window=14).rsi()

    breakout_ok, breakout_pattern, breakout_days_ago = trigger_with_breakout(o, h, l, c)
    hammer = detect_hammer(o, h, l, c)
    engulfing = detect_bullish_engulfing(o, h, l, c)
    morning_star = detect_morning_star(o, h, l, c) if len(c) >= 3 else False
    long_lower_wick, wick_days_ago = long_lower_wick_recent(o, h, l, c, lookback_days=3)

    info = {}
    try:
        info = tk.info
    except Exception:
        pass

    exchange = info.get("exchange")
    index_name, index_symbol = determine_index(ticker.upper(), exchange)
    volume_health = market_volume_health(index_symbol)

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

    # 3단계: 기술적 위치 - Z-score 기반(종목별 개별 계산). RSI 절대값은 참고 표시용.
    # 백테스트 검증: Z<=-1.0일 때 승률 58.1%, Z>-1.0일 때 27.3% (43건 표본)
    if rsi_zscore is not None:
        stage3 = "pass" if rsi_zscore <= -1.5 else \
                 "warn" if rsi_zscore <= -1.0 else "fail"
    else:
        stage3 = "fail"

    # 2단계: 다이버전스 게이트 - stage_divergence는 위에서 이미 pass/warn/fail로 계산됨

    # 4단계(ADX)는 백테스트 재검증 결과 두 독립 표본에서 방향이 계속 뒤집혀 폐기함.
    # adx_last 값 자체는 참고용으로 남겨두되, 점수 계산에는 반영하지 않음 (나중에 재검토).

    # 4단계: 트리거캔들
    # 백테스트 검증: 돌파확정 pass / 긴아래꼬리만 있어도 약한 warn(+2.6%p, 표본158건) / 없으면 fail
    if breakout_ok:
        stage_trigger = "pass"
    elif long_lower_wick:
        stage_trigger = "warn"
    else:
        stage_trigger = "fail"

    stop_pct = round(float(atr.iloc[-1]) * 1.5 / last_close * 100, 1)

    # 1단계: 펀더멘털 - PEG가 좋을수록 업사이드 커트라인을 낮춰줌 (서로 보완)
    if peg is not None and peg <= 1.0:
        upside_threshold = 5
    elif peg is not None and peg <= 1.5:
        upside_threshold = 10
    else:
        upside_threshold = 15

    stage1_flags = {
        "upside_ge_threshold": (upside_pct or 0) >= upside_threshold,
        "forward_pe_present": forward_pe is not None,
    }
    stage1 = "pass" if all(stage1_flags.values()) else \
             "warn" if stage1_flags["upside_ge_threshold"] else "fail"

    # ── 눌림목(재진입) 계산식 ──────────────────
    # 백테스트 검증(대형주50종목):
    #   - 20/40일선 ±3% 근접이 핵심 게이트 (근접 단독으로 이미 기준선 대비 효과 대부분)
    #   - RSI 50~60구간이 60에 가까울수록 약하게 개선(효과는 작음, 참고용 가중치)
    #   - 셋업(근접+RSI50+) 당일 고가를 1~3일 내 종가로 돌파해야 진입확정(트리거) - 표본 늘고 승률도 개선
    #   - 손절: 직전 스윙저점 이탈 시 (16.8% vs 75.3%로 매우 강력한 근거, 별도 신호로만 관리 - 점수 미반영)
    #   - 폐기: 고점/저점 우상향, OBV, 급락캔들없음, 40일선기울기, ATR기반이격도 (전부 역효과/무의미)
    sma20 = c.rolling(20).mean()
    sma40_addon = sma40  # 신규진입에서 이미 계산된 40일선 재사용

    PULLBACK_MA_TOLERANCE = 0.03
    dist_sma20 = abs(last_close - sma20.iloc[-1]) / sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else None
    dist_sma40 = abs(last_close - sma40_addon.iloc[-1]) / sma40_addon.iloc[-1] if not pd.isna(sma40_addon.iloc[-1]) else None
    near_sma20 = bool(dist_sma20 is not None and dist_sma20 <= PULLBACK_MA_TOLERANCE)
    near_sma40 = bool(dist_sma40 is not None and dist_sma40 <= PULLBACK_MA_TOLERANCE)
    pullback_gate = near_sma20 or near_sma40
    addon_stage_gate = "pass" if pullback_gate else "fail"

    # RSI 존: 60이상 pass, 50~60 warn, 50미만 fail
    if rsi_last >= 60:
        addon_stage_rsi = "pass"
    elif rsi_last >= 50:
        addon_stage_rsi = "warn"
    else:
        addon_stage_rsi = "fail"

    # 셋업/트리거: 게이트+RSI50이상이 성립한 최근 1~3일 중 하루를 "셋업일"로 보고,
    # 그날 고가를 그 이후(오늘 포함) 종가가 넘었는지 확인
    setup_confirmed = False
    setup_days_ago = None
    trigger_confirmed = False
    trigger_lag = None
    for days_ago in range(1, 4):
        idx = len(c) - 1 - days_ago
        if idx < 40:
            continue
        setup_price = float(c.iloc[idx])
        setup_sma20 = float(sma20.iloc[idx]) if not pd.isna(sma20.iloc[idx]) else None
        setup_sma40 = float(sma40_addon.iloc[idx]) if not pd.isna(sma40_addon.iloc[idx]) else None
        setup_rsi = float(rsi.iloc[idx]) if not pd.isna(rsi.iloc[idx]) else None
        if setup_sma20 is None or setup_sma40 is None or setup_rsi is None:
            continue
        setup_near = (abs(setup_price - setup_sma20) / setup_sma20 <= PULLBACK_MA_TOLERANCE) or \
                     (abs(setup_price - setup_sma40) / setup_sma40 <= PULLBACK_MA_TOLERANCE)
        if setup_near and setup_rsi >= 50:
            setup_confirmed = True
            setup_days_ago = days_ago
            setup_high = float(h.iloc[idx])
            if float(c.iloc[-1]) > setup_high:
                trigger_confirmed = True
                trigger_lag = days_ago
            break
    if trigger_confirmed:
        addon_stage_trigger = "pass"
    elif setup_confirmed:
        addon_stage_trigger = "warn"  # 셋업은 있지만 아직 돌파 대기중
    else:
        addon_stage_trigger = "fail"

    # 손절신호(점수 미반영, 별도 참고): 직전 스윙저점 이탈 여부
    pullback_stop_signal = None
    try:
        _lows = find_realtime_lows(c.values, order=5)
        if len(_lows) >= 1:
            _recent_low_pos = _lows[-1]
            _recent_low_price = float(c.iloc[_recent_low_pos])
            _stopped = bool(last_close < _recent_low_price)
            pullback_stop_signal = {"stop_reference_price": round(_recent_low_price, 2), "stopped_out": _stopped}
    except Exception:
        pass

    addon_stop_pct = round(abs(last_close - sma20.iloc[-1]) / last_close * 100, 1)

    stages_addon = {
        "1_fundamentals": {"status": stage1, "upside_pct": upside_pct, "forward_pe": forward_pe, "peg": peg},
        "2_pullback_gate": {"status": addon_stage_gate, "near_sma20": near_sma20, "near_sma40": near_sma40},
        "3_rsi_zone": {"status": addon_stage_rsi, "rsi": round(rsi_last, 1)},
        "4_trigger_confirmed": {"status": addon_stage_trigger, "setup_confirmed": setup_confirmed,
                                 "setup_days_ago": setup_days_ago, "trigger_confirmed": trigger_confirmed,
                                 "trigger_lag": trigger_lag},
        "9_position_sizing": {"stop_pct_from_20sma": addon_stop_pct},
    }
    addon_known_weights = {k: ADDON_WEIGHTS[k] for k in ADDON_WEIGHTS if stages_addon[k]["status"] != "unknown"}
    if addon_known_weights:
        addon_raw = sum(addon_known_weights[k] * STATUS_SCORE[stages_addon[k]["status"]] for k in addon_known_weights)
        addon_score = round(addon_raw / sum(addon_known_weights.values()) * 100)
    else:
        addon_score = 0

    stages = {
        "1_market_health": {"status": "neutral", "index_name": index_name, "index_symbol": index_symbol,
                             "etf": volume_health["etf"], "volume_diff_pct_3m": volume_health["diff_pct"],
                             "price_up_latest": volume_health["price_up_latest"]},
        "2_fundamentals": {"status": stage1, "upside_pct": upside_pct, "forward_pe": forward_pe, "peg": peg},
        "3_divergence_gate": {"status": stage_divergence, "bullish_divergence": bullish_divergence, "divergence_present": divergence_present, "gap_days": gap_days, "signal_fresh": signal_fresh},
        "4_zscore": {"status": stage3, "rsi": round(rsi_last, 1), "rsi_zscore_1y": round(rsi_zscore, 2) if rsi_zscore is not None else None},
        "5_trigger_candle": {"status": stage_trigger, "breakout_confirmed": breakout_ok, "pattern": breakout_pattern, "days_ago": breakout_days_ago, "hammer": bool(hammer), "bullish_engulfing": bool(engulfing), "morning_star": bool(morning_star), "long_lower_wick": bool(long_lower_wick), "wick_days_ago": wick_days_ago, "adx_reference": round(adx_last, 1)},
    }

    known_weights = {k: WEIGHTS[k] for k in WEIGHTS if stages[k]["status"] != "unknown"}
    if known_weights:
        raw_score = sum(known_weights[k] * STATUS_SCORE[stages[k]["status"]] for k in known_weights)
        score = round(raw_score / sum(known_weights.values()) * 100)
    else:
        score = 0

    # 다이버전스 상태에 따른 점수 상한 캡: 위반(fail)은 30점, 애매(warn)는 60점, 충족(pass)은 캡 없음
    if stage_divergence == "fail":
        score = min(score, 30)
    elif stage_divergence == "warn":
        score = min(score, 60)

    result = {
        "ticker": ticker.upper(),
        "price": round(last_close, 2),
        "updated": pd.Timestamp.now("UTC").isoformat(),
        "as_of_date": str(hist.index[-1].date()),
        "score": score,
        "stages": stages,
        "entry_atr_stop_pct": stop_pct,
        "score_addon": addon_score,
        "stages_addon": stages_addon,
        "sector": sector,
        "sector_relative_strength_up": sector_rs,
        "uptrend_entry_signal": uptrend_entry_signal,
        "divergence_invalidated_signal": divergence_invalidated_signal,
        "pullback_stop_signal": pullback_stop_signal,
    }
    if write_file:
        with open(f"{OUT_DIR}/{ticker.upper()}.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def load_recent_history(ticker, days=4):
    """score_history.jsonl에서 해당 티커의 최근 기록(날짜 오름차순)을 최대 days개 반환."""
    if not os.path.exists(HISTORY_PATH):
        return []
    rows = []
    with open(HISTORY_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("ticker") == ticker:
                rows.append(rec)
    rows.sort(key=lambda r: r["date"])
    # 같은 날짜 중복 시 마지막 값만 유지
    dedup = {}
    for r in rows:
        dedup[r["date"]] = r
    rows = list(dedup.values())
    return rows[-days:]


def compute_signal(ticker, today_score, score_key="score"):
    """상승신호/하락신호/신호없음 판정. score_key: 'score' 또는 'score_addon'."""
    hist = load_recent_history(ticker, days=4)
    if not hist:
        return {"signal": "none", "reasons": []}

    prev = hist[-1][score_key]
    change = today_score - prev

    def zone(s):
        if s >= 65: return 2
        if s >= 40: return 1
        return 0

    crossed_up = zone(today_score) > zone(prev)
    crossed_down = zone(today_score) < zone(prev)
    spike_up = change >= 25
    spike_down = change <= -25

    streak_up = len(hist) >= 2 and all(
        hist[i][score_key] < hist[i+1][score_key] for i in range(len(hist)-1)
    ) and today_score > hist[-1][score_key]
    streak_down = len(hist) >= 2 and all(
        hist[i][score_key] > hist[i+1][score_key] for i in range(len(hist)-1)
    ) and today_score < hist[-1][score_key]

    reasons = []
    change_str = f"전일 {prev} → 오늘 {today_score} (전일 대비 {'+' if change>=0 else ''}{change})"
    if spike_up:
        reasons.append({"label": "급등 기준 (+25 이상)", "detail": change_str})
    if crossed_up:
        reasons.append({"label": "구간 상향 돌파", "detail": None})
    if streak_up:
        reasons.append({"label": "3일 연속 상승", "detail": None})

    if reasons:
        return {"signal": "up", "reasons": reasons}

    if spike_down:
        reasons.append({"label": "급락 기준 (-25 이상)", "detail": change_str})
    if crossed_down:
        reasons.append({"label": "구간 하향 이탈", "detail": None})
    if streak_down:
        reasons.append({"label": "3일 연속 하락", "detail": None})

    if reasons:
        return {"signal": "down", "reasons": reasons}

    return {"signal": "none", "reasons": []}


def log_snapshot(r):
    """나중 승률 재보정용 원본 데이터. 날짜별로 한 줄씩 누적(jsonl), 덮어쓰지 않음.
    r에 as_of_date가 있으면 그 날짜로 기록(백필용), 없으면 오늘(UTC) 날짜로 기록."""
    snap = {
        "date": r.get("as_of_date") or pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
        "ticker": r["ticker"],
        "price": r["price"],
        "score": r["score"],
        "score_addon": r["score_addon"],
        "stage_status": {k: v["status"] for k, v in r["stages"].items() if "status" in v},
        "stage_status_addon": {k: v["status"] for k, v in r["stages_addon"].items() if "status" in v},
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(snap, ensure_ascii=False) + "\n")


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
            r["signal"] = compute_signal(t, r["score"], "score")
            r["signal_addon"] = compute_signal(t, r["score_addon"], "score_addon")
            results.append(r)
            log_snapshot(r)
            print(f"  {t}: score={r['score']} signal={r['signal']['signal']}")
        except Exception as e:
            print(f"  {t}: FAILED ({e})", file=sys.stderr)
        time.sleep(0.6)

    rankings = sorted(
        [{"ticker": r["ticker"], "score": r["score"], "score_addon": r["score_addon"], "price": r["price"],
          "upside_pct": r["stages"]["2_fundamentals"]["upside_pct"],
          "rsi": r["stages"]["4_zscore"]["rsi"],
          "divergence": r["stages"]["3_divergence_gate"]["status"],
          "trigger": r["stages"]["5_trigger_candle"]["status"],
          "signal": r["signal"], "signal_addon": r["signal_addon"],
          "uptrend_entry_signal": r["uptrend_entry_signal"],
          "divergence_invalidated_signal": r["divergence_invalidated_signal"]}
         for r in results],
        key=lambda x: x["score"], reverse=True
    )
    with open(f"{OUT_DIR}/rankings.json", "w") as f:
        json.dump({"updated": pd.Timestamp.now("UTC").isoformat(), "rankings": rankings}, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(results)}/{len(tickers)}개 성공, rankings.json 저장")


if __name__ == "__main__":
    main()
