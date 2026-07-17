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


def _market_already_closed_today():
    """미 동부시간 기준, 오늘 정규장(16:00 ET)이 이미 마감됐는지 확인.
    주말/공휴일 등 세세한 휴장일까지는 안 따지고, 단순히 '오늘 16:00 ET가 지났는지'만 본다.
    (장중에 실행하면 yfinance가 주는 마지막 행이 미확정 데이터라, 이 경우 잘라내기 위함)"""
    try:
        from zoneinfo import ZoneInfo
        now_et = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
    except Exception:
        now_et = pd.Timestamp.now(tz="US/Eastern")
    market_close_today = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return now_et >= market_close_today


def get_confirmed_history(ticker_or_obj, period="1y"):
    """yfinance로 일봉 데이터를 가져오되, 오늘 미국 정규장이 아직 안 끝났다면
    마지막 행(장중 미확정 데이터)을 잘라내서 항상 '가장 최근에 완결된 거래일'까지만
    반환한다. 워크플로우를 언제(장중이든 장마감후든) 돌리든 같은 결과가 나오게 하기 위함.
    ticker_or_obj: 티커 문자열 또는 이미 만든 yf.Ticker 객체."""
    tk = ticker_or_obj if hasattr(ticker_or_obj, "history") else yf.Ticker(ticker_or_obj)
    hist = tk.history(period=period)
    if hist.empty:
        return hist
    if not _market_already_closed_today():
        today_et_date = pd.Timestamp.now(tz="US/Eastern").date()
        last_row_date = hist.index[-1].date()
        if last_row_date == today_et_date:
            hist = hist.iloc[:-1]
    return hist

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
    """종목이 속한 지수의 대표 ETF를 기준으로 시장건전성(참고용, 점수 미반영)을 계산한다.

    구성:
      축1 - 거래량 수준: 오늘 거래량이 3개월평균 대비 몇 % 높은지/낮은지
      축2 - 거래량 추세: 최근3일 평균거래량이 직전3일 평균거래량보다 늘고있는지/줄고있는지
      축3 - 지수 당일 변동폭: |오늘종가-어제종가|/어제종가, 3%+ 급락 시 상태를 한단계 격하

    백테스트 검증(QQQ+SPY+DIA, 5년, 표본3570건, 다음날 지수 반응 기준):
      높음(30%+)+하락중: 60.7%(+6.4%p, 135건) -> pass  ("급등후 숨고르기, 아직 안 끝남")
      낮음(-30%↓)+상승중: 58.1%(+3.7%p, 136건) -> pass  ("바닥에서 관심 되살아나는 초입")
      낮음(-30%↓)+하락중: 49.7%(-4.7%p, 459건) -> warn  (유일하게 뚜렷히 나쁜 조합, 표본충분)
      그 외 전부: 기준선(54.4%) 근처, 뚜렷한 신호 없음 -> neutral

    축3(변동폭)은 백테스트 미검증, 트레이더 일반 상식 기준. 오늘 3%+ 급락이면
    위에서 계산한 상태를 한단계 격하(pass->neutral->warn->fail).
    axis3: 지수 당일 변동폭(3%+급락시 격하), 지수 자체 반전캔들(격상)
    axis4: 20/40/60일선 골든크로스(최근10일, 격상) / 데드크로스(최근10일, 격하)

    백테스트 검증(QQQ+SPY+DIA, 5년):
      20-60일선 데드크로스(최근10일내 발생): 10일뒤 상승비율 50.3%(-10.2%p, 표본310건) -> 격하
      20-40일선 골든크로스(최근10일내 발생): 5일뒤 상승비율 61.4%(+2.8%p, 표본430건) -> 약하게 격상
    ※ '오늘'이 아니라 yfinance가 제공하는 '가장 최근에 확정된(마감된) 거래일' 기준이다.
    """
    etf_symbol = INDEX_ETF_MAP.get(index_symbol)
    empty_result = {"etf": None, "latest_volume": None, "avg_volume_3m": None, "diff_pct": None,
                     "trend_pct": None, "trend_up": None, "daily_change_pct": None,
                     "volatility_label": None, "index_trigger_candle": None,
                     "golden_cross_recent": None, "dead_cross_recent": None, "status": "neutral"}
    if etf_symbol is None:
        return empty_result
    if etf_symbol in _VOLUME_CACHE:
        return _VOLUME_CACHE[etf_symbol]
    try:
        hist = get_confirmed_history(etf_symbol, period="1y")
        o, h, l = hist["Open"], hist["High"], hist["Low"]
        vol = hist["Volume"]
        close = hist["Close"]

        latest_vol = float(vol.iloc[-1])
        avg_vol_3m = float(vol.iloc[-63:-1].mean())
        diff_pct = round((latest_vol / avg_vol_3m - 1) * 100, 1) if avg_vol_3m > 0 else None

        recent3_avg = float(vol.iloc[-3:].mean())
        prior3_avg = float(vol.iloc[-6:-3].mean())
        trend_up = bool(recent3_avg > prior3_avg) if prior3_avg > 0 else None
        trend_pct = round((recent3_avg / prior3_avg - 1) * 100, 1) if prior3_avg > 0 else None

        daily_change_pct = round((float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100, 2) if len(close) >= 2 else None

        # 축1x축2 조합으로 base_status 결정
        base_status = "neutral"
        if diff_pct is not None and trend_up is not None:
            if (diff_pct >= 30 and not trend_up) or (diff_pct <= -30 and trend_up):
                base_status = "pass"
            elif diff_pct <= -30 and trend_up is False:
                base_status = "warn"

        # 축3: 오늘 3%이상 급락이면 한 단계 격하
        DOWNGRADE = {"pass": "neutral", "neutral": "warn", "warn": "fail", "fail": "fail"}
        status = base_status
        volatility_label = "정상"
        if daily_change_pct is not None:
            abs_change = abs(daily_change_pct)
            if abs_change >= 3:
                volatility_label = "급변"
                if daily_change_pct <= -3:
                    status = DOWNGRADE[base_status]
            elif abs_change >= 2:
                volatility_label = "변동성큼"
            elif abs_change >= 1:
                volatility_label = "다소변동"

        # 지수 자체에 오늘 반전캔들(해머/불리시엔걸핑)이 확인되면 한 단계 격상.
        # 급락(축3 격하)과 반전캔들(격상)이 같은 날 동시에 걸리면 서로 상쇄되어
        # 원래 상태로 돌아간다 - "급락했지만 반전 조짐도 있다"는 애매함을 반영.
        UPGRADE = {"fail": "warn", "warn": "neutral", "neutral": "pass", "pass": "pass"}
        index_trigger_candle = bool(detect_hammer(o, h, l, close) or detect_bullish_engulfing(o, h, l, close))
        if index_trigger_candle:
            status = UPGRADE[status]

        # axis4: 골든크로스는 20-40일선, 데드크로스는 20-60일선 (서로 다른 쌍, 검증된 대로 분리유지).
        # 각 쌍 안에서도 최근10일 안에 크로스가 여러 번 있었을 수 있으므로, "가장 최근" 것만 반영.
        CROSS_LOOKBACK = 10
        sma20 = close.rolling(20).mean()
        sma40 = close.rolling(40).mean()
        sma60 = close.rolling(60).mean()
        above_20_40 = sma20 > sma40
        above_20_60 = sma20 > sma60

        def latest_cross_in_window(above_series, lookback):
            """윈도우 안에서 가장 최근 크로스 종류('golden'/'dead'/None)를 반환."""
            latest = None
            n_local = len(above_series)
            if n_local <= lookback:
                return None
            for j in range(n_local - lookback + 1, n_local):
                a_now, a_prev = above_series.iloc[j], above_series.iloc[j - 1]
                if pd.isna(a_now) or pd.isna(a_prev):
                    continue
                if a_now and not a_prev:
                    latest = "golden"
                elif not a_now and a_prev:
                    latest = "dead"
            return latest

        golden_cross_type = latest_cross_in_window(above_20_40, CROSS_LOOKBACK)  # 20-40일선
        dead_cross_type = latest_cross_in_window(above_20_60, CROSS_LOOKBACK)    # 20-60일선
        golden_cross_recent = (golden_cross_type == "golden")
        dead_cross_recent = (dead_cross_type == "dead")

        if dead_cross_recent:
            status = DOWNGRADE[status]
        if golden_cross_recent:
            status = UPGRADE[status]

        result = {"etf": etf_symbol, "latest_volume": int(latest_vol), "avg_volume_3m": int(avg_vol_3m),
                   "diff_pct": diff_pct, "trend_pct": trend_pct, "trend_up": trend_up,
                   "daily_change_pct": daily_change_pct, "volatility_label": volatility_label,
                   "index_trigger_candle": index_trigger_candle,
                   "golden_cross_recent": golden_cross_recent, "dead_cross_recent": dead_cross_recent,
                   "status": status}
    except Exception:
        result = empty_result
    _VOLUME_CACHE[etf_symbol] = result
    return result


def index_macro_status(index_symbol):
    if index_symbol in _INDEX_CACHE:
        return _INDEX_CACHE[index_symbol]
    try:
        hist = get_confirmed_history(index_symbol, period="1y")["Close"]
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
    # 1_market_health(시장건전성)는 참고용이라 여기 없음 - 점수 계산에 반영 안 됨 (신규진입과 동일)
    "2_fundamentals": 1.5,          # 3.0에서 하향 - 펀더멘털 혼자 점수를 과하게 끌어올리던 문제 수정
    "3_pullback_gate": 2.0,         # 20/40일선 근접 - 단순 추정(어디까지 더 빠질지 모름)이라 트리거보다 낮게
    "4_rsi_zone": 1.0,              # RSI 50~60구간 - 약한 보조 신호
    "5_trigger_confirmed": 2.5,     # 트리거캔들(해머/엔걸핑/긴아래꼬리)+거래량 - 실제 반전 확인된 사실이라 더 높게
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
    (trigger_with_breakout처럼 돌파 확정까지는 요구하지 않음 - 약한 warn 신호이므로)
    ※ pos2(저점)가 있으면 wick_near_low()를 대신 쓰는 게 맞다 - 이 함수는 pos2를 못 구한
    예외적인 경우(저점 후보가 부족한 경우)의 폴백용으로만 남겨둔다."""
    n = len(c)
    for days_ago in range(0, lookback_days + 1):
        idx = n - 1 - days_ago
        if idx < 0:
            break
        sub_o, sub_h, sub_l, sub_c = o.iloc[:idx+1], h.iloc[:idx+1], l.iloc[:idx+1], c.iloc[:idx+1]
        if detect_long_lower_wick(sub_o, sub_h, sub_l, sub_c):
            return True, days_ago
    return False, None


def wick_near_low(o, h, l, c, pos2, max_forward_days=3):
    """저점(pos2) 시점부터 그 이후 max_forward_days일 사이에서만 긴 아래꼬리를 찾는다.
    '오늘 기준 최근 며칠'이 아니라 '저점 근처'로 앵커링해서, 이미 저점에서 멀리
    반등한 뒤에 우연히 나온 무관한 캔들을 트리거로 착각하지 않도록 한다."""
    n = len(c)
    if pos2 is None:
        return False, None
    for idx in range(pos2, min(pos2 + max_forward_days + 1, n)):
        sub_o, sub_h, sub_l, sub_c = o.iloc[:idx+1], h.iloc[:idx+1], l.iloc[:idx+1], c.iloc[:idx+1]
        if detect_long_lower_wick(sub_o, sub_h, sub_l, sub_c):
            days_ago = (n - 1) - idx
            return True, days_ago
    return False, None


def trigger_with_breakout(o, h, l, c):
    """최근 1~3일 안에 반전캔들이 확정되고, 그 이후 종가가 그 캔들의 고점을 돌파했는지 확인.
    ※ pos2(저점)가 있으면 breakout_near_low()를 대신 쓰는 게 맞다 - 이 함수는 pos2를 못
    구한 예외적인 경우의 폴백용으로만 남겨둔다."""
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


def breakout_near_low(o, h, l, c, pos2, max_forward_days=3):
    """저점(pos2) 시점부터 그 이후 max_forward_days일 사이에서 반전캔들(해머/엔걸핑/모닝스타)이
    있었는지 찾고, 오늘 종가가 그 캔들의 고점을 돌파했는지 확인한다. '오늘 기준 최근 며칠'이
    아니라 '저점 근처'로 앵커링해서, 이미 저점과 무관하게 멀리 떨어진 뒤의 캔들을
    트리거로 착각하지 않도록 한다."""
    n = len(c)
    if pos2 is None:
        return False, None, None
    for idx in range(pos2, min(pos2 + max_forward_days + 1, n)):
        sub_o, sub_h, sub_l, sub_c = o.iloc[:idx+1], h.iloc[:idx+1], l.iloc[:idx+1], c.iloc[:idx+1]
        is_hammer = detect_hammer(sub_o, sub_h, sub_l, sub_c)
        is_engulf = detect_bullish_engulfing(sub_o, sub_h, sub_l, sub_c)
        is_star = detect_morning_star(sub_o, sub_h, sub_l, sub_c) if idx + 1 >= 3 else False
        if (is_hammer or is_engulf or is_star) and float(c.iloc[-1]) > float(h.iloc[idx]):
            pattern_name = "해머" if is_hammer else ("불리시엔걸핑" if is_engulf else "모닝스타")
            days_ago = (n - 1) - idx
            return True, pattern_name, days_ago
    return False, None, None


def rs_line(ticker_close, bench_close):
    ratio = (ticker_close / bench_close).dropna()
    if len(ratio) < 64:
        return None
    return bool(ratio.iloc[-1] > ratio.iloc[-63])


def compute_divergence_cycle_state(pos2, price2, o_vals, h_vals, l_vals, c_vals):
    """저점2(다이버전스 확정 시점) 이후 오늘까지를 하루씩 시뮬레이션해서 현재
    사이클 상태를 판정한다.

    상태 흐름:
      divergence(±3% 이내, 아직 승패 미결)
        -> +3% 초과 상승: uptrend
        -> -3% 초과 하락: invalidated (그 이후 얼마나 더 빠지든 유지)
      invalidated
        -> 하락구간 마지막 음봉의 고가를 (그날이든 나중이든) 종가가 돌파: rescue_attempt
      rescue_attempt
        -> 저점2 가격 자체를 재돌파: uptrend (무효화 깊이와 무관)
      uptrend
        -> 저점2를 다시 깨고 내려가면: downtrend

    참고: uptrend_start_day는 uptrend 상태로 "처음" 전환된 날짜 인덱스로,
    호출부에서 이 시점 이후 며칠이 지났는지로 신호 신선도(freshness)를 판단한다.
    """
    state = "divergence"
    rescue_ref_high = None
    invalidation_day = None
    rescue_day = None
    uptrend_start_day = None
    downtrend_start_day = None
    n = len(c_vals)
    for t in range(pos2 + 1, n):
        price_t = c_vals[t]
        pct_vs_low = (price_t - price2) / price2 * 100 if price2 > 0 else 0

        if state == "divergence":
            if pct_vs_low > 3:
                state = "uptrend"
                uptrend_start_day = t
            elif pct_vs_low < -3:
                state = "invalidated"
                invalidation_day = t
                rescue_ref_high = h_vals[t]  # 폴백: 음봉을 못 찾으면 이 날 고가
                for k in range(t, pos2, -1):
                    if c_vals[k] < o_vals[k]:  # 음봉(종가<시가)
                        rescue_ref_high = h_vals[k]
                        break
        elif state == "invalidated":
            if price_t > rescue_ref_high:
                state = "rescue_attempt"
                rescue_day = t
        elif state == "rescue_attempt":
            if price_t > price2:
                state = "uptrend"
                uptrend_start_day = t
        elif state == "uptrend":
            if price_t < price2:
                state = "downtrend"
                downtrend_start_day = t
        # downtrend는 그대로 유지 (새 다이버전스가 생기면 pos2 자체가 갱신되어 재시작됨)

    return {
        "state": state,
        "invalidation_day": invalidation_day,
        "rescue_day": rescue_day,
        "rescue_ref_high": rescue_ref_high,
        "uptrend_start_day": uptrend_start_day,
        "downtrend_start_day": downtrend_start_day,
    }


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


RECENT_BEARISH_LOOKBACK_DAYS = 126     # 6개월 - nasdaq_chart.html detectDivergence와 동일
RECENT_BEARISH_NEAR_HIGH_RATIO = 0.97
RECENT_BEARISH_RSI_MIN_DECLINE = 3
RECENT_BEARISH_WINDOW_DAYS = 10        # "최근"으로 볼 범위 - 백테스트에서 이 범위일 때만 효과 확인됨
RECENT_BEARISH_QUALITY_BONUS = 0.15    # 백테스트: 있음 62.1%(348건) vs 없음 51.1%(3632건) 승률차 반영


def recent_bearish_divergence(pos, close_values, rsi_values):
    """pos(정석 볼리시 다이버전스 확정 저점) 기준 직전 RECENT_BEARISH_WINDOW_DAYS
    거래일 안에, 6개월 최고점 대비 베어리시 다이버전스(nasdaq_chart.html의
    detectDivergence 베어리시 판정과 동일 기준)가 한 번이라도 있었는지 확인한다.
    백테스트(232종목, 15년) 검증: 있으면 이후 10일 단기 승률이 51.1%->62.1%로
    상승(있음 표본 348건). 30일/60일로 넓히면 효과가 2~3%p로 줄어드는 단기 효과라,
    이 함수도 RECENT_BEARISH_WINDOW_DAYS=10 범위 안에서만 체크한다."""
    start = max(RECENT_BEARISH_LOOKBACK_DAYS, pos - RECENT_BEARISH_WINDOW_DAYS + 1)
    for t in range(start, pos + 1):
        rsi_t = rsi_values[t]
        if pd.isna(rsi_t):
            continue
        window_close = close_values[t - RECENT_BEARISH_LOOKBACK_DAYS:t]
        window_rsi = rsi_values[t - RECENT_BEARISH_LOOKBACK_DAYS:t]
        if np.all(pd.isna(window_rsi)):
            continue
        high_idx = int(np.nanargmax(window_close))
        high_p = window_close[high_idx]
        high_rsi = window_rsi[high_idx]
        if pd.isna(high_rsi):
            continue
        if close_values[t] >= high_p * RECENT_BEARISH_NEAR_HIGH_RATIO and rsi_t < high_rsi - RECENT_BEARISH_RSI_MIN_DECLINE:
            return True
    return False


def analyze(ticker, trim_days=0, write_file=True):
    """trim_days: 0이면 '가장 최근 확정 거래일' 기준. 1이면 그 하루 전(소급 계산용,
    score_history 백필에 사용). write_file: False면 data/<TICKER>.json을 덮어쓰지 않는다."""
    tk = yf.Ticker(ticker)
    hist = get_confirmed_history(tk, period="1y")
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
    sma40 = c.rolling(40).mean()  # 눌림목 게이트에서 사용 (20/40일선 근접 판정)

    # ── 검증된 새 다이버전스 게이트: 진짜 스윙저점(전후 5일보다 낮은 지점) 기반 ──
    # 백테스트 검증: 68건 표본에서 Z-score 낮을수록 승률 66.7%→52.4%→26.1% 단조감소 확인
    # 추가: 오늘이 정확히 새 저점을 찍은 날이 아니어도, 마지막 저점가 대비 +3% 이내면
    # 신호가 아직 유효(저점 근처에서 다지는 중)한 것으로 간주한다.
    TOLERANCE_PCT = 0.03
    MIN_GAP_DAYS = 5  # 3일에서 상향(SMCI 사례: 3일 간격은 "같은 되돌림 안의 노이즈"를 다이버전스로 착각할 위험 확인)
    MAX_GAP_DAYS = 30  # 30일(약 1.5개월) 넘게 떨어진 저점은 다른 하락 사이클로 간주해 배제
    close_values = c.values
    realtime_lows = find_realtime_lows(close_values, order=5)
    bullish_divergence = False     # pass인 경우만 True (점수 캡 판정용)
    divergence_present = False     # pass 또는 warn (약한 신호 포함) - 캡 완화용
    hidden_bullish_divergence = False  # 히든 다이버전스: 가격 higher low + RSI lower low (추세지속 신호)
    recent_bearish_bonus_applied = False  # 정석 다이버전스 직전 10거래일 내 베어리시(6개월최고점) 있었는지
    stage_divergence = "fail"
    gap_days = None
    signal_fresh = None  # True=오늘 신규 확정, False=오차범위 내 유지, None=신호없음
    pos1, pos2 = None, None  # 다이버전스 구간(저점1~저점2) - 거래량 비교에도 재사용
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

        DIVERGENCE_RSI_MIN = 0    # 이 이하 개선폭이면 보너스 없음(기본점만)
        DIVERGENCE_RSI_MAX = 10   # 이 이상 개선폭이면 보너스 만점
        DIVERGENCE_BASE_SCORE = 0.5  # 구조(가격↓+RSI↑) 자체를 충족한 것만으로 보장되는 기본점
        DIVERGENCE_WARN_CAP = 0.6  # 이중바닥(warn)은 아무리 개선폭이 커도 이 배수까지만 (가격확인 없는 구조적 약점)
        DIVERGENCE_WARN_UPGRADE_THRESHOLD = 10  # 이중바닥이어도 RSI개선폭이 이 이상이면 pass로 격상
        DOUBLE_BOTTOM_MAX_PREMIUM_PCT = 0.10  # 저점2가 저점1보다 이 이상(10%) 높으면 재테스트가 아니라
                                                # 그냥 상승추세 중 눌림목이므로 이중바닥으로 안 봄
        divergence_quality = None
        rsi_improvement = None

        if pos1 is not None:
            price1, price2 = float(c.iloc[pos1]), price2_raw
            rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
            if not (pd.isna(rsi1) or pd.isna(rsi2)):
                price_lower_low_gate = price2 < price1
                rsi_improved = rsi2 > rsi1
                rsi_improvement = rsi2 - rsi1
                # 구조(가격↓+RSI↑) 충족만으로 기본 0.5점을 보장하고, RSI개선폭에 따라
                # 나머지 0.5점을 보너스로 얹는다. (RKLB 사례: 개선폭 0.7p여도 구조는 맞으므로
                # 0.07점처럼 극단적으로 낮게 나오던 문제 수정)
                bonus = 0.5 * max(0.0, min(1.0, (rsi_improvement - DIVERGENCE_RSI_MIN) / (DIVERGENCE_RSI_MAX - DIVERGENCE_RSI_MIN)))
                base_quality = DIVERGENCE_BASE_SCORE + bonus
                if is_today_new_low or within_tolerance:
                    if price_lower_low_gate and rsi_improved:
                        stage_divergence = "pass"      # 정석 다이버전스
                        bullish_divergence = True
                        divergence_present = True
                        signal_fresh = is_today_new_low
                        divergence_quality = base_quality  # 정석은 그대로 (신뢰도 최고)
                        # 직전 10거래일 내 베어리시(6개월 최고점) 다이버전스가 있었다면
                        # 단기 승률이 유의미하게 높았던 게 백테스트로 확인돼서 가산점 적용.
                        if recent_bearish_divergence(pos2, close_values, rsi.values):
                            divergence_quality = min(1.0, divergence_quality + RECENT_BEARISH_QUALITY_BONUS)
                            recent_bearish_bonus_applied = True
                    elif rsi_improved and rsi_improvement >= DIVERGENCE_WARN_UPGRADE_THRESHOLD and price2 <= price1 * (1 + DOUBLE_BOTTOM_MAX_PREMIUM_PCT):
                        # 이중바닥이지만 RSI 개선폭이 매우 커서(10포인트 이상) 정석급으로 격상.
                        # bullish_divergence는 False로 유지(가격은 실제로 안 낮아졌으므로 사실 그대로
                        # 표시하되), 판정(status)과 점수는 정석과 동일하게 취급한다.
                        stage_divergence = "pass"
                        divergence_present = True
                        signal_fresh = is_today_new_low
                        divergence_quality = base_quality  # 상한선 없이 그대로
                    elif rsi_improved and price2 <= price1 * (1 + DOUBLE_BOTTOM_MAX_PREMIUM_PCT):
                        stage_divergence = "warn"      # 가격은 안 낮아졌지만 RSI는 개선(이중바닥 성격)
                        divergence_present = True
                        signal_fresh = is_today_new_low
                        divergence_quality = base_quality * DIVERGENCE_WARN_CAP  # 이중바닥은 상한선 적용
                    elif rsi_improved:
                        # 저점2가 저점1보다 DOUBLE_BOTTOM_MAX_PREMIUM_PCT 넘게 높으면, 이건 저점을
                        # 재테스트한 게 아니라 이미 강하게 반등한 상태에서의 눌림목일 뿐이라 다이버전스로
                        # 보지 않는다(예: GEV - 저점1 866 -> 저점2 1042, +20%인데 이중바닥으로 잘못 잡히던 사례).
                        stage_divergence = "fail"
                    else:
                        # 히든 다이버전스: 가격은 higher low(상승추세 구조 유지), RSI는 lower low(모멘텀 눌림).
                        # 정석과는 반대 성격(반전이 아니라 추세지속) - 신규진입 게이트에서는 warn 등급으로만 인정.
                        rsi_decline = rsi1 - rsi2
                        price_higher_low = price2 > price1
                        HIDDEN_RSI_MIN_DECLINE = 3  # 노이즈 배제용 최소 하락폭 (정석 쪽 임계값과 동일)
                        if price_higher_low and rsi_decline >= HIDDEN_RSI_MIN_DECLINE:
                            stage_divergence = "warn"
                            hidden_bullish_divergence = True
                            divergence_present = True
                            signal_fresh = is_today_new_low
                            hidden_bonus = 0.5 * max(0.0, min(1.0, (rsi_decline - HIDDEN_RSI_MIN_DECLINE) / (DIVERGENCE_RSI_MAX - HIDDEN_RSI_MIN_DECLINE)))
                            divergence_quality = (DIVERGENCE_BASE_SCORE + hidden_bonus) * DIVERGENCE_WARN_CAP
                        else:
                            stage_divergence = "fail"      # RSI도 개선 안 됨
                else:
                    # 오차범위(3%)를 이미 넘어선 상태(가격이 그만큼 반등함).
                    # RSI가 개선됐던 저점쌍이었다면, 이건 "다이버전스 실패"가 아니라
                    # "이미 성공해서 상승추세로 넘어간 것"이므로 위반(fail) 대신 중립으로 본다.
                    # (실제 방향 확인은 uptrend_entry_signal이 담당)
                    stage_divergence = "neutral" if rsi_improved else "fail"

    # 다이버전스 사이클 상태머신: 저점2(다이버전스) -> ±3% 유지 -> 상승추세/무효화 ->
    # (무효화중 마지막음봉 고가 돌파)반등시도 -> (저점2 재돌파)상승추세 -> (저점2 재붕괴)하락추세.
    # uptrend_entry_signal / divergence_invalidated_signal / divergence_stop_signal /
    # downtrend_signal은 전부 이 상태머신 하나의 결과에서 파생된다.
    uptrend_entry_signal = None
    divergence_invalidated_signal = None
    downtrend_signal = None
    DIVERGENCE_STOP_WINDOW_DAYS = 5
    divergence_stop_signal = None

    anchor_pos2 = None
    if len(realtime_lows) >= 2:
        pos2_u = realtime_lows[-1]
        candidates_u = [p for p in realtime_lows
                        if p != pos2_u and MIN_GAP_DAYS <= (pos2_u - p) <= MAX_GAP_DAYS]
        pos1_u = min(candidates_u, key=lambda p: close_values[p]) if candidates_u else None
        if pos1_u is not None:
            price1_u, price2_u = float(c.iloc[pos1_u]), float(c.iloc[pos2_u])
            rsi1_u, rsi2_u = float(rsi.iloc[pos1_u]), float(rsi.iloc[pos2_u])
            # "정석"(price2<price1)이든 "이중바닥"(price2가 price1보다 살짝 높지만
            # DOUBLE_BOTTOM_MAX_PREMIUM_PCT 이내)이든, RSI가 개선됐으면 둘 다 유효한
            # 다이버전스 앵커로 인정한다 (3단계 게이트와 동일 기준 - PFE 사례에서
            # "정석"만 체크해서 진짜 최근 다이버전스를 놓치던 버그 수정).
            if (not (pd.isna(rsi1_u) or pd.isna(rsi2_u)) and rsi2_u > rsi1_u
                    and price2_u <= price1_u * (1 + DOUBLE_BOTTOM_MAX_PREMIUM_PCT)):
                anchor_pos2 = pos2_u  # 가장 최근 저점쌍 자체가 다이버전스(정석 또는 이중바닥)

        # 가장 최근 저점쌍이 다이버전스가 아니었다면, 그 이전 것들 중 가장 가까운
        # 다이버전스(정석 또는 이중바닥)를 30일 이내 범위에서 역순으로 찾는다.
        if anchor_pos2 is None:
            for idx in range(len(realtime_lows) - 2, -1, -1):
                candidate_pos2 = realtime_lows[idx]
                if pos2_u - candidate_pos2 > MAX_GAP_DAYS:
                    break
                own_candidates = [p for p in realtime_lows
                                   if p != candidate_pos2 and MIN_GAP_DAYS <= (candidate_pos2 - p) <= MAX_GAP_DAYS]
                own_pos1 = min(own_candidates, key=lambda p: close_values[p]) if own_candidates else None
                if own_pos1 is None:
                    continue
                own_price1, own_price2 = float(c.iloc[own_pos1]), float(c.iloc[candidate_pos2])
                own_rsi1, own_rsi2 = float(rsi.iloc[own_pos1]), float(rsi.iloc[candidate_pos2])
                if pd.isna(own_rsi1) or pd.isna(own_rsi2):
                    continue
                if own_rsi2 > own_rsi1 and own_price2 <= own_price1 * (1 + DOUBLE_BOTTOM_MAX_PREMIUM_PCT):
                    anchor_pos2 = candidate_pos2
                    break

    if anchor_pos2 is not None:
        anchor_price2 = float(c.iloc[anchor_pos2])
        cycle = compute_divergence_cycle_state(anchor_pos2, anchor_price2, o.values, h.values, l.values, close_values)
        today_idx = len(close_values) - 1
        price_today = float(c.iloc[-1])
        gain_since_low = (price_today - anchor_price2) / anchor_price2 if anchor_price2 > 0 else 0
        TREND_PHASE_FRESH_DAYS = 5  # 이 안이면 "진입", 넘으면 "유지중"으로 라벨만 바뀜(상태 자체는 동일)

        if cycle["state"] == "uptrend":
            days_since_uptrend = today_idx - cycle["uptrend_start_day"] if cycle["uptrend_start_day"] is not None else None
            phase = "entry" if (days_since_uptrend is not None and days_since_uptrend <= TREND_PHASE_FRESH_DAYS) else "ongoing"
            uptrend_entry_signal = {
                "active": True,
                "phase": phase,  # "entry"=상승추세 진입(5일 이내), "ongoing"=상승추세 유지중
                "days_since_divergence": today_idx - anchor_pos2,
                "days_since_trend_start": days_since_uptrend,
                "gain_since_low_pct": round(gain_since_low * 100, 1),
                "divergence_date_index": anchor_pos2,
            }
        elif cycle["state"] == "downtrend":
            days_since_downtrend = today_idx - cycle["downtrend_start_day"] if cycle["downtrend_start_day"] is not None else None
            phase = "entry" if (days_since_downtrend is not None and days_since_downtrend <= TREND_PHASE_FRESH_DAYS) else "ongoing"
            downtrend_signal = {
                "active": True,
                "phase": phase,  # "entry"=하락추세 진입(5일 이내), "ongoing"=하락추세 유지중
                "days_since_trend_start": days_since_downtrend,
                "divergence_date_index": anchor_pos2,
                "price_vs_low_pct": round(gain_since_low * 100, 1),
            }
        elif cycle["state"] in ("invalidated", "rescue_attempt"):
            drop_pct = (anchor_price2 - price_today) / anchor_price2 * 100 if anchor_price2 > 0 else 0
            divergence_invalidated_signal = {
                "active": True,
                "prior_divergence_date_index": anchor_pos2,
                "days_since_prior_divergence": today_idx - anchor_pos2,
                "price_drop_since_pct": round(drop_pct, 1),
                "rescue_attempted": cycle["state"] == "rescue_attempt",
            }
            # 손절신호: 무효화 상태(반등시도 전)로 DIVERGENCE_STOP_WINDOW_DAYS일 이상 지속되면
            if cycle["state"] == "invalidated" and cycle["invalidation_day"] is not None:
                days_since_break = today_idx - cycle["invalidation_day"]
                if days_since_break >= DIVERGENCE_STOP_WINDOW_DAYS:
                    divergence_stop_signal = {
                        "active": True,
                        "days_since_break": days_since_break,
                        "reason": f"무효화 이후 {days_since_break}일간 반등시도(음봉고가 돌파) 없음",
                    }

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

    # ADX: ta 라이브러리 내부 구현이 데이터가 너무 짧으면(최근 상장 등) 인덱스 범위를
    # 벗어나면서 크래시한다(예: SPCX, "index 14 out of bounds for axis 0 with size 9").
    # ADX는 참고 표시용일 뿐 점수 계산엔 반영 안 하므로, 실패하면 None으로 두고 넘어간다.
    try:
        adx_last = float(adx_ind.adx().iloc[-1])
    except Exception:
        adx_last = None

    weekly = hist.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    weekly_rsi = RSIIndicator(weekly["Close"], window=14).rsi()

    # 트리거캔들은 "오늘 기준 최근 며칠"이 아니라, 반드시 pos2(가장 최근 저점) 시점부터
    # 그 이후에 발생한 것만 인정한다. AMSC 사례에서 확인됨: 저점(pos2) 확정 전에
    # 나왔던 캔들(가격이 아직 더 빠지기 전의 "가짜 반전 시도")을 트리거로 착각하면 안 됨.
    if pos2 is not None:
        breakout_ok, breakout_pattern, breakout_days_ago = breakout_near_low(o, h, l, c, pos2)
        long_lower_wick, wick_days_ago = wick_near_low(o, h, l, c, pos2)
    else:
        breakout_ok, breakout_pattern, breakout_days_ago = trigger_with_breakout(o, h, l, c)
        long_lower_wick, wick_days_ago = long_lower_wick_recent(o, h, l, c, lookback_days=3)
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
    volume_health = market_volume_health(index_symbol)

    target_mean = info.get("targetMeanPrice")
    forward_pe = info.get("forwardPE")
    peg = info.get("pegRatio")
    sector = info.get("sector")
    upside_pct = round((target_mean / last_close - 1) * 100, 1) if target_mean else None

    sector_rs = None
    if sector in SECTOR_ETF:
        try:
            etf_close = get_confirmed_history(SECTOR_ETF[sector], period="1y")["Close"]
            sector_rs = rs_line(c, etf_close)
        except Exception:
            pass

    # 3단계: 기술적 위치 - Z-score 기반(종목별 개별 계산). RSI 절대값은 참고 표시용.
    # 백테스트 검증: Z<=-1.0일 때 승률 58.1%, Z>-1.0일 때 27.3% (43건 표본)
    # 라벨(pass/warn/fail)은 기존 임계값 그대로 유지하되, 실제 점수 계산은 이산값(1.0/0.5/0.0)이
    # 아니라 min~max 사이 연속값으로 매겨서 "낮을수록 더 좋다"를 세밀하게 반영한다.
    # min(-3.0)에 가까울수록 만점(1.0), max(1.0)에 가까울수록 최저점(0.0).
    ZSCORE_SCALE_MIN = -3.0
    ZSCORE_SCALE_MAX = 1.0
    zscore_quality = None
    if rsi_zscore is not None:
        stage3 = "pass" if rsi_zscore <= -1.5 else \
                 "warn" if rsi_zscore <= -1.0 else "fail"
        clamped = max(ZSCORE_SCALE_MIN, min(ZSCORE_SCALE_MAX, rsi_zscore))
        zscore_quality = (ZSCORE_SCALE_MAX - clamped) / (ZSCORE_SCALE_MAX - ZSCORE_SCALE_MIN)
    else:
        stage3 = "fail"

    # 2단계: 다이버전스 게이트 - stage_divergence는 위에서 이미 pass/warn/fail로 계산됨

    # 4단계(ADX)는 백테스트 재검증 결과 두 독립 표본에서 방향이 계속 뒤집혀 폐기함.
    # adx_last 값 자체는 참고용으로 남겨두되, 점수 계산에는 반영하지 않음 (나중에 재검토).

    # 오더블럭: 저점2 캔들의 몸통(시가~종가)을 그 이후 며칠(최대 3일) 안에 완전히
    # 뒤덮는(engulf) 캔들이 나오면, 두 캔들 몸통의 교집합 구간을 오더블럭으로 잡는다.
    # 뒤덮는 캔들의 몸통이 저점2 몸통의 2배 이상이면 "강한" 오더블럭(돌파 확정 없이도
    # pass 인정), 아니면 "약한" 오더블럭(warn)으로 구분한다. 오더블럭이 확정된 이후
    # 오늘까지 종가가 그 하단 아래로 내려간 적이 있으면 "이탈중"으로 판정한다.
    order_block = None
    if pos2 is not None:
        o2, c2v = float(o.iloc[pos2]), float(c.iloc[pos2])
        body2_low, body2_high = min(o2, c2v), max(o2, c2v)
        body2_size = body2_high - body2_low
        for idx in range(pos2 + 1, min(pos2 + 4, len(c))):
            o_t, c_t = float(o.iloc[idx]), float(c.iloc[idx])
            bodyT_low, bodyT_high = min(o_t, c_t), max(o_t, c_t)
            covers = bodyT_low <= body2_low and bodyT_high >= body2_high
            if covers:
                ob_low, ob_high = body2_low, body2_high  # 완전 포함이므로 교집합 = 저점2 몸통
                bodyT_size = bodyT_high - bodyT_low
                strong = bool(body2_size > 0 and bodyT_size >= 2 * body2_size)
                future_close = c.iloc[idx + 1:]
                breached = bool((future_close < ob_low).any()) if len(future_close) > 0 else False
                order_block = {
                    "low": round(ob_low, 2), "high": round(ob_high, 2),
                    "strong": strong, "breach": breached,
                    "trigger_day_idx": idx,
                }
                break

    # 트리거캔들 거래량 확인: 트리거캔들 당일 거래량이, 그 다이버전스 구간(저점1~저점2)의
    # 평균 거래량보다 높으면 "거래량 실린 트리거"로 보고 warn을 pass로 승격시킨다.
    trigger_day_idx = None
    if breakout_ok:
        trigger_day_idx = len(c) - 1 - breakout_days_ago
    elif long_lower_wick:
        trigger_day_idx = len(c) - 1 - wick_days_ago

    trigger_volume_confirmed = False
    avg_vol_divergence_window = None
    trigger_day_volume = None
    if trigger_day_idx is not None and pos1 is not None and pos2 is not None and pos1 <= pos2:
        avg_vol_divergence_window = float(v.iloc[pos1:pos2 + 1].mean())
        trigger_day_volume = float(v.iloc[trigger_day_idx])
        trigger_volume_confirmed = bool(trigger_day_volume > avg_vol_divergence_window)

    # 4단계: 트리거캔들
    # 백테스트 검증: 돌파확정 pass / 긴아래꼬리만 있어도 약한 warn(+2.6%p, 표본158건) / 없으면 fail
    # 긴아래꼬리(warn)에 거래량까지 실렸다면 pass로 승격 (이론적 근거: 매수세 유입 강도가
    # 돌파확정과 비슷한 수준의 신뢰도를 가진다고 보되, 아직 백테스트 검증 전 - 추후 확인 필요)
    if breakout_ok:
        stage_trigger = "pass"
    elif long_lower_wick and trigger_volume_confirmed:
        stage_trigger = "pass"
    elif order_block is not None and order_block["strong"] and not order_block["breach"]:
        stage_trigger = "pass"   # 뒤덮는 캔들 몸통이 저점2의 2배 이상 - 돌파 확정 없이도 강한 신호로 인정
    elif order_block is not None and not order_block["breach"]:
        stage_trigger = "warn"   # 뒤덮는 캔들은 있지만 몸통이 약함(2배 미만) - 긴아래꼬리와 동급
    elif long_lower_wick:
        stage_trigger = "warn"
    else:
        stage_trigger = "fail"

    stop_pct = round(float(atr.iloc[-1]) * 1.5 / last_close * 100, 1)

    # 1단계: 펀더멘털 - PEG가 좋을수록 업사이드 커트라인을 낮춰줌 (서로 보완)
    # 업사이드(애널리스트 목표가) 데이터 자체가 없는 종목은 PEG만으로 판정한다.
    if upside_pct is None:
        if peg is not None and peg <= 1.0:
            stage1 = "pass"
        elif peg is not None and peg <= 2.0:
            stage1 = "warn"
        else:
            stage1 = "fail"
    else:
        if peg is not None and peg <= 1.0:
            upside_threshold = 5
        elif peg is not None and peg <= 1.5:
            upside_threshold = 10
        else:
            upside_threshold = 15

        stage1_flags = {
            "upside_ge_threshold": upside_pct >= upside_threshold,
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
    sma60 = c.rolling(60).mean()
    sma120 = c.rolling(120).mean()
    sma200 = c.rolling(200).mean()

    PULLBACK_MA_TOLERANCE = 0.03
    dist_sma20 = abs(last_close - sma20.iloc[-1]) / sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else None
    dist_sma40 = abs(last_close - sma40_addon.iloc[-1]) / sma40_addon.iloc[-1] if not pd.isna(sma40_addon.iloc[-1]) else None
    dist_sma120 = abs(last_close - sma120.iloc[-1]) / sma120.iloc[-1] if not pd.isna(sma120.iloc[-1]) else None
    dist_sma200 = abs(last_close - sma200.iloc[-1]) / sma200.iloc[-1] if not pd.isna(sma200.iloc[-1]) else None
    dist_sma60_val = abs(last_close - sma60.iloc[-1]) / sma60.iloc[-1] if not pd.isna(sma60.iloc[-1]) else None
    near_sma20_raw = bool(dist_sma20 is not None and dist_sma20 <= PULLBACK_MA_TOLERANCE)
    near_sma40_raw = bool(dist_sma40 is not None and dist_sma40 <= PULLBACK_MA_TOLERANCE)
    near_sma60_raw = bool(dist_sma60_val is not None and dist_sma60_val <= PULLBACK_MA_TOLERANCE)
    near_sma120_raw = bool(dist_sma120 is not None and dist_sma120 <= PULLBACK_MA_TOLERANCE)
    near_sma200_raw = bool(dist_sma200 is not None and dist_sma200 <= PULLBACK_MA_TOLERANCE)

    # "근접"만으로는 위에서 내려와 닿은 건지, 아래에서 올라와 닿은 건지 구분이 안 된다.
    # 진짜 눌림목은 "최근에 이평선 위에 있다가 지금 눌린 것"이어야 하므로,
    # 최근 10일(오늘 제외) 중 종가가 그 이평선보다 확실히(3% 이상) 위에 있었던 적이
    # 있어야만 근접을 인정한다.
    RECENT_ABOVE_LOOKBACK = 10
    RECENT_ABOVE_MARGIN = 0.03
    recent_close = c.iloc[-RECENT_ABOVE_LOOKBACK - 1:-1]
    recent_sma20 = sma20.iloc[-RECENT_ABOVE_LOOKBACK - 1:-1]
    recent_sma40 = sma40_addon.iloc[-RECENT_ABOVE_LOOKBACK - 1:-1]
    recent_sma120 = sma120.iloc[-RECENT_ABOVE_LOOKBACK - 1:-1]
    recent_sma200 = sma200.iloc[-RECENT_ABOVE_LOOKBACK - 1:-1]
    recent_sma60 = sma60.iloc[-RECENT_ABOVE_LOOKBACK - 1:-1]
    was_above_20_recently = bool(((recent_close > recent_sma20 * (1 + RECENT_ABOVE_MARGIN)).any())) if len(recent_close) > 0 else False
    was_above_40_recently = bool(((recent_close > recent_sma40 * (1 + RECENT_ABOVE_MARGIN)).any())) if len(recent_close) > 0 else False
    was_above_60_recently = bool(((recent_close > recent_sma60 * (1 + RECENT_ABOVE_MARGIN)).any())) if len(recent_close) > 0 else False
    was_above_120_recently = bool(((recent_close > recent_sma120 * (1 + RECENT_ABOVE_MARGIN)).any())) if len(recent_close) > 0 else False
    was_above_200_recently = bool(((recent_close > recent_sma200 * (1 + RECENT_ABOVE_MARGIN)).any())) if len(recent_close) > 0 else False

    # "위에 있다가 살짝 눌린 것"과 "위에 있다가 확실히 무너져서 아래로 갔다가 반등하며
    # 다시 그 선에 닿은 것"은 다르다 - 후자는 눌림목이 아니라 V자 반등이므로 제외한다.
    # 최근 5일 안에 종가가 그 선보다 3% 이상 아래로 떨어진 적이 있으면 무효화.
    RECENT_BREAK_LOOKBACK = 5
    RECENT_BREAK_MARGIN = 0.03
    recent_close_5 = c.iloc[-RECENT_BREAK_LOOKBACK - 1:-1]
    recent_sma20_5 = sma20.iloc[-RECENT_BREAK_LOOKBACK - 1:-1]
    recent_sma40_5 = sma40_addon.iloc[-RECENT_BREAK_LOOKBACK - 1:-1]
    recent_sma60_5 = sma60.iloc[-RECENT_BREAK_LOOKBACK - 1:-1]
    recent_sma120_5 = sma120.iloc[-RECENT_BREAK_LOOKBACK - 1:-1]
    recent_sma200_5 = sma200.iloc[-RECENT_BREAK_LOOKBACK - 1:-1]
    broke_below_20_recently = bool(((recent_close_5 < recent_sma20_5 * (1 - RECENT_BREAK_MARGIN)).any())) if len(recent_close_5) > 0 else False
    broke_below_40_recently = bool(((recent_close_5 < recent_sma40_5 * (1 - RECENT_BREAK_MARGIN)).any())) if len(recent_close_5) > 0 else False
    broke_below_60_recently = bool(((recent_close_5 < recent_sma60_5 * (1 - RECENT_BREAK_MARGIN)).any())) if len(recent_close_5) > 0 else False
    broke_below_120_recently = bool(((recent_close_5 < recent_sma120_5 * (1 - RECENT_BREAK_MARGIN)).any())) if len(recent_close_5) > 0 else False
    broke_below_200_recently = bool(((recent_close_5 < recent_sma200_5 * (1 - RECENT_BREAK_MARGIN)).any())) if len(recent_close_5) > 0 else False

    near_sma20 = near_sma20_raw and was_above_20_recently and not broke_below_20_recently
    near_sma40 = near_sma40_raw and was_above_40_recently and not broke_below_40_recently
    near_sma60 = near_sma60_raw and was_above_60_recently and not broke_below_60_recently
    near_sma120 = near_sma120_raw and was_above_120_recently and not broke_below_120_recently
    near_sma200 = near_sma200_raw and was_above_200_recently and not broke_below_200_recently

    # 60일선이 20/40일선보다 가장 아래에 있는 상태에서, 20일선과 40일선 "둘 다" 함께
    # 60일선에 근접(수렴)해오면 추세 힘빠짐으로 보고 무시한다.
    # (하나만 수렴한 경우는 아직 전체 추세가 흔들리는 것으로 보기 어려워 무시하지 않음)
    sma20_now_v = float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else None
    sma40_now_v = float(sma40_addon.iloc[-1]) if not pd.isna(sma40_addon.iloc[-1]) else None
    sma60_now_v = float(sma60.iloc[-1]) if not pd.isna(sma60.iloc[-1]) else None
    sma120_now_v = float(sma120.iloc[-1]) if not pd.isna(sma120.iloc[-1]) else None
    sma200_now_v = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else None
    sma60_is_lowest = bool(sma60_now_v is not None and sma20_now_v is not None and sma40_now_v is not None
                           and sma60_now_v < sma20_now_v and sma60_now_v < sma40_now_v)

    sma20_converged_to_60 = bool(sma60_now_v is not None and sma20_now_v is not None and sma60_now_v > 0
                                  and abs(sma20_now_v - sma60_now_v) / sma60_now_v <= PULLBACK_MA_TOLERANCE)
    sma40_converged_to_60 = bool(sma60_now_v is not None and sma40_now_v is not None and sma60_now_v > 0
                                  and abs(sma40_now_v - sma60_now_v) / sma60_now_v <= PULLBACK_MA_TOLERANCE)
    ma_converged_ignore = sma60_is_lowest and (sma20_converged_to_60 and sma40_converged_to_60)

    # 최근 며칠 사이 이평선 간격(20일선-60일선)이 넓어지고 있는지 좁아지고 있는지 확인.
    # 정배열 여부와 무관하게 계산한다 - 간격(20-60)이 음수(역배열)여도 "나아지는 중인지
    # 나빠지는 중인지" 비교 자체는 그대로 의미가 있다(예: 역배열이어도 간격이 좁혀지는
    # 중이면 정배열 쪽으로 회복 중이라는 뜻).
    # 참고: 60/120/200일선 터치는 그 자체로 이미 20-60일선 간격이 좁혀진 상태를 내포하므로
    # (깊은 하락 없이는 물리적으로 그 선까지 안 내려옴), 이 넓어짐/좁아짐 로직은 20/40일선
    # 근접 케이스에만 적용하고 60/120/200일선은 별도 고정 등급을 쓴다.
    MA_SPREAD_LOOKBACK = 10
    is_jeong_baeyeol = bool(sma20_now_v is not None and sma40_now_v is not None and sma60_now_v is not None
                             and sma20_now_v > sma40_now_v > sma60_now_v)
    ma_spread_widening = False
    if sma20_now_v is not None and sma60_now_v is not None and len(sma20) > MA_SPREAD_LOOKBACK:
        sma20_past_v = sma20.iloc[-1 - MA_SPREAD_LOOKBACK]
        sma60_past_v = sma60.iloc[-1 - MA_SPREAD_LOOKBACK]
        if not (pd.isna(sma20_past_v) or pd.isna(sma60_past_v)):
            spread_now = sma20_now_v - sma60_now_v
            spread_past = float(sma20_past_v) - float(sma60_past_v)
            ma_spread_widening = bool(spread_now > spread_past)

    near_ma_signal = near_sma20 or near_sma40 or near_sma60 or near_sma120 or near_sma200
    near_ma_count = int(near_sma20) + int(near_sma40)  # 1개 또는 2개 근접 (20/40일선 가중치 조정용)

    if ma_converged_ignore and near_ma_signal:
        addon_stage_gate = "fail"  # 60일선이 가장 아래인데 20/40일선이 그쪽으로 수렴 -> 신호 무시
    else:
        addon_stage_gate = "pass" if near_ma_signal else "fail"

    # RSI 존: 60이상 pass, 50~60 warn, 50미만 fail (라벨은 유지)
    # 점수 계산은 이산값(1.0/0.5/0.0) 대신 min~max 사이 연속값으로, RSI가 높을수록(60 방향)
    # 더 좋게, 낮을수록(50 미만 방향) 더 나쁘게 세밀하게 반영한다.
    RSI_ZONE_SCALE_MIN = 40   # 이 이하면 최저점(0.0)
    RSI_ZONE_SCALE_MAX = 65   # 이 이상이면 만점(1.0)
    if rsi_last >= 60:
        addon_stage_rsi = "pass"
    elif rsi_last >= 50:
        addon_stage_rsi = "warn"
    else:
        addon_stage_rsi = "fail"
    rsi_zone_clamped = max(RSI_ZONE_SCALE_MIN, min(RSI_ZONE_SCALE_MAX, rsi_last))
    rsi_zone_quality = (rsi_zone_clamped - RSI_ZONE_SCALE_MIN) / (RSI_ZONE_SCALE_MAX - RSI_ZONE_SCALE_MIN)

    # 히든 다이버전스(가격 higher low + RSI lower low)는 RSI 절대구간과 무관하게 모멘텀이
    # 살아있다는 독립적 증거이므로, 4단계를 충족(만점)으로 끌어올린다.
    # (3단계 눌림목게이트/ma_converged_ignore와는 완전히 분리되어 있어 그쪽 무효화 로직의 영향을 안 받음)
    if hidden_bullish_divergence:
        addon_stage_rsi = "pass"
        rsi_zone_quality = 1.0

    # 4단계: 트리거캔들 + 거래량 (2/3단계와 완전히 독립 - 이평선/RSI 조건 없음, 순수 캔들+거래량만 봄)
    # hammer, engulfing, long_lower_wick, wick_days_ago는 신규진입 5단계에서 이미 계산된 값을 재사용.
    VOL_LOOKBACK_20 = 20

    def _volume_confirmed_for_day(day_idx):
        if day_idx is None or day_idx < VOL_LOOKBACK_20:
            return False
        avg_vol = float(v.iloc[day_idx - VOL_LOOKBACK_20:day_idx].mean())
        day_vol = float(v.iloc[day_idx])
        return bool(avg_vol > 0 and day_vol > avg_vol)

    has_reversal_candle = bool(hammer or engulfing)  # 오늘 기준
    reversal_day_idx = len(c) - 1
    reversal_volume_confirmed = _volume_confirmed_for_day(reversal_day_idx) if has_reversal_candle else False

    wick_present = bool(long_lower_wick)  # 0~3일 이내
    wick_day_idx = (len(c) - 1 - wick_days_ago) if (wick_present and wick_days_ago is not None) else None
    wick_volume_confirmed = _volume_confirmed_for_day(wick_day_idx) if wick_day_idx is not None else False

    trigger_weight_tier = None
    if has_reversal_candle and reversal_volume_confirmed:
        addon_stage_trigger = "pass"
        trigger_weight_tier = "reversal_volume"   # 가장 높은 가중치
    elif has_reversal_candle:
        addon_stage_trigger = "pass"
        trigger_weight_tier = "reversal_only"     # 중간 가중치
    elif wick_present and wick_volume_confirmed:
        addon_stage_trigger = "pass"
        trigger_weight_tier = "wick_volume"        # 가장 낮은 pass 가중치
    elif wick_present:
        addon_stage_trigger = "warn"                # 거래량 없는 긴아래꼬리만
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
        "1_market_health": {"status": volume_health["status"], "index_name": index_name, "index_symbol": index_symbol,
                             "etf": volume_health["etf"], "volume_diff_pct_3m": volume_health["diff_pct"],
                             "volume_trend_pct_3d": volume_health["trend_pct"], "volume_trend_up": volume_health["trend_up"],
                             "daily_change_pct": volume_health["daily_change_pct"], "volatility_label": volume_health["volatility_label"],
                             "index_trigger_candle": volume_health["index_trigger_candle"],
                             "golden_cross_recent": volume_health["golden_cross_recent"], "dead_cross_recent": volume_health["dead_cross_recent"]},
        "2_fundamentals": {"status": stage1, "upside_pct": upside_pct, "forward_pe": forward_pe, "peg": peg},
        "3_pullback_gate": {"status": addon_stage_gate, "near_sma20": near_sma20, "near_sma40": near_sma40, "near_sma60": near_sma60, "near_sma120": near_sma120, "near_sma200": near_sma200, "near_ma_count": near_ma_count, "ma_converged_ignored": bool(ma_converged_ignore and near_ma_signal), "was_above_20_recently": was_above_20_recently, "was_above_40_recently": was_above_40_recently, "is_jeong_baeyeol": is_jeong_baeyeol, "ma_spread_widening": ma_spread_widening},
        "4_rsi_zone": {"status": addon_stage_rsi, "rsi": round(rsi_last, 1), "rsi_zone_quality": round(rsi_zone_quality, 3), "hidden_bullish_divergence": hidden_bullish_divergence},
        "5_trigger_confirmed": {"status": addon_stage_trigger, "has_reversal_candle": has_reversal_candle,
                                 "reversal_volume_confirmed": reversal_volume_confirmed,
                                 "wick_present": wick_present, "wick_volume_confirmed": wick_volume_confirmed,
                                 "trigger_weight_tier": trigger_weight_tier},
        "9_position_sizing": {"stop_pct_from_20sma": addon_stop_pct},
    }
    addon_known_weights = {k: ADDON_WEIGHTS[k] for k in ADDON_WEIGHTS if stages_addon[k]["status"] != "unknown"}
    # 3단계(눌림목게이트, 구 2단계): pass는 그대로 두되, 근접한 이평선 종류에 따라 가중치를 차등 적용.
    # 60/120/200일선 터치는 그 자체로 이례적으로 깊은 눌림이라(정상적인 조정으로는 잘 안 옴,
    # 물리적으로 20-60일선 간격이 이미 좁혀진 상태를 내포) 넓어짐/좁아짐 로직 없이 고정 등급.
    # 20/40일선만 기존처럼 간격 넓어짐/좁아짐으로 세분화(정배열 여부 우선 체크).
    # 우선순위(가장 깊은 눌림 우선): 200일선 > 120일선 > 60일선 > 40일선 > 20일선
    if "3_pullback_gate" in addon_known_weights and stages_addon["3_pullback_gate"]["status"] == "pass":
        if near_sma200:
            gate_multiplier = 1.00      # 200일선 (원래 1.4 -> 1.0 기준 비례축소)
        elif near_sma120:
            gate_multiplier = 0.86      # 120일선 (원래 1.2)
        elif near_sma60:
            gate_multiplier = 0.71      # 60일선 (원래 1.0)
        elif not is_jeong_baeyeol:
            gate_multiplier = 0.36      # 20/40일선만 근접 + 역배열 (원래 0.5)
        elif near_sma40:  # 40일선 근접(둘 다 근접이어도 40일선 쪽 값 우선)
            gate_multiplier = 0.57 if ma_spread_widening else 0.54  # 원래 0.8 / 0.75
        else:  # 20일선만 근접
            gate_multiplier = 0.54 if ma_spread_widening else 0.36  # 원래 0.75 / 0.5
        addon_known_weights["3_pullback_gate"] = addon_known_weights["3_pullback_gate"] * gate_multiplier

    # 5단계(트리거, 구 4단계): pass 안에서도 어떤 조합인지에 따라 가중치 차등
    # 긴아래꼬리+거래량(가장낮음) < 반전캔들만(중간) < 반전캔들+거래량(가장높음)
    TRIGGER_WEIGHT_MULTIPLIER = {"wick_volume": 0.5, "reversal_only": 0.75, "reversal_volume": 1.0}
    if "5_trigger_confirmed" in addon_known_weights and stages_addon["5_trigger_confirmed"]["status"] == "pass" and trigger_weight_tier:
        addon_known_weights["5_trigger_confirmed"] = addon_known_weights["5_trigger_confirmed"] * TRIGGER_WEIGHT_MULTIPLIER[trigger_weight_tier]

    if addon_known_weights:
        # 4단계(RSI존, 구 3단계)는 이산값 대신 연속값(rsi_zone_quality)을 그대로 써서
        # "RSI가 60에 가까울수록 더 좋다"를 세밀하게 반영. 나머지는 기존 방식 유지.
        def _addon_stage_score(k):
            if k == "4_rsi_zone":
                return rsi_zone_quality
            return STATUS_SCORE[stages_addon[k]["status"]]
        addon_raw = sum(addon_known_weights[k] * _addon_stage_score(k) for k in addon_known_weights)
        addon_score = round(addon_raw / sum(addon_known_weights.values()) * 100)
    else:
        addon_score = 0

    stages = {
        "1_market_health": {"status": volume_health["status"], "index_name": index_name, "index_symbol": index_symbol,
                             "etf": volume_health["etf"], "volume_diff_pct_3m": volume_health["diff_pct"],
                             "volume_trend_pct_3d": volume_health["trend_pct"], "volume_trend_up": volume_health["trend_up"],
                             "daily_change_pct": volume_health["daily_change_pct"], "volatility_label": volume_health["volatility_label"],
                             "index_trigger_candle": volume_health["index_trigger_candle"],
                             "golden_cross_recent": volume_health["golden_cross_recent"], "dead_cross_recent": volume_health["dead_cross_recent"]},
        "2_fundamentals": {"status": stage1, "upside_pct": upside_pct, "forward_pe": forward_pe, "peg": peg},
        "3_divergence_gate": {"status": stage_divergence, "bullish_divergence": bullish_divergence, "hidden_bullish_divergence": hidden_bullish_divergence, "recent_bearish_bonus_applied": recent_bearish_bonus_applied, "divergence_present": divergence_present, "gap_days": gap_days, "signal_fresh": signal_fresh, "divergence_quality": round(divergence_quality, 3) if divergence_quality is not None else None, "rsi_improvement": round(rsi_improvement, 1) if rsi_improvement is not None else None},
        "4_zscore": {"status": stage3, "rsi": round(rsi_last, 1), "rsi_zscore_1y": round(rsi_zscore, 2) if rsi_zscore is not None else None, "zscore_quality": round(zscore_quality, 3) if zscore_quality is not None else None},
        "5_trigger_candle": {"status": stage_trigger, "breakout_confirmed": breakout_ok, "pattern": breakout_pattern, "days_ago": breakout_days_ago, "hammer": bool(hammer), "bullish_engulfing": bool(engulfing), "morning_star": bool(morning_star), "long_lower_wick": bool(long_lower_wick), "wick_days_ago": wick_days_ago, "adx_reference": round(adx_last, 1) if adx_last is not None else None, "volume_confirmed": trigger_volume_confirmed, "trigger_day_volume": round(trigger_day_volume) if trigger_day_volume else None, "avg_volume_divergence_window": round(avg_vol_divergence_window) if avg_vol_divergence_window else None, "order_block": order_block},
    }

    known_weights = {k: WEIGHTS[k] for k in WEIGHTS if stages[k]["status"] != "unknown"}
    if known_weights:
        # 4단계(Z-score), 3단계(다이버전스)는 이산값(1.0/0.5/0.0) 대신 연속값을 써서
        # "얼마나 강한 신호인지"를 세밀하게 반영한다. 나머지 단계는 기존 방식 유지.
        def _stage_score(k):
            if k == "4_zscore" and zscore_quality is not None:
                return zscore_quality
            if k == "3_divergence_gate" and divergence_quality is not None:
                return divergence_quality
            return STATUS_SCORE[stages[k]["status"]]
        raw_score = sum(known_weights[k] * _stage_score(k) for k in known_weights)
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
        "downtrend_signal": downtrend_signal,
        "pullback_stop_signal": pullback_stop_signal,
        "divergence_stop_signal": divergence_stop_signal,
        "today_reversal_candle": has_reversal_candle,          # 오늘 해머/엔걸핑 여부 (손절 완화 판정용)
        "today_reversal_volume_confirmed": reversal_volume_confirmed,
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


def compute_signal(ticker, today_score, score_key="score", today_date=None):
    """매수신호/신호없음 판정. score_key: 'score' 또는 'score_addon'.
    반환값에 change(전일대비 변화량), prev_score(어제 점수)도 항상 포함한다.

    핵심: 구간을 하나 넘었다는 사실만으로는 부족하다(예: 빨강21→주황48은 구간은 넘었지만
    여전히 warn일 뿐 진짜 매수신호가 아님). 반드시 "오늘 실제로 초록(pass,65+)에
    도달했는지"를 게이트로 걸어서, 그걸 통과했을 때만 매수신호 하위조건(급등/구간돌파/연속상승)을 본다.

    매도(하락) 신호는 별도로 계산하지 않는다 - divergence_stop_signal(손절신호)이
    이미 그 역할(위험 경고)을 담당하고 있어 중복이기 때문.

    today_date: 오늘(as_of_date). 이 날짜와 정확히 같은 히스토리 기록은 "어제"로 쓰지
    않는다 - 같은 거래일에 워크플로우를 여러 번 재실행하면 history에 오늘자 기록이
    먼저 쌓여있을 수 있는데, 그걸 자기 자신과 비교해버리는 버그(change=0)를 방지한다."""
    hist = load_recent_history(ticker, days=5)
    if today_date is not None:
        hist = [h for h in hist if h["date"] != today_date]
    if not hist:
        return {"signal": "none", "reasons": [], "change": None, "prev_score": None}

    prev = hist[-1][score_key]
    change = today_score - prev

    def zone(s):
        cutoff = 70 if score_key == "score" else 65  # 신규진입은 70으로 상향, 눌림목(addon)은 기존 65 유지
        if s >= cutoff: return 2
        if s >= 40: return 1
        return 0

    today_zone = zone(today_score)
    prev_zone = zone(prev)

    crossed_up = today_zone > prev_zone
    spike_up = change >= 25

    streak_up = len(hist) >= 2 and all(
        hist[i][score_key] < hist[i+1][score_key] for i in range(len(hist)-1)
    ) and today_score > hist[-1][score_key]

    reasons = []
    change_str = f"전일 {prev} → 오늘 {today_score} (전일 대비 {'+' if change>=0 else ''}{change})"

    # 매수신호 게이트: 오늘 실제로 초록(pass) 구간이어야만 아래 하위조건들을 인정한다.
    if today_zone == 2:
        if spike_up:
            reasons.append({"label": "급등 기준 (+25 이상)", "detail": change_str})
        if crossed_up:
            reasons.append({"label": "구간 상향 돌파(초록 진입)", "detail": None})
        if streak_up:
            reasons.append({"label": "3일 연속 상승", "detail": None})

    if reasons:
        return {"signal": "up", "reasons": reasons, "change": change, "prev_score": prev}

    return {"signal": "none", "reasons": [], "change": change, "prev_score": prev}


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


def get_usd_krw_rate():
    """USD/KRW 환율을 가져와서 data/exchange_rate.json에 저장한다.
    매수가를 원화로 입력받아 매일 환율로 환산 표시하는 기능에 사용."""
    try:
        hist = get_confirmed_history("USDKRW=X", period="5d")
        if hist.empty:
            return None
        rate = float(hist["Close"].iloc[-1])
        result = {"rate": round(rate, 2), "updated": pd.Timestamp.now("UTC").isoformat(),
                  "as_of_date": str(hist.index[-1].date())}
        with open(f"{OUT_DIR}/exchange_rate.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    except Exception as e:
        print(f"환율 조회 실패: {e}", file=sys.stderr)
        return None


def main():
    if not os.path.exists(WATCHLIST_PATH):
        print(f"watchlist not found: {WATCHLIST_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(WATCHLIST_PATH) as f:
        tickers = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    exchange_rate = get_usd_krw_rate()
    if exchange_rate:
        print(f"환율 갱신: 1USD = {exchange_rate['rate']}KRW ({exchange_rate['as_of_date']})")

    results = []
    for t in tickers:
        try:
            r = analyze(t)
            r["signal"] = compute_signal(t, r["score"], "score", today_date=r["as_of_date"])
            r["signal_addon"] = compute_signal(t, r["score_addon"], "score_addon", today_date=r["as_of_date"])
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
          "divergence_invalidated_signal": r["divergence_invalidated_signal"],
          "downtrend_signal": r["downtrend_signal"],
          "divergence_stop_signal": r["divergence_stop_signal"],
          "today_reversal_candle": r["today_reversal_candle"],
          "today_reversal_volume_confirmed": r["today_reversal_volume_confirmed"]}
         for r in results],
        key=lambda x: x["score"], reverse=True
    )
    with open(f"{OUT_DIR}/rankings.json", "w") as f:
        json.dump({"updated": pd.Timestamp.now("UTC").isoformat(), "rankings": rankings}, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(results)}/{len(tickers)}개 성공, rankings.json 저장")


if __name__ == "__main__":
    main()
