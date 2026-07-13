"""
pullback_score_check.py
사용법: python pullback_score_check.py TICKER

동작: fetch_indicators.py의 새 눌림목 4단계(트리거캔들+거래량) 로직과
      전체 눌림목점수 계산을 그대로 재현해서 출력한다.
"""
import sys
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

PULLBACK_MA_TOLERANCE = 0.03
VOL_LOOKBACK_20 = 20

ADDON_WEIGHTS = {
    "1_fundamentals": 1.5,
    "2_pullback_gate": 2.5,
    "3_rsi_zone": 1.0,
    "4_trigger_confirmed": 2.0,
}
STATUS_SCORE = {"pass": 1.0, "warn": 0.5, "fail": 0.0}


def detect_hammer(o, h, l, c):
    body = abs(c.iloc[-1] - o.iloc[-1])
    lower_wick = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
    upper_wick = h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])
    return bool(body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5)


def detect_bullish_engulfing(o, h, l, c):
    if len(c) < 2:
        return False
    return bool(c.iloc[-1] > o.iloc[-2] and o.iloc[-1] < c.iloc[-2] and c.iloc[-1] > c.iloc[-2] and o.iloc[-2] > c.iloc[-2])


def detect_long_lower_wick(o, h, l, c):
    body = abs(c.iloc[-1] - o.iloc[-1])
    lower_wick = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
    total_range = h.iloc[-1] - l.iloc[-1]
    if total_range <= 0:
        return False
    return bool((lower_wick / total_range) >= 0.5)


def long_lower_wick_recent(o, h, l, c, lookback_days=3):
    n = len(c)
    for days_ago in range(0, lookback_days + 1):
        idx = n - 1 - days_ago
        if idx < 0:
            break
        sub_o, sub_h, sub_l, sub_c = o.iloc[:idx+1], h.iloc[:idx+1], l.iloc[:idx+1], c.iloc[:idx+1]
        if detect_long_lower_wick(sub_o, sub_h, sub_l, sub_c):
            return True, days_ago
    return False, None


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "TTWO"
    hist = yf.Ticker(ticker).history(period="1y")
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    rsi = RSIIndicator(c, window=14).rsi()
    sma20 = c.rolling(20).mean()
    sma40 = c.rolling(40).mean()

    last_close = float(c.iloc[-1])
    rsi_last = float(rsi.iloc[-1])

    print(f"=== {ticker} 눌림목점수 재계산 ===\n")
    print(f"오늘 종가: {last_close:.2f}, RSI: {rsi_last:.1f}\n")

    # 1단계는 펀더멘털(생략 - 화면에서 이미 pass로 확인됨, 여기선 pass로 가정)
    # 2단계
    dist20 = abs(last_close - sma20.iloc[-1]) / sma20.iloc[-1]
    dist40 = abs(last_close - sma40.iloc[-1]) / sma40.iloc[-1]
    near20 = dist20 <= PULLBACK_MA_TOLERANCE
    near40 = dist40 <= PULLBACK_MA_TOLERANCE
    near_count = int(near20) + int(near40)
    stage2_status = "pass" if near_count >= 1 else "fail"
    print(f"[2단계] 20일선거리{dist20*100:.2f}%(근접={near20}) 40일선거리{dist40*100:.2f}%(근접={near40}) => {stage2_status}, 근접개수={near_count}")

    # 3단계
    if rsi_last >= 60:
        stage3_status = "pass"
    elif rsi_last >= 50:
        stage3_status = "warn"
    else:
        stage3_status = "fail"
    print(f"[3단계] RSI {rsi_last:.1f} => {stage3_status}")

    # 4단계
    hammer = detect_hammer(o, h, l, c)
    engulfing = detect_bullish_engulfing(o, h, l, c)
    has_reversal = hammer or engulfing
    wick_present, wick_days_ago = long_lower_wick_recent(o, h, l, c, lookback_days=3)

    def vol_confirmed(day_idx):
        if day_idx is None or day_idx < VOL_LOOKBACK_20:
            return False
        avg = float(v.iloc[day_idx-VOL_LOOKBACK_20:day_idx].mean())
        today_vol = float(v.iloc[day_idx])
        return bool(avg > 0 and today_vol > avg)

    reversal_vol_ok = vol_confirmed(len(c)-1) if has_reversal else False
    wick_day_idx = (len(c)-1-wick_days_ago) if (wick_present and wick_days_ago is not None) else None
    wick_vol_ok = vol_confirmed(wick_day_idx) if wick_day_idx is not None else False

    tier = None
    if has_reversal and reversal_vol_ok:
        stage4_status, tier = "pass", "reversal_volume"
    elif has_reversal:
        stage4_status, tier = "pass", "reversal_only"
    elif wick_present and wick_vol_ok:
        stage4_status, tier = "pass", "wick_volume"
    elif wick_present:
        stage4_status = "warn"
    else:
        stage4_status = "fail"

    print(f"[4단계] 해머={hammer} 엔걸핑={engulfing} 긴아래꼬리={wick_present}({wick_days_ago}일전) 반전거래량확인={reversal_vol_ok} 꼬리거래량확인={wick_vol_ok} => {stage4_status}, tier={tier}\n")

    # 점수 계산 (1단계는 pass로 가정)
    weights = dict(ADDON_WEIGHTS)
    if stage2_status == "pass":
        if near_count == 2:
            gate_mult = 1.0
        elif near40:
            gate_mult = 0.75
        else:
            gate_mult = 0.5
        weights["2_pullback_gate"] *= gate_mult
        print(f"2단계 가중치 배수: {gate_mult} -> {weights['2_pullback_gate']}")

    TRIGGER_MULT = {"wick_volume": 0.5, "reversal_only": 0.75, "reversal_volume": 1.0}
    if stage4_status == "pass" and tier:
        weights["4_trigger_confirmed"] *= TRIGGER_MULT[tier]
        print(f"4단계 가중치 배수: {TRIGGER_MULT[tier]} -> {weights['4_trigger_confirmed']}")

    statuses = {"1_fundamentals": "pass", "2_pullback_gate": stage2_status, "3_rsi_zone": stage3_status, "4_trigger_confirmed": stage4_status}
    raw = sum(weights[k] * STATUS_SCORE[statuses[k]] for k in weights)
    score = round(raw / sum(weights.values()) * 100)
    print(f"\n=== 최종 눌림목점수(1단계 pass 가정): {score} ===")


if __name__ == "__main__":
    main()
