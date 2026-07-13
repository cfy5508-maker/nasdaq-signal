"""
confirmed_data_check.py
사용법: python confirmed_data_check.py [TICKER]
기본값: TICKER=RKLB

동작: fetch_indicators.py의 analyze()를 트리밍 전(원본) / 트리밍 후(확정) 두 버전으로
      각각 실행해서, 1~5단계 전체 결과가 어떻게 달라지는지 비교한다.
      (data/<TICKER>.json은 건드리지 않음 - write_file=False로 안전하게 실행)
"""
import sys
import json
import pandas as pd
import yfinance as yf

import fetch_indicators as fi


def analyze_with_option(ticker, force_no_trim):
    """fetch_indicators.analyze()를 그대로 쓰되, force_no_trim=True면
    get_confirmed_history의 트리밍을 우회해서 원본(장중포함) 데이터로 계산하도록
    임시로 _market_already_closed_today를 True로 패치한다."""
    original_func = fi._market_already_closed_today
    if force_no_trim:
        fi._market_already_closed_today = lambda: True  # 항상 "마감됨"으로 위장 -> 트리밍 안 함
    try:
        result = fi.analyze(ticker, trim_days=0, write_file=False)
    finally:
        fi._market_already_closed_today = original_func
    return result


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "RKLB"

    now_et, closed = None, None
    try:
        from zoneinfo import ZoneInfo
        now_et = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
    except Exception:
        now_et = pd.Timestamp.now(tz="US/Eastern")
    closed = fi._market_already_closed_today()

    print(f"=== 시각 확인 ===")
    print(f"지금 미 동부시간(ET): {now_et}")
    print(f"장마감 판정(현재 실제 상태): {closed}\n")

    print(f"=== {ticker}: 원본(트리밍 없음, 장중 포함 가능) vs 확정(트리밍 적용) 비교 ===\n")

    r_raw = analyze_with_option(ticker, force_no_trim=True)
    r_confirmed = analyze_with_option(ticker, force_no_trim=False)

    print(f"[기준일자]")
    print(f"  원본 as_of_date: {r_raw['as_of_date']}")
    print(f"  확정 as_of_date: {r_confirmed['as_of_date']}")
    print(f"  가격: 원본={r_raw['price']} / 확정={r_confirmed['price']}\n")

    print(f"[최종 점수]")
    print(f"  원본 score: {r_raw['score']}  |  확정 score: {r_confirmed['score']}")
    print(f"  원본 score_addon: {r_raw['score_addon']}  |  확정 score_addon: {r_confirmed['score_addon']}\n")

    stage_labels = {
        "1_market_health": "1단계 시장건전성",
        "2_fundamentals": "2단계 펀더멘털",
        "3_divergence_gate": "3단계 다이버전스",
        "4_zscore": "4단계 Z-score",
        "5_trigger_candle": "5단계 트리거캔들",
    }
    for key, label in stage_labels.items():
        raw_s = r_raw["stages"].get(key, {})
        conf_s = r_confirmed["stages"].get(key, {})
        print(f"[{label}]")
        print(f"  원본: {json.dumps(raw_s, ensure_ascii=False)}")
        print(f"  확정: {json.dumps(conf_s, ensure_ascii=False)}")
        same = raw_s == conf_s
        print(f"  동일여부: {same}\n")

    print(f"[눌림목 단계]")
    addon_labels = {
        "1_fundamentals": "1단계 펀더멘털",
        "2_pullback_gate": "2단계 눌림목게이트",
        "3_rsi_zone": "3단계 RSI존",
        "4_trigger_confirmed": "4단계 셋업/트리거",
    }
    for key, label in addon_labels.items():
        raw_s = r_raw["stages_addon"].get(key, {})
        conf_s = r_confirmed["stages_addon"].get(key, {})
        print(f"  {label}")
        print(f"    원본: {json.dumps(raw_s, ensure_ascii=False)}")
        print(f"    확정: {json.dumps(conf_s, ensure_ascii=False)}")
        print(f"    동일여부: {raw_s == conf_s}")


if __name__ == "__main__":
    main()
