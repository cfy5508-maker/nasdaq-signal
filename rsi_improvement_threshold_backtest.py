"""
rsi_improvement_threshold_backtest.py
사용법: python rsi_improvement_threshold_backtest.py [FORWARD_DAYS] [MODE]
기본값: FORWARD_DAYS=60, MODE=confirmed

MODE:
  - confirmed(기본): 저점2가 "확정된 바닥"인지 확인한다(앞으로 order일 안에도 더
    낮은 종가가 없어야 함) - 백테스트라 미래를 볼 수 있어서 가능한 방식.
    이게 없으면(realtime 모드), 아직 하락이 안 끝난 도중의 스냅샷이 저점으로
    잘못 잡혀서, "그 다음날 저점이 더 깨진 것"을 "정석 다이버전스의 실패"로
    잘못 카운트하게 된다(실제로는 애초에 바닥 시도가 아니었던 것).
  - realtime: fetch_indicators.py 운영 코드와 동일한 방식(미래 미참조, 저점
    확정을 기다리지 않음) - 운영 코드가 실시간 신호라 원래부터 이 한계가 있고,
    대신 "다이버전스 무효화" 신호로 사후 대응하는 구조.

동작: fetch_indicators.py 3단계 게이트의 다이버전스(정석: 가격 lower low + RSI 개선,
이중바닥: 가격은 저점1의 +10% 이내 + RSI 개선) 발생 시마다, RSI 개선폭(포인트)
구간별로 신저점(=실패) 달성률을 정석/이중바닥 각각 따로 뽑아서 비교한다.
(상대값(%) 기준은 이미 검증 결과 변별력이 없어 절대값만 본다)

신저점(실패) 판정: 다이버전스 확정일(low2) 이후 FORWARD_DAYS 거래일 이내에,
가격이 price2보다 낮은 저점을 찍고 그 저점 대비 3% 이상 반등까지 확인되면 '달성'
(bullish_divergence_low_backtest.py와 동일 기준).

종목 범위: 대형주 134 + 소형주 98 = 232종목(기존 백테스트들과 동일 유니버스).
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MODE = sys.argv[2] if len(sys.argv) > 2 else "confirmed"  # "confirmed"(기본, 저점 확정 필터) 또는 "realtime"(기존 방식)

ORDER = 5
MIN_GAP_DAYS = 5           # fetch_indicators.py 운영값과 동일
MAX_GAP_DAYS = 30          # fetch_indicators.py 운영값과 동일
DOUBLE_BOTTOM_MAX_PREMIUM_PCT = 0.10   # 운영 코드와 동일(저점2가 저점1보다 10% 넘게 높으면 제외)
REBOUND_CONFIRM_PCT = 0.03  # 저점 확정에 필요한 최소 반등폭(운영 코드 TOLERANCE_PCT와 동일)
HISTORY_PERIOD = "15y"

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
    "NKE", "PM", "MO", "CL", "KHC", "GILD", "AMGN", "BMY", "LLY", "ABT",
    "DHR", "HON", "CAT", "DE", "MMM", "RTX", "LMT", "NOC", "GD", "FDX",
    "UNP", "NSC", "CMCSA", "TMUS", "CHTR", "EA", "ADP", "INTU", "NOW", "PANW",
    "CRWD", "SNPS", "CDNS", "MU", "AMAT", "LRCX", "KLAC", "ASML", "SHW", "APD",
    "ECL", "DUK", "SO", "D", "AEP", "EXC", "PEG", "WM", "RSG", "PSA",
    "AMT", "PLD", "EQIX", "SPG", "O", "CCI", "SCHW", "AXP", "BLK", "SPGI",
    "MCO", "ICE", "CME", "USB", "PNC", "TFC", "COF", "MET", "PRU", "AIG",
    "TRV", "ALL", "HUM", "CI", "ELV", "CVS", "MDT", "SYK", "BSX", "ISRG",
    "VRTX", "REGN", "ZTS", "BDX",
]
SMALLCAP_TICKERS = [
    "RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
    "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ",
    "FUBO", "PATH", "ASAN", "BBAI", "SOUN", "RUN", "ENVX", "JOBY", "ACHR", "LAZR",
    "QS", "FSLY", "PLUG", "NIO", "XPEV", "BYND", "WOLF", "GPRO", "DNA", "HOOD",
    "AI", "PTON", "SNAP", "LYFT", "DASH", "PINS", "ETSY", "BILL", "TTD", "ROKU",
    "DOCU", "ZM", "TWLO", "OKTA", "DDOG", "NET", "ZS", "MDB", "TEAM", "WDAY",
    "VEEV", "HUBS", "ESTC", "PCTY", "PAYC", "APPF", "SQ", "SHOP", "W", "CHWY",
    "ABNB", "UBER", "MTCH", "BMBL", "YELP", "GRPN", "Z", "ZG", "RDFN", "COMP",
    "CPNG", "SE", "MELI", "JD", "PDD", "BABA", "NTES", "BIDU", "CARG", "VRM",
    "AN", "KMX", "WBD", "PARA", "LYV", "SIRI", "SPOT", "PENN",
]
TICKERS = LARGECAP_TICKERS + SMALLCAP_TICKERS


def find_realtime_lows(values, order=5):
    positions = []
    n = len(values)
    for i in range(order, n):
        window = values[i - order:i]
        if values[i] < window.min():
            positions.append(i)
    return positions


def find_confirmed_lows(values, order=5):
    """find_realtime_lows와 달리 '앞으로 order일 안에도 더 낮은 종가가 없는지'까지
    확인한다(양방향 확정) - 백테스트라 미래를 볼 수 있으니 가능한 방식.
    이러면 '아직 안 끝난 하락 중간의 스냅샷'이 저점으로 잘못 잡히는 걸 막는다."""
    positions = []
    n = len(values)
    for i in range(order, n - order):
        past_window = values[i - order:i]
        future_window = values[i + 1:i + 1 + order]
        if values[i] < past_window.min() and values[i] < future_window.min():
            positions.append(i)
    return positions


def analyze_ticker(ticker, cap, mode="realtime"):
    hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD)
    if hist.empty or len(hist) < 60:
        return []
    close = hist["Close"].values
    rsi = RSIIndicator(pd.Series(close), window=14).rsi().values

    swing_lows = find_confirmed_lows(close, ORDER) if mode == "confirmed" else find_realtime_lows(close, ORDER)

    last_low_idx = None
    last_low_price = None
    last_low_rsi = None
    records = []

    for idx in swing_lows:
        price2, rsi2 = close[idx], rsi[idx]
        if np.isnan(rsi2):
            continue

        if (last_low_idx is not None
                and MIN_GAP_DAYS <= (idx - last_low_idx) <= MAX_GAP_DAYS
                and not np.isnan(last_low_rsi)):
            price1, rsi1 = last_low_price, last_low_rsi
            is_classic = (rsi2 > rsi1) and (price2 < price1)
            is_double_bottom = (rsi2 > rsi1) and (price2 >= price1) and (price2 <= price1 * (1 + DOUBLE_BOTTOM_MAX_PREMIUM_PCT))
            if is_classic or is_double_bottom:
                rsi_improve_abs = rsi2 - rsi1

                future_close = close[idx + 1: idx + 1 + FORWARD_DAYS]
                achieved = False
                if len(future_close) > 0:
                    below = future_close < price2
                    if below.any():
                        trough_rel = int(np.argmin(future_close))
                        trough_price = future_close[trough_rel]
                        if trough_price < price2:
                            after_trough = future_close[trough_rel + 1:]
                            if len(after_trough) > 0:
                                rebound_pct = (after_trough.max() - trough_price) / trough_price
                                if rebound_pct >= REBOUND_CONFIRM_PCT:
                                    achieved = True

                records.append({
                    "ticker": ticker, "low_idx": idx, "cap": cap,
                    "type": "classic" if is_classic else "double_bottom",
                    "rsi_improve_abs": rsi_improve_abs,
                    "achieved": achieved,
                })
        last_low_idx, last_low_price, last_low_rsi = idx, price2, rsi2

    return records


def bucket_abs(v):
    if v < 5: return "0~5p"
    if v < 10: return "5~10p"
    if v < 15: return "10~15p"
    if v < 20: return "15~20p"
    return "20p+"


def print_bucket_table(df, order):
    summary = df.groupby("bucket_abs").agg(
        표본수=("achieved", "count"),
        신저점달성률=("achieved", "mean"),
    )
    summary["신저점달성률"] = (summary["신저점달성률"] * 100).round(1)
    summary = summary.reindex(order).dropna(how="all")
    print(summary.to_string())


def main():
    print(f"저점 탐지 방식: {MODE} ({'저점 확정 필터 적용' if MODE=='confirmed' else '기존 실시간 방식'})")
    all_records = []
    for t in LARGECAP_TICKERS:
        try:
            recs = analyze_ticker(t, "large", MODE)
            all_records.extend(recs)
            n_classic = sum(1 for r in recs if r["type"] == "classic")
            print(f"{t}(대형): 정석 {n_classic}건 / 이중바닥 {len(recs)-n_classic}건")
        except Exception as e:
            print(f"{t}: 실패 ({e})")
    for t in SMALLCAP_TICKERS:
        try:
            recs = analyze_ticker(t, "small", MODE)
            all_records.extend(recs)
            n_classic = sum(1 for r in recs if r["type"] == "classic")
            print(f"{t}(소형): 정석 {n_classic}건 / 이중바닥 {len(recs)-n_classic}건")
        except Exception as e:
            print(f"{t}: 실패 ({e})")

    if not all_records:
        print("표본 없음")
        return

    df = pd.DataFrame(all_records)
    df["bucket_abs"] = df["rsi_improve_abs"].apply(bucket_abs)
    order = ["0~5p", "5~10p", "10~15p", "15~20p", "20p+"]

    for cap_label, cap_key in [("대형주", "large"), ("소형주", "small")]:
        df_cap = df[df["cap"] == cap_key]
        for type_label, type_key in [("정석", "classic"), ("이중바닥", "double_bottom")]:
            df_sub = df_cap[df_cap["type"] == type_key]
            print(f"\n=== [{cap_label} · {type_label}] RSI 개선폭(포인트)별 신저점 달성률 (FORWARD_DAYS={FORWARD_DAYS}) ===")
            if len(df_sub) == 0:
                print("표본 없음")
                continue
            print_bucket_table(df_sub, order)
            print(f"전체 표본: {len(df_sub)}건 / 전체 신저점달성률: {df_sub['achieved'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
