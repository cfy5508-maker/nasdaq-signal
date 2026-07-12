"""
smci_realtime_divergence_scan.py
사용법: python smci_realtime_divergence_scan.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10

동작: 미래 데이터로 저점을 "확정"하지 않는다. 대신 매일(i), 그날까지의
      과거 데이터만 보고 "지금까지 최근 ORDER일보다 낮은 새로운 저점을
      오늘 막 찍었다"고 판단되는 모든 날을 저점 후보로 표시한다.
      이렇게 실시간으로 잡을 수 있는 저점 후보들 사이에서 불리쉬
      다이버전스가 성립하는 모든 구간을 표시하고 승률을 확인한다.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

TICKER = "SMCI"
FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ORDER = 5
MAX_GAP = 60
MIN_GAP = 5  # 5일 미만은 진짜 다이버전스로 안 봄(너무 짧은 하락 중간 지점일 가능성)


def find_realtime_lows(c, order=5):
    positions = []
    values = c.values
    n = len(values)
    for i in range(order, n):
        past_window = values[i-order:i]
        if values[i] < past_window.min():
            positions.append(i)
    return positions


def main():
    hist = yf.Ticker(TICKER).history(period="2y")
    c = hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    n = len(hist)

    low_positions = find_realtime_lows(c, order=ORDER)
    print(f"=== {TICKER}: 실시간(확정대기 없음) 저점 후보 {len(low_positions)}개 ===\n")

    print(f"[연속 저점후보 쌍에서 다이버전스 판정 - 전부 표시]")
    rows = []
    div_count = 0
    for idx in range(len(low_positions) - 1):
        pos1 = low_positions[idx]
        pos2 = low_positions[idx + 1]
        gap = pos2 - pos1
        if gap > MAX_GAP or gap < MIN_GAP:
            continue

        price1, price2 = float(c.iloc[pos1]), float(c.iloc[pos2])
        rsi1, rsi2 = float(rsi.iloc[pos1]), float(rsi.iloc[pos2])
        if pd.isna(rsi1) or pd.isna(rsi2):
            continue

        is_divergence = bool(price2 < price1 and rsi2 > rsi1)
        date1, date2 = str(hist.index[pos1].date()), str(hist.index[pos2].date())
        marker = "다이버전스 O" if is_divergence else "다이버전스 X"

        if is_divergence:
            div_count += 1
            # Z-score 계산: pos2 시점 기준, 그 이전 1년 RSI 평균/표준편차 대비
            rsi_window = rsi.iloc[max(0, pos2-252):pos2+1].dropna()
            zscore = None
            if len(rsi_window) >= 60:
                rsi_mean = float(rsi_window.mean())
                rsi_std = float(rsi_window.std())
                if rsi_std > 0:
                    zscore = (rsi2 - rsi_mean) / rsi_std

            fwd_str = ""
            if pos2 + FORWARD_DAYS < n:
                entry = float(c.iloc[pos2])
                fwd_return = float(c.iloc[pos2 + FORWARD_DAYS]) / entry - 1
                win = fwd_return > 0
                rows.append({"date": date2, "fwd_return": fwd_return, "win": win, "zscore": zscore})
                z_str = f"{zscore:.2f}" if zscore is not None else "N/A"
                fwd_str = f" | Z={z_str} | {FORWARD_DAYS}일뒤 {fwd_return*100:+.2f}% | {'승리' if win else '패배'}"
            print(f"  {date1}(가{price1:.1f},R{rsi1:.1f}) -> {date2}(가{price2:.1f},R{rsi2:.1f}) | 간격{gap}일 | {marker}{fwd_str}")

    print(f"\n=== 실시간 방식 다이버전스 총 {div_count}건 (수익률 계산 가능 {len(rows)}건) ===")
    if rows:
        wins = sum(1 for r in rows if r["win"])
        print(f"전체 승률: {wins}/{len(rows)} = {wins/len(rows)*100:.1f}%\n")

        # Z-score 구간별 승률
        z_rows = [r for r in rows if r["zscore"] is not None]
        z_rows.sort(key=lambda r: r["zscore"])
        print("[Z-score 낮은 순 정렬]")
        for r in z_rows:
            print(f"  {r['date']}: Z={r['zscore']:.2f} | {r['fwd_return']*100:+.2f}% | {'승리' if r['win'] else '패배'}")

        if len(z_rows) >= 3:
            n_third = len(z_rows) // 3
            lower = z_rows[:n_third]
            upper = z_rows[-n_third:]
            w_low = sum(1 for r in lower if r["win"])
            w_up = sum(1 for r in upper if r["win"])
            print(f"\nZ-score 낮은 1/3(극단): {w_low}/{len(lower)} = {w_low/len(lower)*100:.1f}%")
            print(f"Z-score 높은 1/3(덜 극단): {w_up}/{len(upper)} = {w_up/len(upper)*100:.1f}%")
    else:
        print("표본 없음")


if __name__ == "__main__":
    main()
