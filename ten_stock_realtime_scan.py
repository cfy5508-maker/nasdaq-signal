"""
ten_stock_realtime_scan.py
사용법: python ten_stock_realtime_scan.py [FORWARD_DAYS]
기본값: FORWARD_DAYS=10
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

LARGECAP_NEW = ["ORCL", "ADBE", "TXN", "LOW", "GS"]
SMALLCAP_NEW = ["CHPT", "AFRM", "NIO", "DKNG", "PLUG"]
TICKERS = LARGECAP_NEW + SMALLCAP_NEW

FORWARD_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ORDER = 5
MAX_GAP = 60
MIN_GAP = 3


def find_realtime_lows(c, order=5):
    positions = []
    values = c.values
    n = len(values)
    for i in range(order, n):
        past_window = values[i-order:i]
        if values[i] < past_window.min():
            positions.append(i)
    return positions


def analyze_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return None
    c = hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi()
    n = len(hist)

    low_positions = find_realtime_lows(c, order=ORDER)

    rows = []
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
        if not is_divergence:
            continue

        rsi_window = rsi.iloc[max(0, pos2-252):pos2+1].dropna()
        zscore = None
        if len(rsi_window) >= 60:
            rsi_mean = float(rsi_window.mean())
            rsi_std = float(rsi_window.std())
            if rsi_std > 0:
                zscore = (rsi2 - rsi_mean) / rsi_std

        if pos2 + FORWARD_DAYS < n:
            entry = float(c.iloc[pos2])
            fwd_return = float(c.iloc[pos2 + FORWARD_DAYS]) / entry - 1
            date2 = str(hist.index[pos2].date())
            rows.append({"ticker": ticker, "date": date2, "fwd_return": fwd_return,
                         "win": fwd_return > 0, "zscore": zscore})
    return rows


def main():
    all_rows = []
    print(f"=== 10종목 실시간 다이버전스 검증 (MIN_GAP={MIN_GAP}일, {FORWARD_DAYS}일 뒤 기준) ===\n")
    for t in TICKERS:
        try:
            rows = analyze_ticker(t)
        except Exception as e:
            print(f"{t}: 에러 ({e})")
            continue
        if rows is None:
            print(f"{t}: 데이터 없음")
            continue
        wins = sum(1 for r in rows if r["win"])
        wr = wins / len(rows) * 100 if rows else None
        print(f"[{t}] 다이버전스 {len(rows)}건" + (f" | 승률 {wr:.1f}%" if wr is not None else " | 신호 없음"))
        for r in sorted(rows, key=lambda r: (r["zscore"] is None, r["zscore"])):
            z_str = f"{r['zscore']:.2f}" if r["zscore"] is not None else "N/A"
            print(f"    {r['date']}: Z={z_str} | {r['fwd_return']*100:+.2f}% | {'승리' if r['win'] else '패배'}")
        all_rows.extend(rows)

    print(f"\n=== 전체 종합 (10종목 합산) ===")
    if not all_rows:
        print("신호 없음")
        return
    total_wins = sum(1 for r in all_rows if r["win"])
    print(f"총 다이버전스 신호: {len(all_rows)}건")
    print(f"전체 승률: {total_wins}/{len(all_rows)} = {total_wins/len(all_rows)*100:.1f}%\n")

    z_rows = [r for r in all_rows if r["zscore"] is not None]
    z_rows.sort(key=lambda r: r["zscore"])
    if len(z_rows) >= 3:
        n_third = len(z_rows) // 3
        lower = z_rows[:n_third]
        middle = z_rows[n_third:2*n_third]
        upper = z_rows[2*n_third:]
        for label, group in [("Z-score 낮은 1/3(극단)", lower), ("중간 1/3", middle), ("Z-score 높은 1/3(덜 극단)", upper)]:
            w = sum(1 for r in group if r["win"])
            print(f"{label}: {w}/{len(group)} = {w/len(group)*100:.1f}%")


if __name__ == "__main__":
    main()
