"""
nxt_ohlc_check.py
사용법: python nxt_ohlc_check.py

동작: NXT의 최근 10거래일 OHLC(시가/고가/저가/종가)를 원본 그대로,
      그리고 수정종가(auto_adjust 여부)까지 비교해서 정확한 수치를 확인한다.
"""
import yfinance as yf

def main():
    print("=== NXT OHLC 확인 (auto_adjust=True, 기본값) ===")
    hist_adj = yf.Ticker("NXT").history(period="10d")
    print(hist_adj[["Open", "High", "Low", "Close", "Volume"]].to_string())

    print("\n=== NXT OHLC 확인 (auto_adjust=False, 원본 그대로) ===")
    hist_raw = yf.Ticker("NXT").history(period="10d", auto_adjust=False)
    print(hist_raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].to_string())

    print("\n=== 7/8 캔들 상세 계산 (auto_adjust=True 기준) ===")
    row = hist_adj.iloc[-1]
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    print(f"날짜: {hist_adj.index[-1].date()}")
    print(f"시가: {o}")
    print(f"고가: {h}")
    print(f"저가: {l}")
    print(f"종가: {c}")
    lower_wick = min(o, c) - l
    total_range = h - l
    print(f"아래꼬리: {lower_wick}")
    print(f"전체범위: {total_range}")
    print(f"아래꼬리비율: {lower_wick/total_range*100:.1f}%")


if __name__ == "__main__":
    main()
