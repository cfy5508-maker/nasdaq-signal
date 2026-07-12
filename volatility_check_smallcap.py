"""
volatility_check_smallcap.py
사용법: python volatility_check_smallcap.py

동작: 중소형주 20개의 최근 2년 변동성(연율화, %)을 실측하고,
      이미 확인된 대형주 고변동성 그룹(INTC, AMD, TSLA, ORCL, NVDA 등)과
      나란히 비교해서 "중소형주가 실제로 고변동성인지" 검증한다.
"""
import numpy as np
import pandas as pd
import yfinance as yf

SMALLCAP_TICKERS = ["RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
                     "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ"]

LARGECAP_HIGHVOL = ["INTC", "AMD", "TSLA", "ORCL", "NVDA", "UNH", "TXN", "PYPL", "META", "BA",
                     "CRM", "SBUX", "ACN", "ADBE", "NFLX", "GE", "AMZN", "GOOGL", "TMO", "AAPL"]

LARGECAP_LOWVOL = ["CSCO", "MRK", "DIS", "ABBV", "MSFT", "NEE", "JPM", "LOW", "XOM", "WMT",
                    "PFE", "CVX", "HD", "V", "MA", "PG", "JNJ", "MCD", "LIN", "KO"]


def compute_volatility(ticker):
    hist = yf.Ticker(ticker).history(period="2y")["Close"]
    if hist.empty or len(hist) < 100:
        return None
    daily_ret = hist.pct_change().dropna()
    annualized_vol = daily_ret.std() * (252 ** 0.5) * 100
    return annualized_vol


def summarize(name, tickers):
    vols = []
    print(f"\n=== {name} ===")
    for t in tickers:
        v = compute_volatility(t)
        if v is not None:
            vols.append(v)
            print(f"  {t}: {v:.1f}%")
    if vols:
        arr = np.array(vols)
        print(f"  --- 평균 {arr.mean():.1f}% | 중앙값 {np.median(arr):.1f}% | 최소 {arr.min():.1f}% | 최대 {arr.max():.1f}% ---")
    return vols


def main():
    small_vols = summarize("중소형주 20개", SMALLCAP_TICKERS)
    large_high_vols = summarize("대형주 고변동성 20개 (지난번 결과)", LARGECAP_HIGHVOL)
    large_low_vols = summarize("대형주 저변동성 20개 (지난번 결과, 참고)", LARGECAP_LOWVOL)

    print("\n=== 종합 비교 ===")
    if small_vols:
        print(f"중소형주 평균 변동성: {np.mean(small_vols):.1f}%")
    if large_high_vols:
        print(f"대형주 고변동성 그룹 평균: {np.mean(large_high_vols):.1f}%")
    if large_low_vols:
        print(f"대형주 저변동성 그룹 평균: {np.mean(large_low_vols):.1f}%")

    if small_vols and large_high_vols:
        diff = np.mean(small_vols) - np.mean(large_high_vols)
        print(f"\n중소형주 - 대형주고변동성 = {diff:+.1f}%p")
        if diff > 10:
            print("-> 중소형주가 대형주 고변동성 그룹보다도 확실히 변동성이 큼.")
            print("   그런데도 4단계 효과가 약하다면, '변동성만으로는 설명 안 됨' 확정.")
        elif diff > -5:
            print("-> 중소형주와 대형주 고변동성 그룹의 변동성 수준이 비슷함.")
            print("   변동성이 비슷한데 4단계 효과가 다르다면 시가총액 자체가 핵심 변수.")
        else:
            print("-> 중소형주가 오히려 대형주 고변동성 그룹보다 변동성이 낮음.")
            print("   이 경우 중소형주의 4단계 약세는 변동성 부족 때문일 수도 있음 (재해석 필요).")


if __name__ == "__main__":
    main()
