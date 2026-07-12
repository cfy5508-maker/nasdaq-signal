"""
rsi_volatility_check.py
사용법: python rsi_volatility_check.py

동작: 다이버전스 게이트 없이, 대형주/중소형주 각 종목의 최근 1년 RSI
      자체의 평균과 표준편차(변동폭)를 계산해서 두 그룹을 나란히 비교한다.
"""
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator

LARGECAP_TICKERS = [
    "NVDA", "INTC", "PYPL", "AAPL", "KO", "XOM", "META", "WMT", "BA", "AMD",
    "TSLA", "JPM", "DIS", "PFE", "CVX", "MCD", "NFLX", "GE", "CSCO", "UNH",
    "MSFT", "GOOGL", "AMZN", "JNJ", "PG", "V", "MA", "HD", "ORCL", "ABBV",
    "MRK", "ADBE", "CRM", "TXN", "LIN", "NEE", "LOW", "SBUX", "TMO", "ACN",
    "COST", "AVGO", "QCOM", "IBM", "GS", "BAC", "C", "T", "VZ", "UPS",
]
SMALLCAP_TICKERS = [
    "RKLB", "SMCI", "PLTR", "SOFI", "RIVN", "LCID", "CHPT", "U", "RBLX", "DKNG",
    "AFRM", "UPST", "COIN", "MARA", "RIOT", "CVNA", "ROOT", "OPEN", "CLSK", "IONQ",
    "FUBO", "PATH", "ASAN", "BBAI", "SOUN", "RUN", "ENVX", "JOBY", "ACHR", "LAZR",
    "QS", "FSLY", "PLUG", "NIO", "XPEV", "BYND", "WOLF", "GPRO", "DNA", "HOOD",
    "AI", "PTON", "SNAP", "LYFT", "DASH", "PINS", "ETSY", "BILL", "TTD", "ROKU",
    "W", "CHWY", "PENN", "ABNB", "DOCU", "ZM", "BROS", "CAVA", "SG", "APP",
    "SMR", "OKLO", "VKTX", "RXRX", "ARQQ", "DAVE", "NU", "CELH", "ELF", "FIGS",
]


def check(ticker):
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty or len(hist) < 260:
        return None
    c = hist["Close"]
    rsi = RSIIndicator(c, window=14).rsi().dropna()
    recent_1y = rsi.iloc[-252:] if len(rsi) >= 252 else rsi
    return float(recent_1y.mean()), float(recent_1y.std()), float(recent_1y.min()), float(recent_1y.max())


def run_group(tickers, label):
    means, stds = [], []
    print(f"=== {label} RSI 1년 평균/표준편차(변동폭) ===\n")
    for t in tickers:
        try:
            result = check(t)
        except Exception as e:
            print(f"  {t}: 에러 ({e})")
            continue
        if result is None:
            print(f"  {t}: 데이터 부족")
            continue
        mean, std, mn, mx = result
        means.append(mean); stds.append(std)
        print(f"  {t}: 평균 {mean:.1f} | 표준편차(변동폭) {std:.1f} | 범위 {mn:.1f}~{mx:.1f}")

    print(f"\n--- {label} 종합 ---")
    print(f"평균의 평균: {np.mean(means):.1f}")
    print(f"표준편차의 평균(=평균 변동폭): {np.mean(stds):.1f}")
    print(f"표준편차 최소~최대: {np.min(stds):.1f} ~ {np.max(stds):.1f}\n")
    return np.mean(means), np.mean(stds)


def main():
    large_mean, large_std = run_group(LARGECAP_TICKERS, "대형주")
    small_mean, small_std = run_group(SMALLCAP_TICKERS, "중소형주")

    print("=== 최종 비교 ===")
    print(f"대형주   : 평균RSI {large_mean:.1f} | 평균 변동폭(표준편차) {large_std:.1f}")
    print(f"중소형주 : 평균RSI {small_mean:.1f} | 평균 변동폭(표준편차) {small_std:.1f}")
    print(f"변동폭 차이: {small_std - large_std:+.1f} (중소형주가 이만큼 더/덜 흔들림)")


if __name__ == "__main__":
    main()
