#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
글로벌 주요 지수 수익률 표를 nasdaq_chart.html 안(차트 ③ 아래)에 주입.
- 각 지수의 -12 / -6 / -3 / -1 개월 누적 수익률(%) 계산
- nasdaq_chart.html 의 <!--GTABLE_START--> ~ <!--GTABLE_END--> 사이를 표로 교체
- 나스닥 스텝이 먼저 돌아 /tmp/nasdaq_chart.html 을 만들었으면 그걸 입력으로 사용
사용법:
    python fetch_indices.py            # 실데이터(yfinance)
    python fetch_indices.py --demo     # 네트워크 없이 샘플(이미지 수치)
"""
import os, sys, re
from datetime import datetime, timezone, timedelta
import pandas as pd

INDICES = [
    ("코스피",  "^KS11"),
    ("닛케이",  "^N225"),
    ("대만",    "^TWII"),
    ("유로",    "^STOXX50E"),
    ("CAC 40",  "^FCHI"),
    ("태국",    "^SET.BK"),
    ("다우",    "^DJI"),
    ("호주",    "^AXJO"),
    ("상하이",  "000001.SS"),
    ("DAX",     "^GDAXI"),
    ("인도",    "^BSESN"),
    ("S&P",     "^GSPC"),
    ("FTSE",    "^FTSE"),
    ("나스닥",  "^IXIC"),
    ("베트남",  "^VNINDEX"),   # Yahoo 미제공 가능 → N/A
    ("멕시코",  "^MXX"),
    ("브라질",  "^BVSP"),
    ("항셍",    "^HSI"),
    ("RTS",     "RTSI.ME"),    # 제재 이후 불안정 → N/A
    ("인니",    "^JKSE"),
]
PERIODS = [("-12개월", 12), ("-6개월", 6), ("-3개월", 3), ("-1개월", 1)]
KST = timezone(timedelta(hours=9))

# 차트 페이지 경로 (나스닥 스텝이 만든 /tmp 본을 우선 사용)
CHART_REPO = "nasdaq_chart.html"
CHART_TMP  = "/tmp/nasdaq_chart.html"
START_MARK = "<!--GTABLE_START-->"
END_MARK   = "<!--GTABLE_END-->"


def trailing_return(close, months):
    if close is None or len(close) == 0:
        return None
    close = close.dropna()
    if len(close) < 2:
        return None
    last = close.index[-1]
    target = last - pd.DateOffset(months=months)
    pos = close.index.get_indexer([target], method="nearest")[0]
    past, now = float(close.iloc[pos]), float(close.iloc[-1])
    return None if past == 0 else (now / past - 1.0) * 100.0


def fetch_real():
    import yfinance as yf
    out = {}
    for name, ticker in INDICES:
        try:
            df = yf.download(ticker, period="400d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                print(f"[WARN] no data: {name} ({ticker})", file=sys.stderr)
                out[name] = {p: None for p, _ in PERIODS}; continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            out[name] = {lab: trailing_return(close, m) for lab, m in PERIODS}
        except Exception as e:
            print(f"[ERROR] {name} ({ticker}): {e}", file=sys.stderr)
            out[name] = {p: None for p, _ in PERIODS}
    return out


def fetch_demo():
    raw = {
        "코스피":(201.6,122.0,57.7,16.1),"닛케이":(88.4,43.6,35.6,14.2),
        "대만":(116.6,69.6,42.3,12.9),"유로":(20.6,9.9,14.7,4.8),
        "CAC 40":(10.7,3.4,9.6,3.5),"태국":(47.4,24.0,9.8,2.3),
        "다우":(22.5,6.9,13.5,2.2),"호주":(3.7,1.3,4.6,1.8),
        "상하이":(23.9,6.3,5.2,1.2),"DAX":(7.7,3.5,12.3,1.0),
        "인도":(-4.0,-7.9,4.3,1.6),"S&P":(25.2,8.6,14.9,-0.0),
        "FTSE":(19.0,5.8,5.2,-0.3),"나스닥":(34.6,11.7,20.9,-0.7),
        "베트남":(None,None,None,None),"멕시코":(19.3,3.6,4.7,-1.8),
        "브라질":(24.3,7.7,-3.3,-3.3),"항셍":(1.7,-7.3,-5.4,-6.6),
        "RTS":(None,None,None,None),"인니":(-11.4,-29.3,-13.9,-0.7),
    }
    return {n:{lab:v[i] for i,(lab,_) in enumerate(PERIODS)} for n,v in raw.items()}


def cell(v):
    if v is None:
        return '<td style="padding:9px 7px;text-align:center;border-right:1px solid #2a2a35;border-top:1px solid #2a2a35;color:#52525b;">N/A</td>'
    r = round(v, 1)
    base = ("padding:9px 7px;text-align:center;border-right:1px solid #2a2a35;"
            "border-top:1px solid #2a2a35;font-variant-numeric:tabular-nums;")
    if r == 0:
        return f'<td style="{base}color:#9ca3af;">0.0</td>'
    if r > 0:
        return f'<td style="{base}color:#34d399;">+{r:.1f}</td>'
    return (f'<td style="{base}color:#fb7185;background:rgba(244,63,94,0.10);'
            f'font-weight:600;">{r:.1f}</td>')


def render_fragment(results, asof):
    names = [n for n, _ in INDICES]
    th = ('padding:9px 7px;text-align:center;color:#d1fae5;font-weight:700;'
          'background:#1f4d3a;border-right:1px solid #163a2c;white-space:nowrap;')
    head = "".join(f'<th style="{th}">{n}</th>' for n in names)
    rows = ""
    for lab, _ in PERIODS:
        cells = "".join(cell(results.get(n, {}).get(lab)) for n in names)
        rows += (f'<tr><th style="padding:9px 9px;text-align:center;color:#9ca3af;'
                 f'background:#202028;border-right:1px solid #2a2a35;border-top:1px solid #2a2a35;'
                 f'white-space:nowrap;font-weight:700;">{lab}</th>{cells}</tr>')
    return (
        START_MARK +
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;min-width:1050px;font-size:12px;color:#e2e2e8;">'
        f'<thead><tr><th style="{th}">기간</th>{head}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
        f'<div style="margin-top:7px;font-size:10px;color:#52525b;text-align:right;">'
        f'자료: Yahoo Finance · 기준 {asof} (KST)</div>' +
        END_MARK
    )


def main():
    demo = "--demo" in sys.argv
    results = fetch_demo() if demo else fetch_real()
    asof = datetime.now(KST).strftime("%Y-%m-%d")
    fragment = render_fragment(results, asof)

    src = CHART_TMP if os.path.exists(CHART_TMP) else CHART_REPO
    with open(src, "r", encoding="utf-8") as f:
        html = f.read()

    pat = re.compile(re.escape(START_MARK) + ".*?" + re.escape(END_MARK), re.DOTALL)
    if not pat.search(html):
        print("[ERROR] 차트 페이지에서 표 자리(GTABLE 마커)를 못 찾음", file=sys.stderr)
        sys.exit(1)
    html = pat.sub(lambda _: fragment, html)

    with open(CHART_TMP, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"글로벌 지수 표 주입 완료 ({'demo' if demo else 'live'}) · 입력={src} · 기준 {asof}")


if __name__ == "__main__":
    main()
