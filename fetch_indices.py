#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
글로벌 주요 지수 수익률 표를 nasdaq_chart.html 안(차트 ③ 아래)에 주입.
- 각 지수의 0(MTD) / -1 / -2 / -3 / -6 / -12 개월 수익률(%) 계산
- 0M = 이번 달 1일 대비 현재(%) — 매일 갱신, 참고용
- -2M = 집중도 '주의' 추세 판정용
- nasdaq_chart.html 의 <!--GTABLE_START--> ~ <!--GTABLE_END--> 사이를 표로 교체
"""
import os, sys, re
from datetime import datetime, timezone, timedelta
import pandas as pd

INDICES = [
    ("인니","^JKSE"),("태국","^SET.BK"),("호주","^AXJO"),("브라질","^BVSP"),
    ("멕시코","^MXX"),("FTSE","^FTSE"),("인도","^BSESN"),("항셍","^HSI"),
    ("상하이","000001.SS"),("다우","^DJI"),("DAX","^GDAXI"),("CAC 40","^FCHI"),
    ("닛케이","^N225"),("S&P","^GSPC"),("유로","^STOXX50E"),("코스피","^KS11"),
    ("나스닥","^IXIC"),("대만","^TWII"),
]

CORE = {"대만","나스닥","코스피","유로","S&P","닛케이"}
MID  = {"CAC 40","DAX","다우","상하이","항셍"}

# (라벨, 개월수). 0M은 개월수 0 = MTD 특수처리. 표시 순서 = 이 순서.
PERIODS = [("이번달", 0), ("-1개월", 1), ("-2개월", 2),
           ("-3개월", 3), ("-6개월", 6), ("-12개월", 12)]
KST = timezone(timedelta(hours=9))

CHART_REPO = "nasdaq_chart.html"
CHART_TMP  = "/tmp/nasdaq_chart.html"
START_MARK = "<!--GTABLE_START-->"
END_MARK   = "<!--GTABLE_END-->"


def trailing_return(close, months):
    """N개월 전 대비 누적 수익률(%). months=0이면 MTD(이번 달 1일 대비)."""
    if close is None or len(close) == 0:
        return None
    close = close.dropna()
    if len(close) < 2:
        return None
    now = float(close.iloc[-1])
    if months == 0:
        # MTD: 이번 달 1일(그 이전 마지막 거래일) 종가 대비
        last_date = close.index[-1]
        month_start = last_date.replace(day=1)
        prior = close[close.index < month_start]
        if len(prior) == 0:
            return None  # 이번 달 첫 거래일이면 아직 MTD 없음
        base = float(prior.iloc[-1])  # 전월 말일 종가 = 이번 달 시작 기준점
        return None if base == 0 else (now / base - 1.0) * 100.0
    target = close.index[-1] - pd.DateOffset(months=months)
    pos = close.index.get_indexer([target], method="nearest")[0]
    past = float(close.iloc[pos])
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
                # ↓ 디버그: 마지막 5거래일 날짜·종가 출력
            print(f"[DEBUG] {name}: " + " | ".join(
                f"{str(d)[:10]}={float(v):.1f}" for d,v in list(close.dropna().items())[-5:]),
                file=sys.stderr)
            out[name] = {lab: trailing_return(close, m) for lab, m in PERIODS}
        except Exception as e:
            print(f"[ERROR] {name} ({ticker}): {e}", file=sys.stderr)
            out[name] = {p: None for p, _ in PERIODS}
    return out


def fetch_demo():
    # (이번달, -1, -2, -3, -6, -12) 순
    raw = {
        "인니":(0.3,0.6,-2.1,-20.4,-33.5,-15.2),"태국":(0.4,1.0,3.5,10.8,25.9,42.9),
        "호주":(0.5,1.7,0.2,-1.8,1.0,2.5),"브라질":(0.7,1.7,-1.1,-11.5,5.7,25.6),
        "멕시코":(0.3,1.1,-0.4,-6.0,0.1,16.6),"FTSE":(0.8,2.4,1.1,-1.2,3.4,18.1),
        "인도":(1.2,3.8,2.6,0.1,-8.2,-8.1),"항셍":(-0.5,-0.6,-2.7,-6.3,-7.5,1.0),
        "상하이":(0.6,1.6,1.2,1.8,-1.5,15.6),"다우":(0.9,3.2,4.7,8.9,6.0,18.1),
        "DAX":(0.7,2.8,-1.9,5.5,-0.6,2.3),"CAC 40":(0.5,1.5,2.2,1.0,-0.4,5.7),
        "닛케이":(2.0,8.0,4.7,21.7,33.4,74.8),"S&P":(0.6,2.1,-1.1,10.5,8.3,20.4),
        "유로":(1.0,3.9,3.0,6.6,4.8,15.4),"코스피":(-0.4,-1.3,-2.8,30.3,66.4,139.7),
        "나스닥":(0.6,2.1,-1.0,14.8,10.7,27.1),"대만":(0.5,1.5,0.6,30.1,49.7,101.3),
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


def header_style(name):
    if name in CORE:
        return "background:#1f4d3a;color:#d1fae5;border-right:1px solid #163a2c;"
    if name in MID:
        return "background:#1f3a4d;color:#bae6fd;border-right:1px solid #163040;"
    return "background:#26262e;color:#8b8b9a;border-right:1px solid #1f1f27;"


def _row_label(lab):
    # '이번달' 행은 참고용이라 살짝 다른 톤 + (MTD) 표기
    if lab == "이번달":
        return ('padding:9px 9px;text-align:center;color:#a78bfa;'
                'background:#211f2e;border-right:1px solid #2a2a35;border-top:1px solid #2a2a35;'
                'white-space:nowrap;font-weight:700;', "이번달<br><span style='font-size:8px;color:#6b6b7a;'>MTD·매일</span>")
    return ('padding:9px 9px;text-align:center;color:#9ca3af;'
            'background:#202028;border-right:1px solid #2a2a35;border-top:1px solid #2a2a35;'
            'white-space:nowrap;font-weight:700;', lab)


def render_fragment(results, asof):
    names = [n for n, _ in INDICES]
    thbase = "padding:9px 7px;text-align:center;font-weight:700;white-space:nowrap;"
    corner = thbase + "background:#1f4d3a;color:#d1fae5;border-right:1px solid #163a2c;"
    head = "".join(f'<th style="{thbase}{header_style(n)}">{n}</th>' for n in names)
    rows = ""
    for lab, _ in PERIODS:
        cells = "".join(cell(results.get(n, {}).get(lab)) for n in names)
        style, disp = _row_label(lab)
        rows += f'<tr><th style="{style}">{disp}</th>{cells}</tr>'
    legend = (
        '<div style="margin-top:8px;font-size:10px;color:#8b8b9a;display:flex;gap:14px;flex-wrap:wrap;align-items:center;">'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#1f4d3a;vertical-align:middle;margin-right:4px;"></span>핵심 (반도체 비중 높음)</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#1f3a4d;vertical-align:middle;margin-right:4px;"></span>중간</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#26262e;border:1px solid #3a3a45;vertical-align:middle;margin-right:4px;"></span>무관</span>'
        '<span style="color:#6b6b7a;">· 이번달=이달 1일 대비(참고) · 집중도는 -1~-12로 계산</span>'
        '</div>'
    )
    return (
        START_MARK +
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;min-width:980px;font-size:12px;color:#e2e2e8;">'
        f'<thead><tr><th style="{corner}">기간</th>{head}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
        + legend +
        f'<div style="margin-top:4px;font-size:10px;color:#52525b;text-align:right;">'
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
