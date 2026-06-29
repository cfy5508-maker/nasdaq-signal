#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
글로벌 주요 지수 수익률 테이블 생성기
- 각 지수의 -12 / -6 / -3 / -1 개월 수익률(%)을 계산해 index.html 로 렌더링
- GitHub Actions(cron, 월 1회)에서 실행되도록 설계
사용법:
    python fetch_indices.py            # 실데이터(yfinance) 수집
    python fetch_indices.py --demo     # 네트워크 없이 샘플(이미지 수치)로 미리보기
"""
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

# (표시이름, yfinance 티커)  — 이미지 컬럼 순서와 동일
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
    ("베트남",  "^VNINDEX"),   # ⚠ Yahoo 미제공 가능성 — 아래 주석 참고
    ("멕시코",  "^MXX"),
    ("브라질",  "^BVSP"),
    ("항셍",    "^HSI"),
    ("RTS",     "RTSI.ME"),    # ⚠ 제재 이후 데이터 불안정 — 아래 주석 참고
    ("인니",    "^JKSE"),
]

PERIODS = [("-12개월", 12), ("-6개월", 6), ("-3개월", 3), ("-1개월", 1)]

KST = timezone(timedelta(hours=9))


def trailing_return(close: pd.Series, months: int):
    """N개월 전 거래일(가장 가까운 날) 대비 현재가 수익률(%)."""
    if close is None or len(close) == 0:
        return None
    close = close.dropna()
    if len(close) < 2:
        return None
    last_date = close.index[-1]
    target = last_date - pd.DateOffset(months=months)
    pos = close.index.get_indexer([target], method="nearest")[0]
    past = float(close.iloc[pos])
    now = float(close.iloc[-1])
    if past == 0:
        return None
    return (now / past - 1.0) * 100.0


def fetch_real():
    """yfinance로 각 지수의 종가 시계열을 받아 수익률 딕셔너리 반환."""
    import yfinance as yf
    results = {}
    for name, ticker in INDICES:
        try:
            df = yf.download(ticker, period="400d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                print(f"[WARN] no data: {name} ({ticker})", file=sys.stderr)
                results[name] = {p: None for p, _ in PERIODS}
                continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):   # 멀티컬럼 방어
                close = close.iloc[:, 0]
            results[name] = {label: trailing_return(close, m) for label, m in PERIODS}
        except Exception as e:
            print(f"[ERROR] {name} ({ticker}): {e}", file=sys.stderr)
            results[name] = {p: None for p, _ in PERIODS}
    return results


def fetch_demo():
    """네트워크 없이 미리보기용 — 업로드 이미지의 수치를 그대로 사용."""
    raw = {
        #            -12개월  -6개월  -3개월  -1개월
        "코스피":  (201.6, 122.0,  57.7,  16.1),
        "닛케이":  ( 88.4,  43.6,  35.6,  14.2),
        "대만":    (116.6,  69.6,  42.3,  12.9),
        "유로":    ( 20.6,   9.9,  14.7,   4.8),
        "CAC 40":  ( 10.7,   3.4,   9.6,   3.5),
        "태국":    ( 47.4,  24.0,   9.8,   2.3),
        "다우":    ( 22.5,   6.9,  13.5,   2.2),
        "호주":    (  3.7,   1.3,   4.6,   1.8),
        "상하이":  ( 23.9,   6.3,   5.2,   1.2),
        "DAX":     (  7.7,   3.5,  12.3,   1.0),
        "인도":    ( -4.0,  -7.9,   4.3,   1.6),
        "S&P":     ( 25.2,   8.6,  14.9,  -0.0),
        "FTSE":    ( 19.0,   5.8,   5.2,  -0.3),
        "나스닥":  ( 34.6,  11.7,  20.9,  -0.7),
        "베트남":  ( 37.7,   6.1,  12.8,  -1.0),
        "멕시코":  ( 19.3,   3.6,   4.7,  -1.8),
        "브라질":  ( 24.3,   7.7,  -3.3,  -3.3),
        "항셍":    (  1.7,  -7.3,  -5.4,  -6.6),
        "RTS":     (-10.3,  -8.2,  -7.9, -14.8),
        "인니":    (-11.4, -29.3, -13.9,  -0.7),
    }
    out = {}
    for name, vals in raw.items():
        out[name] = {label: vals[i] for i, (label, _) in enumerate(PERIODS)}
    return out


def fmt(v):
    if v is None:
        return "N/A", "na"
    sign = "+" if v >= 0 else ""          # 음수는 자체 부호 사용
    cls = "pos" if v >= 0 else "neg"
    return f"{sign}{v:.1f}", cls


def render_html(results, asof):
    names = [n for n, _ in INDICES]
    # 헤더
    head_cells = "".join(f"<th>{n}</th>" for n in names)
    # 본문
    rows = ""
    for label, _ in PERIODS:
        cells = ""
        for n in names:
            txt, cls = fmt(results.get(n, {}).get(label))
            cells += f'<td class="{cls}">{txt}</td>'
        rows += f'<tr><th class="period">{label}</th>{cells}</tr>'

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>글로벌 주요 지수 수익률</title>
<style>
  :root {{
    --green:#2c7a52; --green-dark:#225f40; --sage:#e9f1ea;
    --pos:#1b3a8f; --neg:#c0392b; --neg-bg:#fbe3e6; --line:#dfe6df;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:24px; background:#f6f8f6;
         font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",sans-serif;
         color:#1c1c1c; }}
  .wrap {{ max-width:1280px; margin:0 auto; }}
  h1 {{ font-size:18px; margin:0 0 4px; }}
  .sub {{ color:#667; font-size:12px; margin:0 0 16px; }}
  .tablebox {{ overflow-x:auto; border:1px solid var(--line);
              border-radius:10px; background:#fff;
              box-shadow:0 1px 4px rgba(0,0,0,.05); }}
  table {{ border-collapse:collapse; width:100%; min-width:1100px; font-size:14px; }}
  th, td {{ text-align:center; padding:12px 8px; white-space:nowrap; }}
  thead th {{ background:var(--green); color:#fff; font-weight:700;
              border-right:1px solid var(--green-dark); position:sticky; top:0; }}
  thead th:first-child {{ background:var(--green-dark); }}
  tbody th.period {{ background:var(--sage); color:var(--green-dark);
                     font-weight:700; border-right:1px solid var(--line); }}
  tbody td {{ border-right:1px solid var(--line); border-top:1px solid var(--line);
              font-variant-numeric:tabular-nums; }}
  td.pos {{ color:var(--pos); }}
  td.neg {{ color:var(--neg); background:var(--neg-bg); font-weight:600; }}
  td.na  {{ color:#aab; }}
  tbody tr:hover td {{ background:#f3f7f3; }}
  tbody tr:hover td.neg {{ background:#f7d8dc; }}
  .foot {{ margin-top:10px; font-size:11px; color:#889; text-align:right; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>글로벌 주요 지수 수익률 (%)</h1>
  <p class="sub">기간별 누적 수익률 · 매월 자동 업데이트</p>
  <div class="tablebox">
    <table>
      <thead><tr><th>기간</th>{head_cells}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <p class="foot">자료: Yahoo Finance · 기준일 {asof} (KST) · 자동 생성</p>
</div>
</body>
</html>"""


def main():
    demo = "--demo" in sys.argv
    results = fetch_demo() if demo else fetch_real()
    asof = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    html = render_html(results, asof)
    out = "global_index.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{out} 생성 완료 ({'demo' if demo else 'live'}) · 기준 {asof}")


if __name__ == "__main__":
    main()
