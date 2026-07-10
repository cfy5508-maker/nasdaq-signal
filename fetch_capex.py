#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빅테크 4사(MS·구글·아마존·메타) 분기 capex 합산 + 전분기 대비(QoQ) 변동률을
SEC EDGAR 공식 API에서 받아 배지 JSON으로 저장.

- SEC EDGAR companyconcept API (무료·공개, User-Agent 필수)
- capex 태그가 회사마다 달라서 여러 후보를 자동 시도(tag auto-discovery)
- 분기(10-Q/10-K) 값만 골라 최근 2개 분기 합산 비교 → QoQ %
- data/capex_qoq.json 에 기록

주의: 이 스크립트는 GitHub Actions 러너에서 실행(SEC 접속 가능).
      로컬 샌드박스에선 data.sec.gov 차단으로 실행 안 될 수 있음.
"""
from __future__ import annotations
import json, sys, time
import datetime as dt
from pathlib import Path
import urllib.request

DATA_DIR = Path("data")
OUT = DATA_DIR / "capex_qoq.json"

# SEC는 식별 가능한 User-Agent를 요구(형식: 이름/용도 + 이메일)
UA = {"User-Agent": "nasdaq-signal capex-tracker cfy5508@example.com"}

# 4사 CIK(10자리 zero-pad)
CIKS = {
    "Microsoft": "0000789019",
    "Alphabet":  "0001652044",
    "Amazon":    "0001018724",
    "Meta":      "0001326801",
}

# 회사별 capex 태그(우선순위 순). 아마존은 2017년 이후 ProductiveAssets로 태깅.
COMPANY_TAGS = {
    "Microsoft": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "Alphabet":  ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "Amazon":    ["PaymentsToAcquireProductiveAssets",
                  "PaymentsToAcquirePropertyPlantAndEquipment"],
    "Meta":      ["PaymentsToAcquirePropertyPlantAndEquipment"],
}


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_quarterly_capex(name, cik):
    """(태그, [(end_date, val_quarterly), ...]) 반환. 실패 시 (None, [])."""
    for tag in COMPANY_TAGS.get(name, ["PaymentsToAcquirePropertyPlantAndEquipment"]):
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
        try:
            data = _get(url)
        except Exception as e:
            continue
        units = data.get("units", {}).get("USD", [])
        if not units:
            continue

        # 분기 값 추출: 기간 길이가 ~90일(80~100일)인 것 = 단일 분기
        rows = []
        for u in units:
            s, e = u.get("start"), u.get("end")
            if not s or not e:
                continue
            d0 = dt.date.fromisoformat(s)
            d1 = dt.date.fromisoformat(e)
            days = (d1 - d0).days
            if 80 <= days <= 100:  # 단일 분기만 (누적 9개월/연간 제외)
                rows.append((e, float(u["val"]), days))

        # end 기준 중복 제거(최신 form 우선), 정렬
        best = {}
        for e, v, days in rows:
            if e not in best:
                best[e] = v
        series = sorted(best.items())  # [(end, val), ...] 오름차순
        if len(series) >= 2:
            print(f"[OK] {name}: 태그 '{tag}' — 분기 {len(series)}개, "
                  f"최근 {series[-1][0]} ${series[-1][1]/1e9:.2f}B", file=sys.stderr)
            return tag, series
        else:
            print(f"[..] {name}: 태그 '{tag}'는 분기값 부족({len(series)})", file=sys.stderr)
        time.sleep(0.2)  # SEC 예의상 rate limit
    print(f"[WARN] {name}: 맞는 capex 태그 못 찾음", file=sys.stderr)
    return None, []


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    per_company = {}   # name -> {tag, series}
    for name, cik in CIKS.items():
        tag, series = fetch_quarterly_capex(name, cik)
        per_company[name] = {"tag": tag, "series": series}
        time.sleep(0.3)

    # 4사 공통으로 값이 있는 '최근 2개 분기'를 찾아 합산
    # 각 사의 분기 end 날짜가 조금씩 다르므로, 분기 라벨(YYYY-Qn)로 묶음
    def qlabel(end_iso):
        d = dt.date.fromisoformat(end_iso)
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}"

    # 분기라벨 -> {회사: val}
    bucket = {}
    for name, info in per_company.items():
        for end, val in info["series"]:
            bucket.setdefault(qlabel(end), {})[name] = val

    # 4사 모두 값이 있는 분기만 (합산 비교 공정성)
    full = sorted([q for q, m in bucket.items() if len(m) == len(CIKS)])
    sums = {q: sum(bucket[q].values()) for q in full}

    def yoy(q):
        """q(YYYY-Qn)의 전년동기 대비 증가율(%). 전년 동분기 없으면 None."""
        y, qn = q.split("-")
        prev_year_q = f"{int(y)-1}-{qn}"
        if prev_year_q in sums and sums[prev_year_q]:
            return (sums[q] / sums[prev_year_q] - 1.0) * 100.0
        return None

    # YoY가 계산되는 분기들 (최근순)
    yoy_quarters = [q for q in full if yoy(q) is not None]
    if len(yoy_quarters) < 1:
        print(f"[ERROR] YoY 계산 가능한 분기 없음 (full={full})", file=sys.stderr)
        result = {
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "status": "insufficient",
            "quarters_available": full,
            "per_company_tags": {n: per_company[n]["tag"] for n in CIKS},
        }
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    last_q = yoy_quarters[-1]
    last_yoy = yoy(last_q)
    # 직전 분기(가능하면)의 YoY → 가속/둔화 판정용
    prev_q = yoy_quarters[-2] if len(yoy_quarters) >= 2 else None
    prev_yoy = yoy(prev_q) if prev_q else None
    accel = (last_yoy - prev_yoy) if (prev_yoy is not None) else None  # YoY의 변화(2차)

    # 국면 판정: capex 증가율(YoY)이 가속이냐 둔화냐
    if accel is None:
        phase, color = "기준", "#7f8c8d"
    elif accel >= 3:
        phase, color = "가속", "#c0392b"     # 붐 심화(과열 배경 강화)
    elif accel > -3:
        phase, color = "유지", "#e67e22"     # 증가율 횡보
    elif last_yoy > 0:
        phase, color = "둔화", "#f1c40f"     # 증가율 꺾임 ← 버블 후기 경고
    else:
        phase, color = "감소", "#27ae60"     # capex 자체 역성장

    result = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
        "last_quarter": last_q,
        "last_sum_usd_b": round(sums[last_q] / 1e9, 1),
        "yoy_pct": round(last_yoy, 1),
        "prev_quarter": prev_q,
        "prev_yoy_pct": round(prev_yoy, 1) if prev_yoy is not None else None,
        "yoy_change_pp": round(accel, 1) if accel is not None else None,  # 증가율 변화(%p)
        "phase": phase,                # 가속 / 유지 / 둔화 / 감소
        "color": color,
        "per_company_last": {n: round(bucket[last_q].get(n, 0)/1e9, 1) for n in CIKS},
        "per_company_tags": {n: per_company[n]["tag"] for n in CIKS},
        "label": (f"빅테크 capex {last_q} ${sums[last_q]/1e9:.0f}B · "
                  f"YoY {'+' if last_yoy>=0 else ''}{last_yoy:.0f}% ({phase})"),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    acc_txt = f" | 직전 YoY {prev_yoy:+.1f}% → {accel:+.1f}%p {phase}" if accel is not None else ""
    print(f"[ok] {last_q} ${sums[last_q]/1e9:.1f}B · YoY {last_yoy:+.1f}%{acc_txt}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
