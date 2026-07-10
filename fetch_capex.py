#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
아마존이 EDGAR에서 capex를 어떤 태그·어떤 기간 형식으로 올리는지 진단.
한 번 돌려서 로그 확인 후 fetch_capex.py를 고치기 위한 용도.
"""
import json, sys, time
import datetime as dt
import urllib.request

UA = {"User-Agent": "nasdaq-signal capex-diagnose cfy5508@example.com"}
AMZN_CIK = "0001018724"

# 아마존이 쓸 법한 capex 관련 태그 후보 (넓게)
TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    "PurchasesOfPropertyAndEquipment",
    "PaymentsForCapitalImprovements",
    "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
]


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    # 1) 아마존이 실제 보고하는 모든 us-gaap 컨셉 중 capex/property 관련 태그 찾기
    print("=== 아마존 companyfacts에서 capex 관련 태그 스캔 ===", file=sys.stderr)
    try:
        facts = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{AMZN_CIK}.json")
        gaap = facts.get("facts", {}).get("us-gaap", {})
        hits = [k for k in gaap if ("PropertyPlantAndEquipment" in k or "ProductiveAssets" in k
                                    or "Capital" in k) and "Payment" in k or "Purchase" in k]
        cand = [k for k in gaap if "Payment" in k or "Purchase" in k]
        prop = [k for k in cand if "Propert" in k or "Productive" in k or "Capital" in k or "Equipment" in k]
        print("  capex 후보 태그:", file=sys.stderr)
        for k in sorted(prop):
            print(f"    - {k}", file=sys.stderr)
    except Exception as e:
        print(f"  companyfacts 실패: {e}", file=sys.stderr)
        prop = TAGS

    # 2) 각 후보 태그의 최근 값 5개 + 기간 길이 확인
    for tag in sorted(set(list(prop) + TAGS)):
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{AMZN_CIK}/us-gaap/{tag}.json"
        try:
            data = get(url)
        except Exception:
            continue
        units = data.get("units", {}).get("USD", [])
        if not units:
            continue
        rows = []
        for u in units:
            s, e = u.get("start"), u.get("end")
            if not s or not e:
                continue
            days = (dt.date.fromisoformat(e) - dt.date.fromisoformat(s)).days
            rows.append((e, u["val"], days, u.get("form")))
        rows = sorted(set(rows), reverse=True)[:6]
        if rows:
            print(f"\n[{tag}] 최근 값:", file=sys.stderr)
            for e, v, days, form in rows:
                mark = " ← 단일분기?" if 80 <= days <= 100 else ""
                print(f"    end={e} ${v/1e9:.2f}B  기간={days}일 ({form}){mark}", file=sys.stderr)
        time.sleep(0.2)

    print("\n=== 진단 끝. 위에서 2026년 값이 있고 '단일분기'로 뜨는 태그를 쓰면 됨 ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
