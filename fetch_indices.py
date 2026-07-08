#!/usr/bin/env python3
"""
FINRA Customer Margin Balance -> 투자자 현금여력(credit balance) 배지 생성기.

매월 FINRA Margin Statistics를 받아
  여유현금비율 = (Free Credit: Cash + Margin) / Debit(margin debt)
를 계산하고, 2010-02 이후 히스토리 대비 백분위(percentile)를 산출해
badge JSON을 씁니다.

- 1차 소스: margin-statistics.xlsx (1997~ 전체 히스토리, 고정 URL)
- 폴백    : 페이지 HTML 표 (최근 13개월) -> 캐시된 히스토리에 병합
- 캐시    : data/finra_history.csv 에 월별 시계열을 누적 보관
            (xlsx가 일시적으로 막혀도 백분위 계산이 유지됨)

FINRA는 공식 API/피드가 없어 스크래핑이 유일한 경로입니다.
포맷이 바뀌면 깨질 수 있어 방어적으로 파싱합니다.
"""
from __future__ import annotations

import io
import json
import sys
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

XLSX_URL = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
PAGE_URL = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"

# 리포 루트 기준 경로 (GitHub Actions 체크아웃 루트에서 실행)
DATA_DIR = Path("data")
HISTORY_CSV = DATA_DIR / "finra_history.csv"
BADGE_JSON = DATA_DIR / "finra_credit_balance.json"

# 전 회원사 일관 기준(Feb 2010~). 이 이전은 free credit 구조가 달라 백분위에서 제외.
BASELINE_START = pd.Timestamp("2010-02-01")

def _headers(referer=None):
    h = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                  "image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }
    if referer:
        h["Referer"] = referer
        h["Sec-Fetch-Site"] = "same-origin"
    return h


def _http_get(url, referer=None) -> bytes:
    """FINRA는 Akamai 봇 매니저가 TLS 지문(JA3)으로 차단하므로,
    curl_cffi로 실제 크롬 지문을 임퍼소네이트한다. 미설치/실패 시 requests 폴백.
    """
    import time
    headers = _headers(referer)

    # 1) curl_cffi (크롬 TLS/HTTP2 임퍼소네이트) — Akamai 403 우회
    try:
        from curl_cffi import requests as cffi
        last = None
        for attempt in range(3):
            try:
                r = cffi.get(url, headers=headers, impersonate="chrome", timeout=60)
                if r.status_code == 200:
                    return r.content
                last = f"HTTP {r.status_code}"
            except Exception as e:  # noqa: BLE001
                last = repr(e)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"curl_cffi 3회 실패: {last}")
    except ImportError:
        print("[warn] curl_cffi 미설치 — requests로 폴백", file=sys.stderr)

    # 2) plain requests 폴백 (Akamai 없는 로컬/사내망 등)
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.content

CANON = ["month", "debit", "fc_cash", "fc_margin"]


# --------------------------------------------------------------------------- #
# 파싱 유틸
# --------------------------------------------------------------------------- #
def _norm(x) -> str:
    return str(x).strip().lower().replace("\xa0", " ")


def _to_num(x):
    """'1,415,557' / 1415557 / '' -> float | None"""
    if pd.isna(x):
        return None
    s = str(x).replace(",", "").replace("$", "").strip()
    if s in ("", "-", "--", "n/a", "na"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


_MONTH_FORMATS = ("%b-%y", "%b-%Y", "%B-%y", "%B-%Y",
                  "%b %y", "%b %Y", "%B %y", "%B %Y",
                  "%Y-%m", "%m/%Y", "%Y-%m-%d")


def _to_month(x):
    """'May-26' / 'May 2026' / '2026-05' / Timestamp -> month-start Timestamp | None.

    FINRA는 'Mon-YY'(예: May-26)를 쓰는데 pandas 기본 파서는 '26'을 일(day)로
    오인하므로 명시 포맷을 먼저 시도한다. %y는 26->2026, 97->1997로 매핑된다.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(x).normalize().replace(day=1)
    s = str(x).strip()
    if not s:
        return None
    for fmt in _MONTH_FORMATS:
        ts = pd.to_datetime(s, format=fmt, errors="coerce")
        if not pd.isna(ts):
            return ts.normalize().replace(day=1)
    ts = pd.to_datetime(s, errors="coerce")  # 최후의 일반 파서
    return None if pd.isna(ts) else ts.normalize().replace(day=1)


def _map_columns(cols) -> dict:
    """헤더 문자열이 조금 달라져도 필요한 컬럼을 fuzzy 매칭."""
    out = {}
    for c in cols:
        n = _norm(c)
        if "debit" in n:
            out["debit"] = c
        elif "free credit" in n and "cash" in n:
            out["fc_cash"] = c
        elif "free credit" in n and "margin" in n:
            out["fc_margin"] = c
    return out


def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """임의 표 -> [month, debit, fc_cash, fc_margin] 정규화."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    m = _map_columns(df.columns)
    if not {"debit", "fc_cash", "fc_margin"} <= set(m):
        raise ValueError(f"필요 컬럼 매칭 실패: found={list(m)} in {list(df.columns)}")

    # 날짜 컬럼 = 매칭 안 된 컬럼 중 파싱 성공률이 가장 높은 것
    used = set(m.values())
    date_col, best = None, -1.0
    for c in df.columns:
        if c in used:
            continue
        ok = df[c].map(_to_month).notna().mean()
        if ok > best:
            date_col, best = c, ok
    if date_col is None or best < 0.5:
        raise ValueError("날짜 컬럼을 찾지 못함")

    out = pd.DataFrame(
        {
            "month": df[date_col].map(_to_month),
            "debit": df[m["debit"]].map(_to_num),
            "fc_cash": df[m["fc_cash"]].map(_to_num),
            "fc_margin": df[m["fc_margin"]].map(_to_num),
        }
    )
    out = out.dropna(subset=CANON).reset_index(drop=True)
    out["month"] = pd.to_datetime(out["month"])
    return out


def _find_table_in_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """header=None으로 읽은 시트에서 'Debit' 헤더 행을 찾아 표로 복원."""
    for i in range(min(25, len(raw))):
        if any("debit" in _norm(v) for v in raw.iloc[i].tolist()):
            body = raw.iloc[i + 1 :].copy()
            body.columns = raw.iloc[i].tolist()
            return body
    raise ValueError("시트에서 'Debit' 헤더 행을 찾지 못함")


# --------------------------------------------------------------------------- #
# 소스 로더
# --------------------------------------------------------------------------- #
def load_from_xlsx(content: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(content))
    last_err = None
    for sheet in xls.sheet_names:
        try:
            raw = pd.read_excel(xls, sheet_name=sheet, header=None)
            return _normalize_table(_find_table_in_sheet(raw))
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"xlsx 파싱 실패: {last_err}")


def load_from_html(html: str) -> pd.DataFrame:
    for tbl in pd.read_html(io.StringIO(html)):
        try:
            return _normalize_table(tbl)
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("HTML에서 유효한 마진 표를 찾지 못함")


def fetch_latest() -> tuple[pd.DataFrame, str]:
    """(dataframe, source) 반환. xlsx 우선, 실패 시 HTML 폴백."""
    try:
        content = _http_get(XLSX_URL, referer=PAGE_URL)
        return load_from_xlsx(content), "xlsx"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] xlsx 경로 실패 -> HTML 폴백: {e}", file=sys.stderr)

    html = _http_get(PAGE_URL).decode("utf-8", "replace")
    return load_from_html(html), "html"


# --------------------------------------------------------------------------- #
# 병합 / 계산
# --------------------------------------------------------------------------- #
def merge_history(new: pd.DataFrame) -> pd.DataFrame:
    if HISTORY_CSV.exists():
        old = pd.read_csv(HISTORY_CSV, parse_dates=["month"])
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    merged = (
        merged.dropna(subset=CANON)
        .drop_duplicates(subset=["month"], keep="last")
        .sort_values("month")
        .reset_index(drop=True)
    )
    return merged


def _band(pct: float) -> tuple[str, str]:
    if pct <= 5:
        return "EXTREME_LOW", "#c0392b"   # 현금여력 고갈 = risk-on 과열(경고)
    if pct <= 25:
        return "LOW", "#e67e22"
    if pct < 75:
        return "NEUTRAL", "#7f8c8d"
    if pct < 95:
        return "HIGH", "#27ae60"          # dry powder 풍부(강세 연료)
    return "EXTREME_HIGH", "#16a085"


def build_badge(hist: pd.DataFrame, source: str) -> dict:
    hist = hist.copy()
    hist["free_credit"] = hist["fc_cash"] + hist["fc_margin"]
    hist["ratio"] = hist["free_credit"] / hist["debit"]
    hist["net_credit"] = hist["free_credit"] - hist["debit"]

    base = hist[hist["month"] >= BASELINE_START].copy()
    latest = hist.iloc[-1]
    r = float(latest["ratio"])

    # 백분위: 현재보다 낮은 비율의 비중(%). 낮을수록 현금여력 극단적으로 얇음.
    pct = round(float((base["ratio"] < r).mean()) * 100, 1)
    signal, color = _band(pct)

    as_of = pd.Timestamp(latest["month"]).strftime("%Y-%m")
    if pct <= 0:
        label = f"현금여력 사상 최저 · {as_of}"
    elif pct >= 100:
        label = f"현금여력 사상 최고 · {as_of}"
    elif pct <= 50:
        label = f"현금여력 하위 {pct:.0f}% · {as_of}"
    else:
        label = f"현금여력 상위 {100 - pct:.0f}% · {as_of}"

    return {
        "as_of": as_of,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "margin_debt_musd": int(latest["debit"]),
        "free_credit_total_musd": int(latest["free_credit"]),
        "net_credit_balance_musd": int(latest["net_credit"]),
        "cash_cover_ratio": round(r, 4),
        "percentile": pct,
        "percentile_basis": f"{BASELINE_START.strftime('%Y-%m')}~",
        "signal": signal,
        "color": color,
        "label": label,
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new, source = fetch_latest()
    if new.empty:
        print("[error] 신규 데이터가 비어 있음", file=sys.stderr)
        return 1

    hist = merge_history(new)
    hist.to_csv(HISTORY_CSV, index=False)

    badge = build_badge(hist, source)
    BADGE_JSON.write_text(json.dumps(badge, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] source={source} rows={len(hist)} "
          f"as_of={badge['as_of']} ratio={badge['cash_cover_ratio']} "
          f"pct={badge['percentile']} signal={badge['signal']}")
    print(json.dumps(badge, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
