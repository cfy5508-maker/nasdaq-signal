"""
backtest_calibrate.py
사용법: python backtest_calibrate.py

동작:
  1) data/score_history.jsonl 에 쌓인 과거 스냅샷(날짜별 티커 점수/스테이지 상태)을 읽는다.
  2) 각 스냅샷 시점 이후 실제 주가(5/10/20거래일 후)를 yfinance로 조회해 수익률을 라벨링한다.
     (해당 거래일 수가 아직 안 지난 최근 스냅샷은 해당 horizon만 "미확정"으로 남김)
  3) 스냅샷 시점의 시장 국면(QQQ 200일선 대비 +-2%로 상승장/하락장/보합장 판정)을 함께 라벨링한다.
  4) 국면별 + 스테이지별로 pass/fail 그룹의 승률·평균수익률을 집계한다.
     판단 기준은 5일 승률(단기)을 우선한다 - 단기 신호일수록 대응하기 쉽기 때문.
  5) 국면별 가중치(WEIGHTS/ADDON_WEIGHTS)를 자동으로 조정해 data/adaptive_weights.json에 저장한다.
     - 매번 목표치로 한번에 점프하지 않고, 이전 값에서 30%만 이동(damping)해 급변동 방지.
     - 조정폭은 기본 가중치의 0.4~1.8배로 하드 클립.
     - 표본이 부족한(국면당 15건 미만) 스테이지는 조정하지 않고 이전 값을 유지.
     - fetch_indicators.py는 이 파일을 읽어서 "오늘의 시장 국면"에 맞는 가중치를 자동 적용한다.

출력:
  - data/score_history_labeled.jsonl   : 라벨링된 원본 데이터 (재계산 시 매번 덮어씀)
  - data/calibration_report.json       : 집계 결과 (다른 스크립트/대시보드에서 재사용 가능)
  - data/adaptive_weights.json         : 국면별 자동조정 가중치 (fetch_indicators.py가 읽어감)
  - docs/calibration_report.md         : 사람이 읽는 리포트 (GitHub Issue 본문으로도 사용)
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_indicators import get_confirmed_history, WEIGHTS, ADDON_WEIGHTS  # noqa: E402

HISTORY_PATH = "data/score_history.jsonl"
LABELED_PATH = "data/score_history_labeled.jsonl"
REPORT_JSON_PATH = "data/calibration_report.json"
REPORT_MD_PATH = "docs/calibration_report.md"
ADAPTIVE_WEIGHTS_PATH = "data/adaptive_weights.json"
NOTIFICATIONS_PATH = "data/notifications.json"
REPORT_URL = "https://github.com/cfy5508-maker/nasdaq-signal/blob/main/docs/calibration_report.md"
MAX_NOTIFICATIONS_KEPT = 200

FORWARD_HORIZONS = {"fwd_5d": 5, "fwd_10d": 10, "fwd_20d": 20}
PRIMARY_HORIZON = "fwd_5d"          # 단기 승률을 우선 기준으로 삼는다 (대응하기 쉬우니까)
WIN_THRESHOLD = {"fwd_5d": 0.02, "fwd_10d": 0.025, "fwd_20d": 0.03}

MIN_SAMPLE_FOR_VERDICT = 15         # 국면별로 쪼개다 보니 표본이 줄어서 기존 20 -> 15로 완화
DISCRIMINATION_GAP = 0.10           # pass vs fail 승률 차이가 이 값보다 작으면 "변별력 낮음"

REGIME_PROXY = "QQQ"
REGIME_UP = 0.02       # 200일선 대비 +2% 이상 -> 상승장
REGIME_DOWN = -0.02    # 200일선 대비 -2% 이하 -> 하락장
REGIMES = ["bull", "bear", "sideways"]

ALPHA = 0.3            # damping: 목표 가중치 방향으로 이번에 30%만 이동
MULT_MIN, MULT_MAX = 0.4, 1.8   # 기본 가중치 대비 허용 배율 (안전장치)

# ── 실패 패턴 공통점 자동 분석 ──────────────────────────────
LOSS_THRESHOLD = -0.02      # 5일 후 -2% 이하면 "실패"로 간주
MIN_N_FOR_LIFT = 10         # 이 이하 표본이면 lift 계산 안 함 (우연일 가능성)
LIFT_FLAG = 1.3             # 실패군 비중이 전체 대비 이 배수 이상이면 "과대표됨"으로 표시
MIN_N_FOR_TICKER = 5        # 티커별 반복실패 판정 최소 표본
TICKER_LOSS_GAP_FLAG = 0.15 # 전체 평균보다 이 이상 실패율 높으면 "반복실패 티커"로 표시
MIN_N_FOR_SECTOR = 8        # 섹터별 약화 판정 최소 표본
SECTOR_LOSS_GAP_FLAG = 0.15 # 전체 평균보다 이 이상 실패율 높으면 "약한 섹터"로 표시


def load_snapshots(path):
    if not os.path.exists(path):
        print(f"{path} 없음 - 아직 쌓인 스냅샷이 없습니다.")
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_ticker_price_index(tickers):
    """티커별로 (날짜 -> 종가, 정렬된 거래일 리스트)를 한 번씩만 조회해서 캐싱."""
    index = {}
    for i, ticker in enumerate(sorted(tickers)):
        try:
            hist = get_confirmed_history(ticker, period="2y")
        except Exception as e:
            print(f"  [{ticker}] 가격 조회 실패: {e}")
            continue
        if hist is None or hist.empty:
            continue
        dates = [d.strftime("%Y-%m-%d") for d in hist.index]
        closes = hist["Close"].tolist()
        index[ticker] = {"dates": dates, "closes": closes}
        if (i + 1) % 10 == 0:
            print(f"  가격 데이터 조회 중... ({i + 1}/{len(tickers)})")
    return index


def forward_return(price_map, snap_date, horizon_days):
    dates = price_map["dates"]
    closes = price_map["closes"]
    pos = None
    for i, d in enumerate(dates):
        if d >= snap_date:
            pos = i
            break
    if pos is None:
        return None
    target = pos + horizon_days
    if target >= len(dates):
        return None
    base_price = closes[pos]
    fwd_price = closes[target]
    if not base_price:
        return None
    return (fwd_price - base_price) / base_price


def build_sector_map(tickers, known_sectors):
    """known_sectors: {ticker: sector 또는 None} (score_history.jsonl에 이미 기록된 값).
    없는 티커만 yfinance로 조회해서 채운다 - 과거(섹터 필드 도입 전) 스냅샷도 분석 가능하도록."""
    sector_map = dict(known_sectors)
    missing = [t for t in tickers if not sector_map.get(t)]
    for i, t in enumerate(missing):
        try:
            sector_map[t] = yf.Ticker(t).info.get("sector")
        except Exception:
            sector_map[t] = None
        if (i + 1) % 10 == 0:
            print(f"  섹터 정보 조회 중... ({i + 1}/{len(missing)})")
    return sector_map


def build_regime_series():
    """QQQ 200일선 대비 이격도로 날짜별 시장 국면(bull/bear/sideways)을 계산."""
    hist = get_confirmed_history(REGIME_PROXY, period="5y")
    if hist is None or hist.empty:
        return {}
    close = hist["Close"]
    sma200 = close.rolling(200).mean()
    dev = (close - sma200) / sma200
    regimes = {}
    for i in range(len(hist)):
        d = dev.iloc[i]
        if pd.isna(d):
            continue
        date_str = hist.index[i].strftime("%Y-%m-%d")
        if d >= REGIME_UP:
            regimes[date_str] = "bull"
        elif d <= REGIME_DOWN:
            regimes[date_str] = "bear"
        else:
            regimes[date_str] = "sideways"
    return regimes


def regime_on(sorted_regime_dates, regimes, snap_date):
    """snap_date 이전(포함) 가장 최근에 알려진 국면. 없으면 None."""
    lo, hi = 0, len(sorted_regime_dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_regime_dates[mid] <= snap_date:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return None
    return regimes[sorted_regime_dates[lo - 1]]


def label_snapshots(snapshots, price_index, regimes):
    sorted_regime_dates = sorted(regimes.keys())
    labeled = []
    for snap in snapshots:
        ticker = snap.get("ticker")
        pmap = price_index.get(ticker)
        row = dict(snap)
        if pmap is None:
            for key in FORWARD_HORIZONS:
                row[key] = None
        else:
            for key, horizon in FORWARD_HORIZONS.items():
                row[key] = forward_return(pmap, snap["date"], horizon)
        row["regime"] = regime_on(sorted_regime_dates, regimes, snap["date"])
        labeled.append(row)
    return labeled


def summarize_stage(sub_df, status_col, label):
    """특정 스테이지 상태(pass/warn/fail)별 5/10/20일 승률·평균수익률 집계."""
    out = []
    if status_col not in sub_df.columns:
        return out
    sub = sub_df.dropna(subset=[PRIMARY_HORIZON])
    if sub.empty:
        return out
    for status, grp in sub.groupby(status_col):
        n = len(grp)
        row = {"stage": label, "status": status, "n": n}
        for hkey in FORWARD_HORIZONS:
            valid = grp[hkey].dropna()
            if len(valid) == 0:
                row[f"win_rate_{hkey}"] = None
                row[f"avg_return_{hkey}"] = None
            else:
                row[f"win_rate_{hkey}"] = round(float((valid >= WIN_THRESHOLD[hkey]).mean()), 4)
                row[f"avg_return_{hkey}"] = round(float(valid.mean()), 4)
        out.append(row)
    out.sort(key=lambda r: r["status"])
    return out


def build_verdict(stage_rows, current_weight):
    by_status = {r["status"]: r for r in stage_rows}
    pass_r, fail_r = by_status.get("pass"), by_status.get("fail")
    if pass_r is None or fail_r is None:
        return "표본 부족 (pass/fail 둘 다 없음) - 판단 보류", None
    if pass_r["n"] < MIN_SAMPLE_FOR_VERDICT or fail_r["n"] < MIN_SAMPLE_FOR_VERDICT:
        return f"표본 부족 (pass n={pass_r['n']}, fail n={fail_r['n']}) - 판단 보류", None
    pw = pass_r[f"win_rate_{PRIMARY_HORIZON}"]
    fw = fail_r[f"win_rate_{PRIMARY_HORIZON}"]
    if pw is None or fw is None:
        return "표본 부족 - 판단 보류", None
    gap = pw - fw
    text = f"5일 승률 pass {pw:.0%} vs fail {fw:.0%} (차이 {gap:.0%})"
    if gap < DISCRIMINATION_GAP:
        return f"{text} - 변별력 낮음, 가중치 하향", gap
    if gap >= 0.30:
        return f"{text} - 변별력 높음, 가중치 상향", gap
    return f"{text} - 적정 유지", gap


def target_multiplier(gap):
    """5일 승률 gap을 가중치 배율로 변환 (gap=0.10을 기준선 1.0으로)."""
    mult = 1.0 + 2.0 * (gap - DISCRIMINATION_GAP)
    return max(MULT_MIN, min(MULT_MAX, mult))


def load_prev_adaptive_weights():
    if not os.path.exists(ADAPTIVE_WEIGHTS_PATH):
        return {}
    try:
        with open(ADAPTIVE_WEIGHTS_PATH) as f:
            return json.load(f).get("regimes", {})
    except Exception:
        return {}


def adjust_weights(regime, group_label, default_weights, breakdown_by_stage, prev_regimes):
    """국면별로 기본 가중치를 스테이지별 5일 gap 기준으로 자동 조정 (damping+클립 적용)."""
    prev_weights = (prev_regimes.get(regime, {}) or {}).get(group_label, {})
    updated, changes = {}, []
    for stage, default_w in default_weights.items():
        rows = breakdown_by_stage.get(stage, [])
        verdict_text, gap = build_verdict(rows, default_w)
        prev_w = prev_weights.get(stage, default_w)
        if gap is None:
            updated[stage] = round(prev_w, 3)
            changes.append({"stage": stage, "before": round(prev_w, 3), "after": round(prev_w, 3),
                             "reason": verdict_text})
            continue
        mult = target_multiplier(gap)
        target_w = default_w * mult
        new_w = prev_w * (1 - ALPHA) + target_w * ALPHA
        new_w = max(default_w * MULT_MIN, min(default_w * MULT_MAX, new_w))
        updated[stage] = round(new_w, 3)
        changes.append({"stage": stage, "before": round(prev_w, 3), "after": round(new_w, 3),
                         "reason": verdict_text})
    return updated, changes


def load_notifications():
    if not os.path.exists(NOTIFICATIONS_PATH):
        return []
    try:
        with open(NOTIFICATIONS_PATH) as f:
            return json.load(f).get("notifications", [])
    except Exception:
        return []


def collect_change_notifications(report_by_regime, generated_at):
    """이전 값과 실제로 달라진(=0.001 이상 차이) 가중치 변경만 알림으로 남긴다."""
    regime_kr = {"bull": "상승장", "bear": "하락장", "sideways": "보합장"}
    notifs = []
    for regime, info in report_by_regime.items():
        for group_label, changes in [("신규진입", info["main_changes"]), ("추가매수", info["addon_changes"])]:
            for c in changes:
                if abs(c["after"] - c["before"]) < 0.001:
                    continue
                notifs.append({
                    "id": f"{generated_at}-{regime}-{group_label}-{c['stage']}",
                    "date": generated_at[:10],
                    "regime": regime, "regime_kr": regime_kr[regime],
                    "group": group_label, "stage": c["stage"],
                    "before": c["before"], "after": c["after"], "reason": c["reason"],
                })
    return notifs


def write_notifications(report_by_regime, generated_at):
    new_notifs = collect_change_notifications(report_by_regime, generated_at)
    existing = load_notifications()
    all_notifs = (existing + new_notifs)[-MAX_NOTIFICATIONS_KEPT:]
    os.makedirs(os.path.dirname(NOTIFICATIONS_PATH) or ".", exist_ok=True)
    with open(NOTIFICATIONS_PATH, "w") as f:
        json.dump({"updated_at": generated_at, "report_url": REPORT_URL, "notifications": all_notifs},
                   f, ensure_ascii=False, indent=2)
    print(f"알림 {len(new_notifs)}건 추가 (누적 {len(all_notifs)}건, 실제 변경 없으면 0건)")


def analyze_failure_patterns(df, main_stage_status, addon_stage_status):
    """5일 후 -2% 이하로 실패한 스냅샷들의 공통점을 자동으로 찾는다:
       1) 어떤 스테이지 상태가 실패군에 유독 많이 나타나는지 (lift)
       2) 어떤 티커가 반복적으로 실패하는지
       3) 어떤 시장 국면에서 실패율이 높은지
    사람이 미리 지정한 규칙이 아니라, 데이터에서 나온 패턴만 표시한다."""
    base = df.dropna(subset=[PRIMARY_HORIZON])
    if base.empty:
        return None
    is_loss = base[PRIMARY_HORIZON] <= LOSS_THRESHOLD
    loss_df = base[is_loss]
    overall_loss_rate = float(is_loss.mean())
    n_loss = int(is_loss.sum())

    overrep = []
    for status_df, stages in [(main_stage_status, WEIGHTS.keys()), (addon_stage_status, ADDON_WEIGHTS.keys())]:
        for stage in stages:
            if stage not in status_df.columns:
                continue
            col = status_df.loc[base.index, stage]
            baseline_counts = col.value_counts(normalize=True)
            loss_counts_raw = status_df.loc[loss_df.index, stage].value_counts()
            loss_counts = status_df.loc[loss_df.index, stage].value_counts(normalize=True)
            for status, loss_pct in loss_counts.items():
                n_loss_status = int(loss_counts_raw.get(status, 0))
                baseline_pct = float(baseline_counts.get(status, 0))
                if n_loss_status < MIN_N_FOR_LIFT or baseline_pct <= 0:
                    continue
                lift = loss_pct / baseline_pct
                if lift >= LIFT_FLAG:
                    overrep.append({
                        "stage": stage, "status": status, "n_loss": n_loss_status,
                        "loss_group_pct": round(float(loss_pct), 3),
                        "baseline_pct": round(baseline_pct, 3),
                        "lift": round(float(lift), 2),
                    })
    overrep.sort(key=lambda r: -r["lift"])

    ticker_stats = base.groupby("ticker")[PRIMARY_HORIZON].apply(
        lambda s: pd.Series({"n": len(s), "loss_rate": float((s <= LOSS_THRESHOLD).mean())})
    ).unstack()
    repeat_offenders = []
    if not ticker_stats.empty:
        flagged = ticker_stats[(ticker_stats["n"] >= MIN_N_FOR_TICKER) &
                                (ticker_stats["loss_rate"] >= overall_loss_rate + TICKER_LOSS_GAP_FLAG)]
        for ticker, row in flagged.sort_values("loss_rate", ascending=False).iterrows():
            repeat_offenders.append({"ticker": ticker, "n": int(row["n"]), "loss_rate": round(float(row["loss_rate"]), 3)})

    weak_sectors = []
    if "sector" in base.columns:
        sector_base = base.dropna(subset=["sector"])
        if not sector_base.empty:
            sector_stats = sector_base.groupby("sector")[PRIMARY_HORIZON].apply(
                lambda s: pd.Series({"n": len(s), "loss_rate": float((s <= LOSS_THRESHOLD).mean())})
            ).unstack()
            flagged_sectors = sector_stats[(sector_stats["n"] >= MIN_N_FOR_SECTOR) &
                                            (sector_stats["loss_rate"] >= overall_loss_rate + SECTOR_LOSS_GAP_FLAG)]
            for sector, row in flagged_sectors.sort_values("loss_rate", ascending=False).iterrows():
                weak_sectors.append({"sector": sector, "n": int(row["n"]), "loss_rate": round(float(row["loss_rate"]), 3)})

    regime_loss_rate = {}
    if "regime" in base.columns:
        for regime in REGIMES:
            sub = base[base["regime"] == regime]
            if len(sub) >= MIN_N_FOR_LIFT:
                regime_loss_rate[regime] = {"n": int(len(sub)),
                                             "loss_rate": round(float((sub[PRIMARY_HORIZON] <= LOSS_THRESHOLD).mean()), 3)}

    return {
        "loss_threshold": LOSS_THRESHOLD,
        "overall_loss_rate": round(overall_loss_rate, 3),
        "n_loss_samples": n_loss,
        "n_total_samples": int(len(base)),
        "stage_status_overrepresented": overrep[:15],
        "repeat_offender_tickers": repeat_offenders[:10],
        "weak_sectors": weak_sectors[:10],
        "regime_loss_rate": regime_loss_rate,
    }


def main():
    print("1) score_history.jsonl 로딩...")
    snapshots = load_snapshots(HISTORY_PATH)
    if not snapshots:
        print("스냅샷이 없어 종료합니다. (fetch_indicators.py가 매일 쌓는 걸 기다려주세요)")
        return

    tickers = {s["ticker"] for s in snapshots if s.get("ticker")}
    print(f"   스냅샷 {len(snapshots)}건, 티커 {len(tickers)}개")

    print("2) 가격 데이터 조회 및 5/10/20일 forward return 라벨링...")
    price_index = build_ticker_price_index(tickers)

    print("3) 시장 국면(QQQ 200일선 기준) 라벨링...")
    regimes = build_regime_series()

    labeled = label_snapshots(snapshots, price_index, regimes)

    os.makedirs(os.path.dirname(LABELED_PATH) or ".", exist_ok=True)
    with open(LABELED_PATH, "w") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(labeled)

    print("3-1) 섹터 정보 보강...")
    known_sectors = {}
    if "sector" in df.columns:
        for t, sec in zip(df["ticker"], df["sector"]):
            if sec and t not in known_sectors:
                known_sectors[t] = sec
    sector_map = build_sector_map(tickers, known_sectors)
    df["sector"] = df["ticker"].map(sector_map)

    labeled_n = int(df[PRIMARY_HORIZON].notna().sum()) if PRIMARY_HORIZON in df.columns else 0
    print(f"   5일 후 결과가 확정된 스냅샷: {labeled_n}건 (나머지는 아직 진행 중)")
    if labeled_n == 0:
        print("확정된 스냅샷이 아직 없어 리포트를 만들 수 없습니다. 다음 실행에서 다시 시도하세요.")
        return

    main_stage_status = pd.json_normalize(df["stage_status"]) if "stage_status" in df.columns else pd.DataFrame()
    addon_stage_status = pd.json_normalize(df["stage_status_addon"]) if "stage_status_addon" in df.columns else pd.DataFrame()

    prev_regimes = load_prev_adaptive_weights()

    print("4) 국면별·스테이지별 집계 및 가중치 자동조정...")
    report_by_regime = {}
    adaptive_out = {"generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                    "primary_horizon": PRIMARY_HORIZON, "regimes": {}}

    regime_counts = df["regime"].value_counts(dropna=True).to_dict()

    for regime in REGIMES:
        regime_mask = df["regime"] == regime
        sub_df = df[regime_mask]
        n_regime = int(regime_mask.sum())

        main_breakdown, addon_breakdown = {}, {}
        for stage_key in WEIGHTS:
            if stage_key not in main_stage_status.columns:
                continue
            merged = pd.concat([sub_df[list(FORWARD_HORIZONS)], main_stage_status.loc[sub_df.index, [stage_key]]], axis=1)
            main_breakdown[stage_key] = summarize_stage(merged, stage_key, stage_key)
        for stage_key in ADDON_WEIGHTS:
            if stage_key not in addon_stage_status.columns:
                continue
            merged = pd.concat([sub_df[list(FORWARD_HORIZONS)], addon_stage_status.loc[sub_df.index, [stage_key]]], axis=1)
            addon_breakdown[stage_key] = summarize_stage(merged, stage_key, stage_key)

        main_updated, main_changes = adjust_weights(regime, "main", WEIGHTS, main_breakdown, prev_regimes)
        addon_updated, addon_changes = adjust_weights(regime, "addon", ADDON_WEIGHTS, addon_breakdown, prev_regimes)

        adaptive_out["regimes"][regime] = {"main": main_updated, "addon": addon_updated, "n_snapshots": n_regime}
        report_by_regime[regime] = {
            "n_snapshots": n_regime,
            "main_breakdown": main_breakdown, "addon_breakdown": addon_breakdown,
            "main_changes": main_changes, "addon_changes": addon_changes,
        }

    with open(ADAPTIVE_WEIGHTS_PATH, "w") as f:
        json.dump(adaptive_out, f, ensure_ascii=False, indent=2)

    write_notifications(report_by_regime, adaptive_out["generated_at"])

    print("4-1) 실패 패턴 공통점 자동 분석...")
    failure_patterns = analyze_failure_patterns(df, main_stage_status, addon_stage_status)

    overall_win_rate = {h: float((df[h].dropna() >= WIN_THRESHOLD[h]).mean()) if df[h].notna().any() else None
                         for h in FORWARD_HORIZONS}
    report = {
        "generated_at": adaptive_out["generated_at"],
        "labeled_snapshots": labeled_n,
        "total_snapshots": len(snapshots),
        "regime_counts": regime_counts,
        "overall_win_rate": overall_win_rate,
        "by_regime": report_by_regime,
        "failure_patterns": failure_patterns,
    }
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("5) 마크다운 리포트 작성...")
    os.makedirs(os.path.dirname(REPORT_MD_PATH) or ".", exist_ok=True)
    md = [
        "# 📊 매매 계산식 캘리브레이션 리포트\n",
        f"_생성: {report['generated_at']} · 확정 스냅샷 {labeled_n}건 / 전체 {len(snapshots)}건 "
        f"· 판단 기준: 5일 승률(단기 우선) · 승 기준: 5일 +{WIN_THRESHOLD['fwd_5d']:.0%} / "
        f"10일 +{WIN_THRESHOLD['fwd_10d']:.0%} / 20일 +{WIN_THRESHOLD['fwd_20d']:.0%}_\n",
    ]
    for h in FORWARD_HORIZONS:
        wr = overall_win_rate[h]
        md.append(f"- 전체 승률({h}): {wr:.1%}" if wr is not None else f"- 전체 승률({h}): 데이터 없음")
    md.append("\n**가중치는 이번 실행에서 자동으로 조정되어 `data/adaptive_weights.json`에 반영되었습니다** "
               "(국면당 표본 15건 미만인 스테이지는 이전 값을 그대로 유지, 급변동 방지를 위해 매번 목표치의 30%만 이동).\n")

    regime_kr = {"bull": "상승장", "bear": "하락장", "sideways": "보합장"}

    if failure_patterns:
        fp = failure_patterns
        md.append(f"## 🔍 실패 패턴 공통점 (자동 분석, 사람 규칙 아님)\n")
        md.append(f"_5일 후 {fp['loss_threshold']:.0%} 이하를 '실패'로 정의 · 실패 {fp['n_loss_samples']}건 / "
                   f"전체 {fp['n_total_samples']}건 (전체 실패율 {fp['overall_loss_rate']:.1%})_\n")

        if fp["stage_status_overrepresented"]:
            md.append("### 실패군에 유독 많이 나타나는 스테이지 상태\n")
            md.append("_lift = (실패군 내 비중) / (전체 내 비중). 1.3배 이상이면 표시_\n")
            md.append("| 스테이지 | 상태 | 실패군 내 비중 | 전체 비중 | lift | n(실패) |")
            md.append("|---|---|---|---|---|---|")
            for r in fp["stage_status_overrepresented"]:
                md.append(f"| {r['stage']} | {r['status']} | {r['loss_group_pct']:.0%} | "
                          f"{r['baseline_pct']:.0%} | {r['lift']}x | {r['n_loss']} |")
        else:
            md.append("_표본 부족으로 뚜렷한 스테이지 패턴 없음._\n")

        if fp["repeat_offender_tickers"]:
            md.append("\n### 반복적으로 실패하는 티커\n")
            md.append(f"_전체 평균 실패율보다 {TICKER_LOSS_GAP_FLAG:.0%}p 이상 높고, 표본 {MIN_N_FOR_TICKER}건 이상인 티커_\n")
            md.append("| 티커 | n | 실패율 |")
            md.append("|---|---|---|")
            for r in fp["repeat_offender_tickers"]:
                md.append(f"| {r['ticker']} | {r['n']} | {r['loss_rate']:.0%} |")

        if fp["weak_sectors"]:
            md.append("\n### 실패율이 높은 섹터 (섹터 약화 가능성)\n")
            md.append(f"_전체 평균 실패율보다 {SECTOR_LOSS_GAP_FLAG:.0%}p 이상 높고, 표본 {MIN_N_FOR_SECTOR}건 이상인 섹터_\n")
            md.append("| 섹터 | n | 실패율 |")
            md.append("|---|---|---|")
            for r in fp["weak_sectors"]:
                md.append(f"| {r['sector']} | {r['n']} | {r['loss_rate']:.0%} |")

        if fp["regime_loss_rate"]:
            md.append("\n### 국면별 실패율\n")
            md.append("| 국면 | n | 실패율 |")
            md.append("|---|---|---|")
            for regime, r in fp["regime_loss_rate"].items():
                md.append(f"| {regime_kr.get(regime, regime)} | {r['n']} | {r['loss_rate']:.0%} |")
        md.append("")

    for regime in REGIMES:
        info = report_by_regime[regime]
        md.append(f"## {regime_kr[regime]} ({regime}) - 표본 {info['n_snapshots']}건\n")
        md.append("### 가중치 자동조정 내역\n")
        md.append("| 그룹 | 스테이지 | 이전 | 이후 | 근거 |")
        md.append("|---|---|---|---|---|")
        for group, changes in [("신규진입", info["main_changes"]), ("추가매수", info["addon_changes"])]:
            for c in changes:
                md.append(f"| {group} | {c['stage']} | {c['before']} | {c['after']} | {c['reason']} |")

        md.append("\n### 상세 breakdown (5일 / 10일 / 20일 승률)\n")
        for group, breakdown in [("신규진입", info["main_breakdown"]), ("추가매수", info["addon_breakdown"])]:
            for stage, rows in breakdown.items():
                if not rows:
                    continue
                md.append(f"**{group} · {stage}**\n")
                md.append("| 상태 | n | 5일 승률 | 10일 승률 | 20일 승률 |")
                md.append("|---|---|---|---|---|")
                for r in rows:
                    def fmt(k):
                        v = r.get(k)
                        return f"{v:.1%}" if v is not None else "-"
                    md.append(f"| {r['status']} | {r['n']} | {fmt('win_rate_fwd_5d')} | "
                              f"{fmt('win_rate_fwd_10d')} | {fmt('win_rate_fwd_20d')} |")
                md.append("")

    with open(REPORT_MD_PATH, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"완료: {REPORT_MD_PATH}, {REPORT_JSON_PATH}, {ADAPTIVE_WEIGHTS_PATH}, {LABELED_PATH}")


if __name__ == "__main__":
    main()
