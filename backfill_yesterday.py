"""
backfill_yesterday.py
사용법: python backfill_yesterday.py

동작: watchlist.json의 모든 종목에 대해, "어제(직전 거래일) 종가 기준"으로
      새 계산식(5단계 구조)을 소급 적용해서 score_history.jsonl에 기록한다.
      score_history.jsonl을 리셋한 직후, 오늘 바로 상승/하락 신호 비교가
      가능하도록 "기준점(어제 점수)"을 미리 만들어두는 용도.
      today(오늘)의 data/<TICKER>.json은 건드리지 않는다(write_file=False).
"""
import json
import time
from fetch_indicators import analyze, log_snapshot, WATCHLIST_PATH, HISTORY_PATH


def already_logged(ticker, date_str):
    try:
        with open(HISTORY_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("ticker") == ticker and rec.get("date") == date_str:
                    return True
    except FileNotFoundError:
        return False
    return False


def main():
    with open(WATCHLIST_PATH) as f:
        tickers = json.load(f)

    print(f"=== {len(tickers)}개 종목, 어제 기준 소급 계산 시작 ===\n")
    for t in tickers:
        try:
            r = analyze(t, trim_days=1, write_file=False)
            date_str = r["as_of_date"]
            if already_logged(t, date_str):
                print(f"  {t}: {date_str} 기록 이미 존재, 스킵")
                continue
            log_snapshot(r)
            print(f"  {t}: {date_str} 기준 score={r['score']} score_addon={r['score_addon']} 기록 완료")
        except Exception as e:
            print(f"  {t}: 실패 ({e})")
        time.sleep(0.6)

    print("\n=== 백필 완료 ===")


if __name__ == "__main__":
    main()
