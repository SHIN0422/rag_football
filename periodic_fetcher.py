# periodic_fetcher.py

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# analysis.py의 함수들을 가져옵니다.
import analysis

# 데이터 저장 경로 설정 (없으면 생성)
DATA_DIR = Path(__file__).parent / "game_data"
DATA_DIR.mkdir(exist_ok=True)

# API 호출 간격 (초)
FETCH_INTERVAL_SECONDS = 3600  # 1시간

def fetch_and_save_finished_matches():
    """종료된 경기를 찾아 데이터를 파일로 저장합니다."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 종료된 경기 데이터 확인 시작...")

    # 오늘과 어제 날짜를 기준으로 경기 목록을 가져옵니다.
    # 경기 종료 시점이 애매할 수 있으므로 범위를 넓게 잡습니다.
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        matches_today = analysis.fetch_matches_official(today)
        matches_yesterday = analysis.fetch_matches_official(yesterday)
        all_matches = matches_today + matches_yesterday
    except Exception as e:
        print(f"  [오류] 경기 목록을 가져오는 데 실패했습니다: {e}")
        return

    finished_statuses = {"FT", "AET", "PEN"} # 종료 상태 코드 (Full Time, After Extra Time, Penalty)

    for match_summary in all_matches:
        status = match_summary.get("status")
        fixture_id = match_summary.get("match_id")

        if not fixture_id or status not in finished_statuses:
            continue

        # 이미 파일이 저장되어 있는지 확인
        save_path = DATA_DIR / f"{fixture_id}.json"
        if save_path.exists():
            continue # 이미 처리된 경기는 건너뜁니다.

        print(f"  [발견] 새로운 종료 경기: {match_summary.get('home')} vs {match_summary.get('away')} (ID: {fixture_id})")

        try:
            # 경기에 대한 모든 데이터를 가져옵니다.
            # build_match_data_bundle 함수는 match_summary 외에 더 많은 정보를 API에서 가져옵니다.
            full_data_bundle = analysis.build_match_data_bundle(str(fixture_id), all_matches)
            
            # JSON 파일로 저장
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(full_data_bundle, f, ensure_ascii=False, indent=2)
            
            print(f"  [성공] 경기 데이터 저장 완료: {save_path}")
            time.sleep(5) # API 과부하 방지를 위한 짧은 대기

        except Exception as e:
            print(f"  [오류] 경기 ID {fixture_id} 데이터 처리 중 오류 발생: {e}")

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 확인 완료. 다음 확인까지 {FETCH_INTERVAL_SECONDS}초 대기합니다.")


if __name__ == "__main__":
    while True:
        fetch_and_save_finished_matches()
        time.sleep(FETCH_INTERVAL_SECONDS)