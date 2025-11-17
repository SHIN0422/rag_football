# 분석.py
from typing import List, Tuple
import os, html
import requests
from datetime import datetime
from dotenv import load_dotenv
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
from typing import List, Dict, Any, Tuple, Optional

load_dotenv()
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
API_FOOTBALL_KEY  = os.getenv("APIFOOTBALL_KEY")  # 환경변수에 키 저장

# analysis.py 에 추가

def build_match_data_bundle(fixture_id: str, matches_list: list = None) -> dict:
    """
    하나의 경기에 대한 모든 관련 데이터를 API에서 가져와 하나의 딕셔너리로 묶습니다.
    (기존 rag.py의 _build_full_context_json 함수와 동일한 로직)
    """
    import json
    # fixture_id에 해당하는 경기 메타 정보 찾기
    meta = None
    if matches_list:
        for m in matches_list:
            if str(m.get("match_id")) == str(fixture_id):
                meta = m
                break

    # 필수 데이터: 선수 스탯, 팀 통계, 이벤트, 라인업
    try: rows, _ = fetch_player_stats_official(str(fixture_id))
    except Exception: rows = []
    try: tstats = fetch_team_match_stats_official(str(fixture_id))
    except Exception: tstats = []
    try: events = fetch_fixture_events_official(str(fixture_id))
    except Exception: events = []
    try: lineups = fetch_fixture_lineups_official(str(fixture_id))
    except Exception: lineups = []

    # 부가 정보 (시즌 통계, 부상)
    home_season, away_season = {}, {}
    inj_h, inj_a = [], []
    if meta and meta.get("league_id") and meta.get("season") and meta.get("home_id") and meta.get("away_id"):
        try:
            lid, ssn = int(meta["league_id"]), int(meta["season"])
            hid, aid = int(meta["home_id"]), int(meta["away_id"])
            home_season = fetch_team_season_stats_official(hid, lid, ssn) or {}
            away_season = fetch_team_season_stats_official(aid, lid, ssn) or {}
            inj_h = fetch_injuries_official(hid, lid, ssn) or []
            inj_a = fetch_injuries_official(aid, lid, ssn) or []
        except Exception:
            pass

    bundle = {
        "meta": meta or {"fixture_id": fixture_id}, # meta가 없어도 기본 정보는 포함
        "players": rows,
        "teams_match_stats": tstats,
        "events": events,
        "lineups": lineups,
        "season_summaries": {"home": home_season, "away": away_season},
        "injuries": {"home": inj_h, "away": inj_a},
    }
    return bundle

# "team.players.0.name" 형태로 평탄화
def _flatten(obj: Any, parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    dict/list를 재귀적으로 'a.b.c': value 형태로 평탄화.
    리스트는 인덱스를 키에 포함(예: 'arr.0.x').
    """
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent}{sep}{k}" if parent else str(k)
            if isinstance(v, (dict, list)):
                out.update(_flatten(v, key, sep))
            else:
                out[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{parent}{sep}{i}" if parent else str(i)
            if isinstance(v, (dict, list)):
                out.update(_flatten(v, key, sep))
            else:
                out[key] = v
    return out

# 숫자로 변환
def _fnum(x) -> Optional[float]:
    """숫자 변환 헬퍼: None/공백/'-' 등은 None으로."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-":
        return None
    try:
        return float(s.replace(",", ".").rstrip("%"))
    except Exception:
        return None

#각 선수의 stats.에서 가져온 스탯 데이터를 api에서 가져와 rows에 저장
def fetch_player_stats_official(fixture_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    한 경기(API-FOOTBALL fixture)의 모든 선수에 대해,
    API 응답의 'statistics' 블록을 '전부' 평탄화하여 포함한 rows를 반환.
    - 반환 rows 각 원소는:
        - 기존 호환 필드: team(이름), player, player_id, pos, min, rating 등
        - team.* : 팀 메타(예: team.id, team.name, team.logo ...)
        - player.* : 선수 메타(예: player.id, player.name, player.age ...)
        - stat.* : 통계 전부(예: stat.games.minutes, stat.dribbles.attempts, stat.cards.yellow, stat.saves.total ...)
        - stat_index : 동일 선수의 statistics가 여러 개일 때 인덱스
    - raw : API 원본 JSON
    """
    key = (globals().get("API_FOOTBALL_KEY")
           or os.getenv("APIFOOTBALL_KEY")
           or os.getenv("API_FOOTBALL_KEY"))
    if not key:
        raise RuntimeError("APIFOOTBALL_KEY가 필요합니다. (.env 설정)")

    headers = {"x-apisports-key": key}
    r = requests.get(
        f"{API_FOOTBALL_BASE}/fixtures/players",
        headers=headers,
        params={"fixture": str(fixture_id)},
        timeout=20,
    )
    r.raise_for_status() # 실패하면 예외 발생
    raw = r.json()
    resp = raw.get("response", []) or []

    rows: List[Dict[str, Any]] = []
    for team_block in resp:
        team_meta = (team_block.get("team") or {})
        team_name = team_meta.get("name")
        team_flat = {f"team.{k}": v for k, v in _flatten(team_meta).items()}

        for p in (team_block.get("players") or []):
            info = p.get("player", {}) or {}
            info_flat = {f"player.{k}": v for k, v in _flatten(info).items()}
            stats_list = p.get("statistics") or [{}]

            for idx, st in enumerate(stats_list):
                # 기존 호환 키(현재 UI/요약 로직 호환)
                games   = (st or {}).get("games", {}) or {}
                shots   = (st or {}).get("shots", {}) or {}
                goals   = (st or {}).get("goals", {}) or {}
                passes  = (st or {}).get("passes", {}) or {}
                tackles = (st or {}).get("tackles", {}) or {}
                duels   = (st or {}).get("duels", {}) or {}
                drib    = (st or {}).get("dribbles", {}) or {}
                fouls   = (st or {}).get("fouls", {}) or {}
                cards   = (st or {}).get("cards", {}) or {}
                pen     = (st or {}).get("penalty", {}) or {}

                base = {
                    "team": team_name,
                    "player": info.get("name"),
                    "player_id": info.get("id"),
                    "pos": games.get("position"),
                    "min": games.get("minutes"),
                    "rating": _fnum(games.get("rating")),
                    "goals": goals.get("total"),
                    "assists": goals.get("assists"),
                    "shots_on": shots.get("on"),
                    "shots_total": shots.get("total"),
                    "key_passes": passes.get("key"),
                    "passes": passes.get("total"),
                    "pass_acc": passes.get("accuracy"),
                    "tackles": tackles.get("total"),
                    "interceptions": tackles.get("interceptions"),
                    "duels_won": duels.get("won") if isinstance(duels, dict) else None,
                    "dribbles_succ": drib.get("success") if isinstance(drib, dict) else None,
                    "fouls_committed": fouls.get("committed") if isinstance(fouls, dict) else None,
                    "fouls_drawn": fouls.get("drawn") if isinstance(fouls, dict) else None,
                    "yellow": cards.get("yellow") if isinstance(cards, dict) else None,
                    "red": cards.get("red") if isinstance(cards, dict) else None,
                    "pen_scored": pen.get("scored") if isinstance(pen, dict) else None,
                    "pen_missed": pen.get("missed") if isinstance(pen, dict) else None,
                    # 식별/디버깅 편의
                    "stat_index": idx,
                }

                # 통계/선수/팀 전체를 평탄화해서 합치기
                st_flat = {f"stat.{k}": v for k, v in _flatten(st).items()}

                row = {**base, **team_flat, **info_flat, **st_flat}
                rows.append(row)

    return rows, raw

# 팀 단위 통계
def fetch_team_match_stats_official(fixture_id: str) -> list:
    """경기 단위 팀 통계 (fixtures/statistics)"""
    key = API_FOOTBALL_KEY or os.getenv("APIFOOTBALL_KEY")
    headers = {"x-apisports-key": key}
    r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/statistics",
                     headers=headers, params={"fixture": fixture_id}, timeout=15)
    r.raise_for_status()
    raw = r.json()
    return raw.get("response", [])

# 경기에서 일어나는 이벤트
def fetch_fixture_events_official(fixture_id: str) -> list:
    """골, 경고, 교체 등 이벤트 (fixtures/events)"""
    key = API_FOOTBALL_KEY or os.getenv("APIFOOTBALL_KEY")
    headers = {"x-apisports-key": key}
    r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/events",
                     headers=headers, params={"fixture": fixture_id}, timeout=15)
    r.raise_for_status()
    raw = r.json()
    return raw.get("response", [])

# 라인업 및 포메이션
def fetch_fixture_lineups_official(fixture_id: str) -> list:
    """라인업 및 포메이션 (fixtures/lineups)"""
    key = API_FOOTBALL_KEY or os.getenv("APIFOOTBALL_KEY")
    headers = {"x-apisports-key": key}
    r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/lineups",
                     headers=headers, params={"fixture": fixture_id}, timeout=15)
    r.raise_for_status()
    raw = r.json()
    return raw.get("response", [])

# 팀 시즌 통계
def fetch_team_season_stats_official(team_id: int, league_id: int, season: int) -> dict:
    """팀 시즌 통계 (teams/statistics)"""
    key = API_FOOTBALL_KEY or os.getenv("APIFOOTBALL_KEY")
    headers = {"x-apisports-key": key}
    r = requests.get(f"{API_FOOTBALL_BASE}/teams/statistics",
                     headers=headers,
                     params={"team": team_id, "league": league_id, "season": season},
                     timeout=15)
    r.raise_for_status()
    raw = r.json()
    return raw.get("response", {})

# 선수 부상 정보
def fetch_injuries_official(team_id: int, league_id: int, season: int) -> list:
    """선수 부상 정보 (injuries)"""
    key = API_FOOTBALL_KEY or os.getenv("APIFOOTBALL_KEY")
    headers = {"x-apisports-key": key}
    r = requests.get(f"{API_FOOTBALL_BASE}/injuries",
                     headers=headers,
                     params={"team": team_id, "league": league_id, "season": season},
                     timeout=15)
    r.raise_for_status()
    raw = r.json()
    return raw.get("response", [])

# 화면에 선수들 스탯 표시
def render_player_stats_html(rows: list, max_rows: int = 5) -> str:
    if not rows:
        return "<em>선수 스탯이 없습니다.</em>"

    # rows에 나타난 순서대로 홈/원정 팀 추정
    team_names = []
    for r in rows:
        t = r.get("team")
        if t and t not in team_names:
            team_names.append(t)
    team_order = {name: i for i, name in enumerate(team_names)}

    def sort_key(r):
        return (
            team_order.get(r.get("team"), 999),      # 먼저 팀별로 묶기
            -(r.get("rating") or -999),              # 팀 내에서 평점 높은 순
            -(r.get("min") or 0),                    # 그다음 출전 시간 많은 순
            -(r.get("goals") or 0),                  # 마지막으로 골 많은 순
        )
    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    cols = [
        "team","player","pos","min","rating","goals","assists","shots_on","shots_total",
        "key_passes","passes","pass_acc","tackles","interceptions","duels_won",
        "dribbles_succ","fouls_committed","fouls_drawn","yellow","red","pen_scored","pen_missed"
    ]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    # 상위 N개만 먼저 보여주기
    body_main, body_extra = [], []
    for i, r in enumerate(rows_sorted):
        tds = []
        for c in cols:
            v = r.get(c, "")
            v = f"{v:.2f}" if (c=="rating" and isinstance(v, float)) else v
            tds.append(f"<td>{html.escape(str(v))}</td>")
        row_html = "<tr>" + "".join(tds) + "</tr>"
        if i < max_rows:
            body_main.append(row_html)
        else:
            body_extra.append(row_html)

    # 최종 HTML 조립
    table_html = (
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr>" + head + "</tr></thead>"
        "<tbody>" + "".join(body_main) + "</tbody></table>"
    )

    if body_extra:  # 더보기 할 게 있으면 details 태그 추가
        table_html += (
            "<details><summary>더보기</summary>"
            "<table style='width:100%;border-collapse:collapse;margin-top:10px;'>"
            "<thead><tr>" + head + "</tr></thead>"
            "<tbody>" + "".join(body_extra) + "</tbody></table>"
            "</details>"
        )

    return table_html

# 경기 목록을 가져옴
def _call_fixtures(params: dict):
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    r = requests.get(f"{API_FOOTBALL_BASE}/fixtures", headers=headers, params=params, timeout=15)
    try:
        js = r.json()
    except Exception:
        print(f"[fixtures] JSON decode error. status={r.status_code}, text[:200]={r.text[:200]!r}")
        return []
    resp = js.get("response", []) or []
    errs = js.get("errors", {}) or {}
    results = js.get("results", len(resp))
    print(f"[fixtures] params={params} -> results={len(resp)} (api says {results}), errors={errs}")
    return resp

# 경기 시간을 한국 시간으로 변환
def _to_local_str(utc_iso: str, tz: str) -> str:
    if not utc_iso: return ""
    try:
        dt_utc = datetime.fromisoformat(utc_iso.replace("Z","+00:00"))
        return dt_utc.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return utc_iso

# 선택한 날짜의 경기 목록을 생성
def fetch_matches_official(date_ymd: str, tz: str = "Asia/Seoul"):
    """
    날짜별 경기 불러오기
      1) date=YYYY-MM-DD
      2) from=같은날 & to=같은날
    """
    if not API_FOOTBALL_KEY:
        raise RuntimeError("APIFOOTBALL_KEY가 설정되지 않았습니다.")
    
    resp = _call_fixtures({"date": date_ymd, "timezone": tz})

    if not resp:
        resp = _call_fixtures({"from": date_ymd, "to": date_ymd, "timezone": tz})

    # 표준화
    out = []
    for it in resp:
        fixture = it.get("fixture", {})
        league  = it.get("league", {})
        teams   = it.get("teams", {})
        out.append({
            "kickoff_local": _to_local_str(fixture.get("date"), tz),
            "league": f"{league.get('country','')}/{league.get('name','')}".strip("/"),
            "league_id": league.get("id"),
            "home": (teams.get("home") or {}).get("name"),
            "away": (teams.get("away") or {}).get("name"),
            "status": (fixture.get("status") or {}).get("short"),
            "match_id": fixture.get("id"),
        })
    return out

# 경기 리스트를 html 렌더링
def render_matches_html(matches: list, with_links: bool = False) -> str:
    if not matches:
        return "<em>해당 날짜의 경기가 없습니다.</em>"
    rows = []
    for m in matches:
        title = f"{m.get('home','?')} vs {m.get('away','?')}"
        if with_links:
            link  = m.get("url") or "#"
            title_html = f"<a href='{html.escape(link)}' target='_blank' rel='noopener noreferrer'>{html.escape(title)}</a>"
        else:
            title_html = html.escape(title)
        rows.append(
            f"<tr>"
            f"<td>{html.escape(m.get('kickoff_local',''))}</td>"
            f"<td>{html.escape(m.get('league',''))}</td>"
            f"<td>{title_html}</td>"
            f"<td>{html.escape(str(m.get('status','')))}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr><th>일시(KST)</th><th>리그</th><th>매치</th><th>상태</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )

#if __name__ == "__main__":
#    fid = "1390848"  # 테스트용 fixture ID
#    print(fetch_injuries_official(fid))