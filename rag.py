from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
import gradio as gr
import re
import json
from pathlib import Path
import math
import html
from urllib.parse import urlparse
from datetime import date, datetime
from datetime import datetime as dt, timedelta
from gradio_flatpickr_calendar import Calendar

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = None

# ★ 추가: 유럽 5대 리그(API-FOOTBALL 리그 ID)
LEAGUE_ID = {
    "EPL": 39,         # 잉글랜드 프리미어리그
    "LALIGA": 140,     # 스페인 라리가
    "SERIEA": 135,     # 이탈리아 세리에 A
    "BUNDES": 78,      # 독일 분데스리가
    "LIGUE1": 61,      # 프랑스 리그 1
}

# matches: 경기 정보를 담은 딕셔너리 리스트
# keep_ids: 사용자가 체크박스로 선택한 리그 id
def _filter_by_league_ids(matches: list[dict], keep_ids: set[int]) -> list[dict]:
    """league_id 가 keep_ids 안에 있는 경기만 남김. keep_ids가 비어있으면 그대로 반환."""
    if not keep_ids:
        return matches
    out = []
    for m in matches or []:
        try:
            lid = int(m.get("league_id") or -1)
        except Exception:
            lid = -1
        if lid in keep_ids:
            out.append(m)
    return out


def apply_preset(preset: str):
    """프리셋 → (시작일, 종료일) 'YYYY-MM-DD' 문자열 튜플. 잘못된 프리셋이면 빈 문자열."""
    if not preset or preset == "날짜 지정 안함":
        return "", ""

    now = dt.now(KST) if KST else dt.now()
    today = now.date()

    if preset == "오늘":
        s = e = today
    elif preset == "어제":
        d1 = today - timedelta(days=1)
        s = e = d1
    elif preset == "최근 7일":
        s, e = today - timedelta(days=6), today
    elif preset == "최근 30일":
        s, e = today - timedelta(days=29), today
    elif preset == "이번 주":
        s = today - timedelta(days=today.weekday())  # 월요일 시작
        e = s + timedelta(days=6)
    elif preset == "이번 달":
        s = today.replace(day=1)
        nm = (s.replace(year=s.year+1, month=1, day=1)
              if s.month == 12 else s.replace(month=s.month+1, day=1))
        e = (nm - timedelta(days=1))
    elif preset == "지난 달":
        first_this = today.replace(day=1)
        e = first_this - timedelta(days=1)
        s = e.replace(day=1)
    else:
        return "", ""  # 타입 일관성

    return s.isoformat(), e.isoformat()

last_final_docs = []  # 마지막 검색 결과(링크 전달용)

# “YYYY-MM-DD” 문자열로 정규화
def _normalize_date_input(x):
    if not x: return None
    try:
        if isinstance(x, str): return x[:10]
        if hasattr(x, "strftime"): return x.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None

# api의 stats.만 가져와 저장한 선수 스탯 rows → 사람이 읽기 쉬운 요약 문자열 변환

import analysis

# 사용자가 선택한 리그의 정보를 통해 경기를 필터링하고 드롭다운 생성
def ui_load_matches_selectable(date_value, use_epl: bool, use_laliga: bool, use_seriea: bool, use_bundes: bool, use_ligue1: bool):
    d = _normalize_date_input(date_value)
    if not d:
        return "<em>날짜를 선택해주세요.</em>", gr.update(choices=[], value=None), []

    try:
        # ★ API-FOOTBALL만 사용
        matches = analysis.fetch_matches_official(d, tz="Asia/Seoul")

        # ★ 체크된 리그만 남기기
        selected_ids = set()
        if use_epl:     selected_ids.add(LEAGUE_ID["EPL"])
        if use_laliga:  selected_ids.add(LEAGUE_ID["LALIGA"])
        if use_seriea:  selected_ids.add(LEAGUE_ID["SERIEA"])
        if use_bundes:  selected_ids.add(LEAGUE_ID["BUNDES"])
        if use_ligue1:  selected_ids.add(LEAGUE_ID["LIGUE1"])

        matches = _filter_by_league_ids(matches, selected_ids)

        # 표 렌더 (render_matches_html 있으면 사용)
        html_table = analysis.render_matches_html(matches, with_links=False) # with_links 없애도 될거 같은데 일단 보류

        # 드롭다운 채우기
        choices = []
        for i, m in enumerate(matches):
            mid = str(m.get("match_id") or "")
            lab = f"{i:02d}. {m.get('kickoff_local','')} | {m.get('league','')} | {m.get('home','?')} vs {m.get('away','?')} (ID:{mid})"
            choices.append((lab, mid))
        default_val = choices[0][1] if choices else None

        return html_table, gr.update(choices=choices, value=default_val), matches
    except Exception as e:
        return f"<em>경기 불러오기 오류: {e}</em>", gr.update(choices=[], value=None), []


def ui_fetch_player_stats(fixture_id: str):
    if not fixture_id:
        return "<em>경기를 먼저 선택하세요.</em>", {}

    try:
        rows, raw = analysis.fetch_player_stats_official(str(fixture_id)) # raw: api 원본 데이터
        html_table = analysis.render_player_stats_html(rows)
        return html_table, {"rows": rows}
    except Exception as e:
        return f"<em>선수 스탯 불러오기 오류: {e}</em>", {}


parser = StrOutputParser()  # (이미 있으면 재사용)

import openai

#gpt한테 시켜서 카테고리 분류
def _detect_stats_mode(q: str) -> str:
    """
    GPT를 이용해 질문을 다음 카테고리 중 하나로 분류:
    'formation' | 'player_comp' | 'tactical' | 'timeline' | 'why'
    """
    prompt = f"""
        당신은 사용자의 축구 질문을 분류하는 AI 도우미입니다.
        다음 중 하나의 카테고리로만 분류하세요:

        - formation: 포메이션, 라인업, 선발 관련
        - player_comp: 선수 비교, vs, 누가 더 잘했는지
        - tactical: 전술 관련 (프레싱, 전환, 라인 등)
        - timeline: 몇 분에 무슨 일이 있었는지 (시간 중심 질문)
        - why: 왜 졌는지, 왜 이겼는지, 원인, 분석
        - freeform: 위에 해당하지 않는 자유 질문

        **오직 아래 중 하나의 단어만 출력하세요** (다른 말 없이):  
        player_comp / tactical / timeline / why / freeform

        질문: \"{q}\"
        답변:
        """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # 또는 gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        result = response.choices[0].message.content.strip().lower()

        # 방어코드: 혹시나 GPT가 이상한 문자열을 반환할 경우
        valid = {"formation", "player_comp", "tactical", "timeline", "why", "freeform"}
        return result if result in valid else "freeform"

    except Exception as e:
        print(f"GPT 분류 오류: {e}")
        return "freeform"

#카테고리 분류에 따라 프롬프트 다르게 해서 답변 포메이션 부분은 나중에 개선, maxtoken도 나중에 생각
def _build_prompt_for_mode(
    mode: str,
    q: str,
    full_ctx_json: str,
    max_tokens: int = 2000,
    config: dict | None = None,
):
    if config is None:
        config = {}

    parser = StrOutputParser()

    model_name = config.get("llm_stats_model", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        max_tokens=int(config.get("llm_stats_max_tokens", max_tokens))
    )

    # ----------------------------------------------------
    # 1) 선수 비교 (player_comp)
    # ----------------------------------------------------
    if mode == "player_comp":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 선수 데이터를 비교 분석하는 축구 통계 전문가다.\n"
                       "반드시 JSON 데이터만 근거로 두 선수의 퍼포먼스를 비교하라.\n"
                       "허위 추정 없이 수치 기반으로만 비교하고, 데이터 부족 시 '정보 부족'이라고 밝혀라.\n"
                       "출력은 마크다운 표를 활용하라."),
            ("human", "질문: {q}\n\n---\nCONTEXT_JSON:\n{ctx_json}")
        ])
        return llm, (prompt | llm | parser)

    # ----------------------------------------------------
    # 2) 타임라인 요약 (timeline)
    # ----------------------------------------------------
    if mode == "timeline":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 축구 경기 타임라인 분석 전문가다.\n"
                       "JSON 데이터를 기반으로 골, 경고, 교체 등의 주요 이벤트를 **시간순**으로 요약하라.\n"
                       "모든 시간은 분 단위로 표시하고, 데이터가 없으면 '정보 없음'이라고 밝혀라."),
            ("human", "질문: {q}\n\n---\nCONTEXT_JSON:\n{ctx_json}")
        ])
        return llm, (prompt | llm | parser)

    # ----------------------------------------------------
    # 3) 승패 원인 분석 (why)
    # ----------------------------------------------------
    if mode == "why":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 축구 경기의 승패 원인을 분석하는 전술 코치다.\n"
                       "주어진 JSON 데이터를 바탕으로 '원인 → 영향 → 대안' 구조로 작성하라.\n"
                       "추정 없이, 수치와 통계에 기반해 설명하라."),
            ("human", "질문: {q}\n\n---\nCONTEXT_JSON:\n{ctx_json}")
        ])
        return llm, (prompt | llm | parser)

    # ----------------------------------------------------
    # 4) 전술 분석 (tactical)
    # ----------------------------------------------------
    if mode == "tactical":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 축구 전술 분석 전문가다.\n"
                       "포메이션, 라인 간격, 프레싱, 빌드업, 전환 전술 등 주요 전술 요소를 중심으로\n"
                       "JSON 데이터를 분석하라.\n"
                       "모호한 해석은 피하고 반드시 데이터 기반으로 설명하라."),
            ("human", "질문: {q}\n\n---\nCONTEXT_JSON:\n{ctx_json}")
        ])
        return llm, (prompt | llm | parser)

    # ----------------------------------------------------
    # 5) 기본 freeform 보고서
    # ----------------------------------------------------
    outline_llm = ChatOpenAI(model=model_name, temperature=0.2, max_tokens=256)
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", "주어진 질문과 JSON 데이터를 바탕으로 적절한 보고서 섹션 제목을 "
                   "3~6개 정도 JSON 배열로 제안하라.\n추측 없이 실제 가능한 내용만 작성할 것."),
        ("human", "질문: {q}\n---\nCONTEXT_JSON:\n{ctx_json}")
    ])
    try:
        raw = (outline_prompt | outline_llm | parser).invoke({"q": q, "ctx_json": full_ctx_json}).strip()
        sections = json.loads(raw)
        if not isinstance(sections, list) or not sections:
            sections = ["요약", "핵심 포인트", "세부 분석", "다음 경기 전망"]
    except Exception:
        sections = ["요약", "핵심 포인트", "세부 분석", "다음 경기 전망"]

    section_lines = "\n".join(f"- {s}" for s in sections)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 축구 분석 보고서를 작성하는 전문가다. 아래 섹션 계획에 따라 마크다운 형식으로 보고서를 작성하라.\n"
                   "각 섹션은 '## 섹션명'으로 시작하며, 데이터 기반 설명과 표/리스트를 활용하라.\n"
                   "허위 추정 금지. 데이터 부족 시 '데이터 부족'이라고 명시할 것."),
        ("human",
         "질문: {q}\n\n---\nCONTEXT_JSON:\n{ctx_json}\n\n"
         "섹션 계획:\n{sections}\n")
    ])
    chain = prompt | llm | parser
    return llm, lambda vars: chain.invoke({"q": q, "ctx_json": full_ctx_json, "sections": section_lines})


# === ADD: 풀 컨텍스트 생성 (표시 X, LLM 전용) ===
def _build_full_context_json(fixture_id: str, matches_state: list) -> str:
    import json
    # 메타 찾기: 없으면 None
    meta = None
    for m in matches_state or []:
        if str(m.get("match_id")) == str(fixture_id):
            meta = m; break

    # 필수: 선수 스탯 + 팀집계/타임라인/라인업
    try: rows, _raw_players = analysis.fetch_player_stats_official(str(fixture_id)) # raw_playres는 json 원본
    except Exception: rows = []
    try: tstats = analysis.fetch_team_match_stats_official(str(fixture_id))
    except Exception: tstats = []
    try: events = analysis.fetch_fixture_events_official(str(fixture_id))
    except Exception: events = []
    try: lineups = analysis.fetch_fixture_lineups_official(str(fixture_id))
    except Exception: lineups = []

    # 시즌/부상(메타 있으면 홈/원정 모두)
    home_season = away_season = {}
    inj_h = inj_a = []
    try:
        if meta and meta.get("league_id") and meta.get("season") and meta.get("home_id") and meta.get("away_id"):
            lid = int(meta["league_id"]); ssn = int(meta["season"])
            hid = int(meta["home_id"]);  aid = int(meta["away_id"])
            try: home_season = analysis.fetch_team_season_stats_official(hid, lid, ssn) or {}
            except Exception: pass
            try: away_season = analysis.fetch_team_season_stats_official(aid, lid, ssn) or {}
            except Exception: pass
            try: inj_h = analysis.fetch_injuries_official(hid, lid, ssn) or []
            except Exception: pass
            try: inj_a = analysis.fetch_injuries_official(aid, lid, ssn) or []
            except Exception: pass
    except Exception:
        pass

    bundle = {
        "meta": {
            "league_id": meta.get("league_id") if meta else None,
            "season": meta.get("season") if meta else None,
            "fixture_id": fixture_id,
            "kickoff_local": meta.get("kickoff_local") if meta else None,
            "home": {"id": meta.get("home_id") if meta else None, "name": meta.get("home") if meta else None},
            "away": {"id": meta.get("away_id") if meta else None, "name": meta.get("away") if meta else None},
            "status": meta.get("status") if meta else None,
        },
        "players": rows,                 # ← 선수별 stat.* 전부
        "teams_match_stats": tstats,     # ← fixtures/statistics
        "events": events,                # ← fixtures/events
        "lineups": lineups,              # ← fixtures/lineups
        "season_summaries": {"home": home_season, "away": away_season},
        "injuries": {"home": inj_h, "away": inj_a},
    }

    # 로그파일에 남김
    try:
        log_path = Path(__file__).parent / f"context_bundle_{fixture_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)
        print(f"[log] context bundle saved: {log_path}")
    except Exception as e:
        print(f"[log] failed to save context bundle: {e}")

    return json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))

# === ADD: 질문 → (전 데이터 자동수집) → LLM 답변 (UI엔 답만 보여줌) ===
def ui_analyze_match_full_auto(question: str, fixture_id: str, matches_state: list):
    # fixture_id가 비어있으면 목록의 첫 경기 자동 선택
    if not fixture_id and (matches_state or []):
        fixture_id = str((matches_state[0] or {}).get("match_id") or "")
    if not fixture_id:
        return "<em>경기를 먼저 불러오세요.</em>"

    ctx_json = _build_full_context_json(fixture_id, matches_state)

    # 컨텍스트는 '전부' 사용. 너무 클 경우를 대비해 간단 압축(문자 길이 기준)
    def _approx_tokens(s: str) -> int: return max(1, len(s)//4)
    budget = 120000  # 모델 토큰 한도에 맞춰 조절(예: 128k 모델)
    if _approx_tokens(ctx_json) > budget:
        # 큰 섹션(특히 players, events)부터 앞부분만 남김
        import json as _json
        try:
            data = _json.loads(ctx_json)
            order = ["meta","lineups","teams_match_stats","players","events","season_summaries","injuries"]
            slim = {}
            for key in order:
                slim[key] = data.get(key)
                s = _json.dumps(slim, ensure_ascii=False, separators=(",", ":"))
                if _approx_tokens(s) > budget and key in {"players","events"} and isinstance(slim[key], list):
                    slim[key] = slim[key][: max(1, len(slim[key])//2)]
            ctx_json = _json.dumps(slim, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            pass


    mode = _detect_stats_mode(question)

    # 모드별 프롬프트 생성
    llm, chain = _build_prompt_for_mode(mode, question, ctx_json, max_tokens=2000)

    try:
        if callable(chain):
            return chain({"q": question, "ctx_json": ctx_json})
        return chain.invoke({"q": question, "ctx_json": ctx_json})
    except Exception as e:
        return f"분석 중 오류: {e}"

    # LLM 체인(간단): 시스템+휴먼
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #      "너는 축구 경기 분석 전문가다. "
    #      "반드시 제공된 JSON 데이터만 사용하여 답변해야 한다. "
    #      "모든 주장과 설명은 반드시 수치와 데이터에 근거해야 한다. "
    #      "추측이나 외부 지식을 추가하지 말고, 답변은 한국어로 작성하라."),
    #     ("human", "질문: {q}\n\n---\nCONTEXT_JSON:\n{ctx_json}")
    # ])
    # chain = prompt | llm | parser
    # try:
    #     return chain.invoke({"q": question, "ctx_json": ctx_json})
    # except Exception as e:
    #     return f"분석 중 오류: {e}"

"""
여기까지 경기 분석에서 사용하는 함수
"""

try:
    # 프로젝트에 따라 위치가 다를 수 있어 두 경로 모두 시도
    from langchain_community.retrievers import BM25Retriever
except Exception:
    from langchain.retrievers import BM25Retriever  # fallback

from langchain_core.documents import Document

# (옵션) 크로스 인코더 — 없으면 자동 폴백
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

load_dotenv()
bm25_all_docs = []

#time 문자열을 datetime으로 변환
def _parse_meta_time(s):
    """metadata['time'] 문자열을 datetime으로 파싱 (여러 포맷 허용)"""
    if not s:
        return None
    s = str(s).strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",   # 2025-06-25 20:05:00
        "%Y-%m-%d",            # 2025-06-25
        "%Y/%m/%d",            # 2025/06/25
        "%Y.%m.%d",            # 2025.06.25
        "%Y-%m-%dT%H:%M:%SZ",  # 2025-06-25T20:05:00Z
        "%Y-%m-%dT%H:%M:%S%z"  # 2025-06-25T20:05:00+0000
    ]
    zfix = s.replace("Z", "+0000") if s.endswith("Z") else s
    for fmt in fmts:
        try:
            if fmt.endswith("%z"):
                return dt.strptime(zfix, fmt)
            return dt.strptime(s, fmt)
        except Exception:
            continue
    return None

# 뉴스의 날짜를 고려해 최신 뉴스가 위로 오게 재정렬 (지금은 날짜 미지정 시에만 적용)
def freshness_reorder(docs, half_life_days=10, weight=0.6, hard_days=None):
    """RRF 결과를 '최근일수록 살짝 가점'으로 재정렬. 날짜 없으면 가점 0."""
    if not docs:
        return docs
    now = dt.now()
    scored = []
    for idx, d in enumerate(docs):
        t = _parse_meta_time((d.metadata or {}).get("time"))
        if t:
            age_days = max(0.0, (now - t).total_seconds() / 86400.0) # 며칠 전 문서인지 계산 1일 = 86400초
            if hard_days is not None and age_days > float(hard_days): # 너무 오래되면 제외(현재 반영 x)
                ConnectionRefusedError
            boost = math.exp(-age_days / float(half_life_days))  # 0~1 최신 문서: 1
        else:
            boost = 0.0
        base = 1.0 / (1.0 + idx)  # 상위일수록 큼, 원래 상위에 있던 문서
        scored.append((base * (1.0 + float(weight) * boost), idx, d)) # boost와 가중치 계산
    scored.sort(key=lambda x: (-x[0], x[1])) # 점수 높은 순, 같을 경우 원래 순서
    return [d for _, _, d in scored]

# 입력된 날짜를 datetime으로 변환
def _parse_date_input(s):
    """UI 입력 YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD → datetime.date"""
    if not s:
        return None
    s = s.strip()
    m = re.match(r"^\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\s*$", s)
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    try:
        return dt(y, mth, d)
    except Exception:
        return None

# 하루의 마지막으로 설정
def _end_of_day(d):
    return d.replace(hour=23, minute=59, second=59, microsecond=999999)

# 날짜 범위의 뉴스를 필터링
def filter_docs_by_date(docs, start_dt, end_dt):
    """문서의 metadata['time']이 [start_dt, end_dt] 범위인 것만"""
    picked = []
    for d in docs or []:
        t = _parse_meta_time((d.metadata or {}).get("time"))
        if t and start_dt <= t <= end_dt:
            picked.append(d)
    return picked

# 날짜 범위의 뉴스를 앞으로 옮김 (체크 박스 안했을 때)
def prefer_date_range_first(docs, start_dt, end_dt):
    """해당 범위 문서를 앞으로(기존 상대 순서는 유지)"""
    in_range, out_range = [], []
    for idx, d in enumerate(docs or []):
        t = _parse_meta_time((d.metadata or {}).get("time"))
        (in_range if (t and start_dt <= t <= end_dt) else out_range).append((idx, d))
    in_range.sort(key=lambda x: x[0])
    out_range.sort(key=lambda x: x[0])
    return [d for _, d in (in_range + out_range)]

# -------------------------
# 설정 / 전역
# -------------------------
# LLM & 임베딩 (원래 설정 존중)
llm = ChatOpenAI(model="gpt-4o-mini")
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
persist_directory = "./news_chroma_db"

# 설정 파일 로드
try:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except Exception:
    config = {}

system_message = config.get(
    "system_message"
)
human_message_template = config.get(
    "human_message_template"
)
# 팀 이름 필터링
translation_dict = config.get("translation_dict", {})

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", human_message_template),
])
parser = StrOutputParser()

# 체인/리트리버 전역
rag_chain = None
hybrid_retriever = None
vector_retriever = None
context_chain = None

# BM25/재랭커 전역
bm25_global = None
bm25_doc_count = 0
reranker = None


# -------------------------
# 유틸
# -------------------------

# 단어 사전에 있는 팀의 한국 이름을 영어로 변경
def translate_query(query: str, dictionary: dict) -> str:
    """간단한 단어 경계 치환(사전 없으면 원문 그대로)"""
    if not query or not dictionary:
        return query
    for kor, eng in dictionary.items():
        query = query.replace(kor, eng)
    return query

# 여러 개의 리스트가 있으면 각각의 등수에 대해 계산에 더한 총 점수로 정렬
def rrf_fuse(result_lists, k=36, C=60):
    """Reciprocal Rank Fusion: 여러 리스트의 등수를 합산해 상위 k개 선택"""
    from collections import defaultdict
    scores, pick = defaultdict(float), {}
    for results in result_lists: 
        for rank, d in enumerate(results):
            key = d.page_content
            scores[key] += 1.0 / (C + rank + 1)
            pick.setdefault(key, d)
    merged = [pick[key] for key, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return merged[:k]

# 한국어 쿼리를 영어로 변환
def gpt_translate_korean_to_english(query: str, model="gpt-4o-mini") -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following Korean football question into English for use in a document search engine. Be concise."),
        ("human", "{q}")
    ])
    chain = prompt | ChatOpenAI(model=model, temperature=0) | StrOutputParser()
    return chain.invoke({"q": query})

# bm25 검색기 생성, 30개
def build_global_bm25():
    """전 코퍼스 BM25 인덱스 재생성 (ids를 include로 요청하지 않음)"""
    global bm25_global, bm25_doc_count
    db = Chroma(
        persist_directory=str((Path(__file__).resolve().parent / "news_chroma_db")),
        embedding_function=hf_embeddings,
        collection_name="news_collection",
    )
    raw = db.get(include=["documents", "metadatas"])  # ✅ ids 금지
    all_docs = [
        Document(page_content=c, metadata=m)
        for c, m in zip(raw.get("documents", []), raw.get("metadatas", []))
        if c
    ]
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = int(config.get("bm25_k", 30))  # 회수 폭 살짝 넓힘
    bm25_global = bm25
    global bm25_all_docs
    bm25_all_docs = all_docs
    # 안전한 문서 수 확인
    try:
        bm25_doc_count = db._collection.count()
    except Exception:
        bm25_doc_count = len(raw.get("documents", []))

    print(f"[bm25] rebuilt: docs={bm25_doc_count}, k={bm25_global.k}")
    return bm25_doc_count

# db문서 수가 달라지면 build_global_bm25를 호출하여 갱신
def refresh_bm25_if_stale():
    """DB 문서 수가 변하면 BM25를 자동 갱신"""
    from langchain_community.vectorstores import Chroma
    from pathlib import Path
    global bm25_doc_count, bm25_global, hf_embeddings

    try:
        db = Chroma(
            persist_directory=str((Path(__file__).resolve().parent / "news_chroma_db")),
            embedding_function=hf_embeddings,
            collection_name="news_collection",
        )
        # ✅ 절대 db.get(include=["ids"]) 쓰지 말 것
        cnt = db._collection.count()
        if cnt != bm25_doc_count or bm25_global is None:
            build_global_bm25()
    except Exception as e:
        print(f"[bm25] refresh failed: {e}")

# 크로스 인코더 설정
def init_reranker_from_config(cfg: dict):
    """크로스 인코더 초기화(옵션) — 실패해도 서비스 계속"""
    global reranker
    if not cfg.get("use_reranker", True):
        return
    if CrossEncoder is None:
        print("[reranker] sentence-transformers 미설치 → 건너뜀")
        return
    model = cfg.get("reranker_model", "BAAI/bge-reranker-base")
    max_len = int(cfg.get("reranker_max_length", 512))
    try:
        reranker = CrossEncoder(model, max_length=max_len)
        print(f"[reranker] loaded: {model}")
    except Exception as e:
        reranker = None
        print(f"[reranker] load failed: {e}")

# 크로스 인코더로 점수를 매겨 재정렬
def rerank_with_cross_encoder(query: str, docs, top_n=12, batch_size=16):
    """크로스 인코더 재랭크 (비활성/실패 시 candidates 앞에서 자름)"""
    if not docs:
        return []
    if reranker is None:
        return docs[:top_n]
    pairs = [[query, d.page_content] for d in docs]
    try:
        scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    except Exception:
        return docs[:top_n]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


# -------------------------
# 체인 구성
# -------------------------
# 초기 세팅
def create_rag_chain():
    global persist_directory, hf_embeddings, prompt_template, llm, parser
    global rag_chain, vector_retriever, context_chain, hybrid_retriever

    # 벡터 DB
    db = Chroma(
        persist_directory=str((Path(__file__).resolve().parent / "news_chroma_db")),
        embedding_function=hf_embeddings,
        collection_name="news_collection",  # 중요
    )

    # 벡터 리트리버: MMR(다양성) 또는 기본
    if config.get("use_mmr", True):
        vector_retriever = db.as_retriever(
            search_type="mmr",  # 중복 적게
            search_kwargs={
                "k": int(config.get("mmr_k", 20)),         # 최종 추출 수 (RRF 후보로)
                "fetch_k": int(config.get("mmr_fetch_k", 80)), # 80개 뽑고 그중에서 선택
                "lambda_mult": float(config.get("mmr_lambda", 0.7)), # 유사도에 가깝게
            },
        )
    else:
        vector_retriever = db.as_retriever(search_kwargs={"k": int(config.get("k", 20))})

    # 전코퍼스 BM25 구축/초기화
    build_global_bm25()

    # RAG 파이프라인
    rag_chain = {
        "context": RunnablePassthrough(),
        "input": RunnablePassthrough()
    } | prompt_template | llm | parser

    # (옵션) 재랭커 준비
    init_reranker_from_config(config)

    # 상태 메시지
    try:
        count = db._collection.count()
    except Exception:
        count = 0
    return f"준비 완료 / 문서 수: {count}"

# 최종 결과로 뽑힌 뉴스중에서 중복되지 않게 링크를 가져옴
def _build_unique_links(docs, max_items=None):
    if not docs:
        return ""
    seen, urls = set(), []
    items = docs[:max_items] if max_items else docs
    for d in items:
        m = d.metadata or {}
        url = (m.get("url") or "").strip()
        if not url:
            continue
        url2 = url
        if url2 and url2 not in seen:
            seen.add(url2)
            urls.append(url2)
    return "\n".join(urls)

# 가져온 링크를 html로 보이게 함
def _links_collapsible_html(docs, max_items_show=5):
    links_str = _build_unique_links(docs, max_items=200)
    if not links_str:
        return ""

    # ★ 여기서 반드시 리스트로 변환해야 함
    links = [u for u in links_str.split("\n") if u]

    def _a(u: str) -> str:
        # <li><a href= > 형식으로 변환
        u2 = (u or "").strip()
        p = urlparse(u2)
        shown = (p.netloc or "") + (p.path or "")
        if len(shown) > 48:
            shown = shown[:47] + "…"
        return (
            f'<li><a href="{html.escape(u2)}" '
            f'target="_blank" rel="noopener noreferrer">{html.escape(shown)}</a></li>'
        )

    head = links[:max_items_show]
    tail = links[max_items_show:]
    head_html = "".join(_a(u) for u in head)
    if not tail:
        return f"<ul>{head_html}</ul>"

    tail_html = "".join(_a(u) for u in tail)
    return (
        f"<ul>{head_html}</ul>"
        f'<details><summary>링크 {len(tail)}개 더보기</summary>'
        f"<ul>{tail_html}</ul>"
        f"</details>"
    )


try:
    from zoneinfo import ZoneInfo
    def _today_kst():
        return datetime.now(ZoneInfo("Asia/Seoul")).date()
except Exception:
    def _today_kst():  # fallback
        return datetime.utcnow().date()


def _parse_ymd(s: str | None) -> date | None:
    if not s:
        return None
    # flatpickr의 string은 보통 YYYY-MM-DD
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        # 혹시 ISO 변형 들어올 때 대비
        try:
            return datetime.fromisoformat(s).date()
        except Exception:
            return None

# 시작일 변경 시: 미래 날짜 금지(경고 + 시작일 비움)
def on_start_change(start_s: str | None, end_s: str | None):
    ds = _parse_ymd(start_s)
    today = _today_kst()
    if ds and ds > today:
        gr.Warning(f"시작일은 오늘({_today_kst().strftime('%Y-%m-%d')})을 넘길 수 없습니다. 시작일을 비웠습니다.")
        return None
    return start_s  # 정상이면 유지

# 종료일 변경 시: (1) 미래 날짜 금지 (2) 시작일보다 빠르면 금지 — 둘 다 경고 + 종료일 비움
def on_end_change(start_s: str | None, end_s: str | None):
    ds, de = _parse_ymd(start_s), _parse_ymd(end_s)
    today = _today_kst()
    if de and de > today:
        gr.Warning(f"종료일은 오늘({_today_kst().strftime('%Y-%m-%d')})을 넘길 수 없습니다. 종료일을 비웠습니다.")
        return None
    if ds and de and de < ds:
        gr.Warning("종료일이 시작일보다 빠릅니다. 종료일을 비웠습니다.")
        return None
    return end_s  # 정상이면 유지

# -------------------------
# 질의 처리
# -------------------------
def ask_question(question: str, start_date: str | None = None, end_date: str | None = None, hard_only: bool = False):

    # BM25 최신화
    refresh_bm25_if_stale()

    if vector_retriever is None or rag_chain is None:
        return ("❌ 시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.", "")

    try:
        # 0) UI 날짜 입력 파싱
        sd = _parse_date_input(start_date)
        ed = _parse_date_input(end_date)
        if sd and not ed: ed = sd
        if ed and not sd: sd = ed
        if sd and ed: ed = _end_of_day(ed)

        # 1) 질의 전처리
        q = translate_query(question, translation_dict).lower()

        # 2) 리트리벌 (벡터 + BM25)
        vector_docs = vector_retriever.invoke(q)
        bm_docs = bm25_global.invoke(gpt_translate_korean_to_english(q)) if bm25_global is not None else []

        if not vector_docs and not bm_docs:
            return ("관련 문서를 찾지 못했습니다. links.json을 업데이트하고 임베딩을 다시 생성해주세요.", "")

        # 3) (날짜 후보 보강) 날짜 지정 시, 전 코퍼스에서 해당 날짜 문서 추가
        # bm25_all_docs: 전체 문서 목록 (Chroma에 저장된 모든 문서의 내용과 메타데이터)
        result_lists = [vector_docs, bm_docs] # 검색해서 나온 결과를 합침
        if sd and ed and bm25_all_docs:
            date_docs_full = filter_docs_by_date(bm25_all_docs, sd, ed)
            if date_docs_full:
                result_lists.append(date_docs_full) # 날짜 사이에 있는 뉴스를 추가

        # 4) RRF 융합
        candidates = rrf_fuse(
            result_lists,
            k=int(config.get("rrf_candidates_k", 36)),
            C=int(config.get("rrf_C", 60)),
        )

        # 5) 날짜 필터/정렬
        if sd and ed:
            if hard_only:
                candidates = filter_docs_by_date(candidates, sd, ed)   # 하드 컷
            else:
                candidates = prefer_date_range_first(candidates, sd, ed)  # 소프트 우선
        else:
            # 날짜 미지정 시에만 '현재 기준' 신선도 적용
            if config.get("freshness_enable", True):
                candidates = freshness_reorder(
                    candidates,
                    half_life_days=int(config.get("freshness_half_life_days", 10)),
                    weight=float(config.get("freshness_weight", 0.6)),
                    hard_days=int(config.get("freshness_hard_days", 0)) or None,
                )

        # 6) (옵션) 재랭크 → 최종 N개
        final_n = int(config.get("rrf_k", 12))
        if config.get("use_reranker", True):
            reranker_query = question or q
            final_docs = rerank_with_cross_encoder(
                reranker_query,
                candidates,
                top_n=final_n,
                batch_size=int(config.get("reranker_batch_size", 16)),
            )
        else:
            final_docs = candidates[:final_n]

        # 7) 컨텍스트 → LLM
        context = "\n\n".join(d.page_content for d in final_docs)
        result = rag_chain.invoke({"context": context, "input": question})

        global last_final_docs
        last_final_docs = final_docs


        links_html = _links_collapsible_html(final_docs, max_items_show=int(config.get("links_head_show", 5)))
        return result, links_html

    except Exception as e:
        return (f"❌ 처리 중 오류 발생: {e}", "")



# === Add: background ingest scheduler ===
import os, sys, time, threading, subprocess
from pathlib import Path

LOCKFILE = Path(__file__).with_name(".ingest.lock")

# 락파일 생성
def _acquire_lock() -> bool:
    try:
        fd = os.open(LOCKFILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False

def _release_lock():
    try: os.remove(LOCKFILE)
    except FileNotFoundError: pass


INGEST_EVERY_MIN = int(os.getenv("INGEST_EVERY_MINUTES", "3"))  # 기본 30분
AUTO_INGEST = os.getenv("AUTO_INGEST", "1") != "0"               # 0이면 비활성

_ingest_lock = threading.Lock()

def ingest_once():
    if not _acquire_lock():
        print("[ingest] another run in progress; skip.")
        return
    try:
        with _ingest_lock:  # 스레드 중복 방지
            base = Path(__file__).resolve().parent
            subprocess.run([sys.executable, str(base/"news_delta_anchor.py"), "--batch","10","--max-pages","40"])
            subprocess.run([sys.executable, str(base/"crawling.py")])
            subprocess.run([sys.executable, str(base/"imbedding.py")])
            print("[ingest] done.")
    finally:
        _release_lock()

def _ingest_loop():
    """프로세스 생존 동안 주기 실행 (오버랩 방지)"""
    # 시작 직후 한 번 실행하고 주기로 반복하고 싶으면 다음 줄 주석 해제
    # ingest_once()
    while True:
        started = time.time()
        try:
            ingest_once()
        except Exception as e:
            print(f"[ingest] unexpected error: {e}")
        # 실행 시간 고려해 다음 실행까지 남은 시간만큼 대기
        sleep_s = max(INGEST_EVERY_MIN * 60 - (time.time() - started), 5)
        time.sleep(sleep_s)




# -------------------------
# Gradio UI (원형 유지)
# -------------------------
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .card { background:#fff; border:1px solid #eee; border-radius:16px; padding:14px 16px; box-shadow:0 4px 14px rgba(0,0,0,0.06); }
    .chip { display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; background:#eef6ff; border:1px solid #cfe2ff; font-weight:600; font-size:13px; }
    .fld { font-size:12px; font-weight:700; color:#222; margin-bottom:6px; letter-spacing:.2px; }
    .subtle { color:#666; font-size:12px; margin-top:6px; }
    .pillwrap { display:flex; flex-wrap:wrap; gap:10px; }
    """) as demo:
    with gr.Row():
        btn_news = gr.Button("📰 뉴스 요약", scale=1)
        btn_analysis = gr.Button("⚽ 경기 분석", scale=1)

    gr.Markdown("""
    # 📄 인공지능 축구 뉴스 챗봇
    **축구 관련 질문을 입력하면 AI가 답변을 제공합니다.
    """)

    # === 화면1: 뉴스 요약 (초기 표시) ===
    with gr.Group(visible=True) as news_group:
        with gr.Row():
            with gr.Column(scale=1):
                status_output = gr.Textbox(label="📢 상태 메세지")
                question_input = gr.Textbox(label="💬 질문 입력", placeholder="궁금한 내용을 적어주세요.")
                preset = gr.Dropdown(
                    label="빠른 선택",
                    choices=["날짜 지정 안함", "오늘", "어제", "최근 7일", "최근 30일", "이번 주", "이번 달", "지난 달"],
                    value="날짜 지정 안함"
                )
                start_date_input = Calendar(label="시작일", type="string", value=None)
                end_date_input   = Calendar(label="종료일", type="string", value=None)
                hard_only_check  = gr.Checkbox(label="날짜 범위만 보기(하드 필터)", value=False)
                submit_button    = gr.Button("🤖 답변 받기")
                answer_output    = gr.Textbox(label="📝 AI 답변")
                links_output     = gr.HTML(label="관련 링크")

                # 변경 시 검증: 반환값으로 해당 컴포넌트를 업데이트
                start_date_input.change(
                    fn=on_start_change,
                    inputs=[start_date_input, end_date_input],
                    outputs=[start_date_input],
                )
                end_date_input.change(
                    fn=on_end_change,
                    inputs=[start_date_input, end_date_input],
                    outputs=[end_date_input],
                )
    # === 화면2: 경기 분석 (초기 숨김) ===
    with gr.Group(visible=False) as analysis_group:
        gr.Markdown("### ⚽ 경기 분석")
        gr.Markdown("- 현재 검색 결과에서 수집된 링크를 기반으로 `분석.py`의 analyze()를 호출합니다.")

        with gr.Row(elem_classes="card"):
            # (A) 데이터 소스 칩
            with gr.Column(scale=1, min_width=180):
                gr.Markdown("<div class='fld'>데이터 소스</div>")
                gr.HTML("<div class='chip'>🛰️ API-FOOTBALL</div>")
                gr.Markdown("<div class='subtle'>공식 API 기반 경기/선수 데이터</div>")

            # (B) 날짜 선택
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("<div class='fld'>날짜</div>")
                match_date = Calendar(label="경기일", type="string", value="")

            # (C) 리그 필터 (5개 체크박스)
            with gr.Column(scale=3, min_width=420):
                gr.Markdown("<div class='fld'>리그 필터</div>")
                with gr.Row(elem_classes="pillwrap"):
                    cb_epl     = gr.Checkbox(label="🏴 Premier League", value=True, show_label=False)
                    cb_laliga  = gr.Checkbox(label="🇪🇸 LaLiga",          value=True, show_label=False)
                    cb_seriea  = gr.Checkbox(label="🇮🇹 Serie A",         value=True, show_label=False)
                    cb_bundes  = gr.Checkbox(label="🇩🇪 Bundesliga",      value=True, show_label=False)
                    cb_ligue1  = gr.Checkbox(label="🇫🇷 Ligue 1",         value=True, show_label=False)
                gr.Markdown("<div class='subtle'>체크 해제된 리그는 목록에서 제외됩니다.</div>")

            # (D) 실행 버튼
            with gr.Column(scale=1, min_width=160):
                gr.Markdown("<div class='fld'>&nbsp;</div>")
                load_btn = gr.Button("🔎 경기 불러오기", variant="primary")


        matches_html = gr.HTML(label="경기 목록")
        match_select = gr.Dropdown(label="경기 선택(선수 스탯은 official 권장)", choices=[], value=None, interactive=True)
        matches_state = gr.State([])          # 옵션: 필요시 확장용
        player_stats_state = gr.State({})     # {"rows": [...]}

        load_btn.click(
            fn=ui_load_matches_selectable,
            inputs=[match_date, cb_epl, cb_laliga, cb_seriea, cb_bundes, cb_ligue1],  # ★ 수정
            outputs=[matches_html, match_select, matches_state]
        )

        fetch_stats_btn = gr.Button("선수 스탯 불러오기")
        stats_html = gr.HTML(label="선수별 스탯")
        fetch_stats_btn.click(
            fn=ui_fetch_player_stats,
            inputs=[match_select],
            outputs=[stats_html, player_stats_state]
        )

        # (analysis_group 내부, 기존 구성 하단에 추가)
        question_full = gr.Textbox(label="💬 질문 (전 데이터 기반)", placeholder="예) 이 경기의 전술적 차이와 결정적 요인을 분석해줘")
        answer_full   = gr.Textbox(label="📝 LLM 답변", lines=12)
        analyze_full_btn = gr.Button("🧠 전체 데이터로 분석하기 (표시 안 함)")

        analyze_full_btn.click(
            fn=ui_analyze_match_full_auto,
            inputs=[question_full, match_select, matches_state],
            outputs=[answer_full]
        )


    # 이벤트 바인딩 (그대로 유지)
    preset.change(fn=apply_preset, inputs=preset, outputs=[start_date_input, end_date_input])
    submit_button.click(
        ask_question,
        inputs=[question_input, start_date_input, end_date_input, hard_only_check],
        outputs=[answer_output, links_output]
    )

    # 버튼으로 화면 전환
    btn_news.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        inputs=None,
        outputs=[news_group, analysis_group],
    )
    btn_analysis.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=None,
        outputs=[news_group, analysis_group],
    )

    demo.load(create_rag_chain, inputs=None, outputs=status_output)

# Gradio Blocks 정의가 모두 끝난 뒤, launch 직전에 추가
if AUTO_INGEST and not globals().get("_ingest_thread_started", False):
    threading.Thread(target=_ingest_loop, daemon=True).start()
    globals()["_ingest_thread_started"] = True
    print(f"[ingest] auto-run enabled every {INGEST_EVERY_MIN} min")


demo.launch(share=True)