# crawling.py
# ------------------------------------------------------------
# - 새 링크(일반 기사) Δ 크롤링 → articles_new.json
# - 라이브(진행중) 증분 폴링 → live_new.json
# - 7일 무갱신 라이브 → live_archive.json로 이동
# - links_new.json은 성공/실패와 무관하게 비움(실패는 errlog 기록)
# - 성공 URL은 links.json(누적)에 합침
# ------------------------------------------------------------

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import os
import hashlib

# Δ 처리 전용 입력/출력
LINKS_NEW_FILE = "links_new.json"      # news_delta_anchor.py 가 만든 Δ 링크
DELTA_FILE     = "articles_new.json"   # 임베딩 입력(이번 런의 새 콘텐츠: 일반 기사만)
LINKS_FILE     = "links.json"          # 누적 링크(성공 URL 추가)
ERRLOG_FILE    = "crawl_errlog.txt"    # 실패 URL과 에러 원인 기록

# 라이브 전용 관리 파일
LIVE_ACTIVE_FILE  = "live_active.json"   # 진행중 라이브 누적/상태
LIVE_DELTA_FILE   = "live_new.json"      # 이번 배치 라이브 Δ (업데이트들만)
LIVE_ARCHIVE_FILE = "live_archive.json"  # 7일↑ 무갱신 라이브

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------- 유틸 ----------
def now_kst_iso() -> str:
    return datetime.now(tz=ZoneInfo("Asia/Seoul")).isoformat()

def now_kst_str() -> str:
    return datetime.now(tz=ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

def to_kst_dt(s: str):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Seoul"))
    except Exception:
        return None

def load_links(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [u for u in data if isinstance(u, str)]
    except Exception:
        return []

def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def log_errors(errs):
    """errs: list[tuple(url, reason)] → ERRLOG_FILE에 append"""
    if not errs:
        return
    with open(ERRLOG_FILE, "a", encoding="utf-8") as f:
        for url, reason in errs:
            f.write(f"[{now_kst_iso()}] {url} :: {reason}\n")

# ---------- 개별 추출 ----------
def extract_skysports_article_content(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        return None, f"request exception: {e}"
    if res.status_code != 200:
        return None, f"http {res.status_code}"

    soup = BeautifulSoup(res.text, 'html.parser')

    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "제목 없음"

    content_div = soup.find('div', class_='sdc-article-body')
    paragraphs = []
    if content_div:
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in content_div.find_all('p')
            if not p.has_attr("class")
        ]
    content = "\n".join(paragraphs)

    date_tag = soup.find('p', class_='sdc-article-date__date-time')
    if date_tag:
        try:
            date_text = date_tag.get_text(strip=True).replace(', UK', '')
            dt = datetime.strptime(date_text, '%A %d %B %Y %H:%M')
            dt_uk = dt.replace(tzinfo=ZoneInfo("Europe/London"))
            dt_kst = dt_uk.astimezone(ZoneInfo("Asia/Seoul"))
            time_str = dt_kst.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            time_str = ""
    else:
        time_str = ""

    author_div = soup.find('div', class_='sdc-article-author')
    author = author_div.get_text(" ", strip=True) if author_div else "없음"

    item = {
        "url": url,
        "title": title,
        "content": content,
        "author": author,
        "time": time_str
    }
    return item, None

def extract_skysports_article_live_content(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        return None, f"request exception: {e}"
    if res.status_code != 200:
        return None, f"http {res.status_code} (page)"

    soup = BeautifulSoup(res.text, 'html.parser')
    channel_div = soup.find('div', {'data-norkon-channel-id': True})
    if not channel_div:
        return None, "norkon channel id not found"

    channel_id = channel_div['data-norkon-channel-id']
    api_url = f'https://livecentercdn.norkon.net/BulletinFeed/sky-sports-prod/{channel_id}/Initial/'
    try:
        res_api = requests.get(api_url, headers=HEADERS, timeout=15)
    except Exception as e:
        return None, f"norkon request exception: {e}"
    if res_api.status_code != 200:
        return None, f"norkon http {res_api.status_code}"

    try:
        data = json.loads(res_api.content.decode('utf-8-sig'))
    except json.JSONDecodeError:
        return None, "norkon json decode error"

    posts = data.get('result', {}).get('addedOrChanged', [])
    live_content = []
    for post in posts:
        html_content = (post.get('content') or {}).get('html')
        if not html_content:
            continue
        text = BeautifulSoup(html_content, 'html.parser').get_text(separator='\n', strip=True)
        if not text:
            continue

        timestamp = post.get('updated') or post.get('created')
        try:
            dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            dt_kst = dt_utc.astimezone(timezone(timedelta(hours=9)))
            time_str = dt_kst.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            time_str = ""

        # 각 업데이트 항목에 원문 페이지 url 주입
        live_content.append({
            "url": url,
            "title": post.get("title", ""),
            "author": post.get("authorName", ""),
            "time": time_str,
            "text": text
        })

    item = {"url": url, "live_updates": live_content}
    return item, None

# ---------- 라이브 증분/아카이브 ----------
def _upd_key(u: dict) -> str:
    base = (u.get("title","") + "\n" + u.get("text","")).strip()
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def diff_live_updates(updates, anchor):
    """
    anchor: {"last_ts": "YYYY-MM-DD HH:MM:SS", "last_keys": [...]}
    updates: [{title, time, text, ...}]
    return: (picked_updates, new_anchor)
    """
    updates = [u for u in (updates or []) if isinstance(u, dict) and u.get("time")]
    updates.sort(key=lambda u: u["time"])  # 문자열 포맷이라 정렬 안전

    last_ts = (anchor or {}).get("last_ts") or "" # 지난번 기준 시각
    last_dt = to_kst_dt(last_ts)
    last_keys = set((anchor or {}).get("last_keys") or []) # 그 시각에 이미 본 업데이트들의 _upd_key 해시

    picked = []
    keys_at_last = set(last_keys)

    for u in updates:
        dt = to_kst_dt(u["time"])
        if not dt:
            continue
        k = _upd_key(u)
        if (last_dt is None) or (dt > last_dt):
            picked.append(u) # 지난 시각보다 “더 최신”
        elif dt == last_dt and k not in last_keys:
            picked.append(u) # 같은 시각이지만 새로 추가된 항목
            keys_at_last.add(k)

    if picked:
        newest_ts = max(p["time"] for p in picked)
        newest_dt = to_kst_dt(newest_ts)
        if (last_dt is None) or (newest_dt > last_dt): # 지난번보다 "더 최신" 업데이트가 있었다
            new_keys = {_upd_key(u) for u in picked if u["time"] == newest_ts}
            anchor2 = {"last_ts": newest_ts, "last_keys": list(new_keys)}
        else: # 최신 시각은 지난번과 같았고, 그 시각에 새 글만 추가됨
            anchor2 = {"last_ts": last_ts, "last_keys": list(keys_at_last)}
    else:
        anchor2 = {"last_ts": last_ts, "last_keys": list(last_keys)}

    return picked, anchor2

def load_live_maps():
    active = load_json(LIVE_ACTIVE_FILE, {})
    archive = load_json(LIVE_ARCHIVE_FILE, {})
    return active, archive

def save_live_maps(active, archive):
    save_json(LIVE_ACTIVE_FILE, active)
    save_json(LIVE_ARCHIVE_FILE, archive)

def process_live(active, archive):
    """
    - active의 모든 라이브 URL을 폴링하여 Δ 업데이트만 선별
    - 7일 무갱신이면 archive로 이동
    """
    delta_items = []

    # active 폴링
    for url, rec in list(active.items()):
        if rec.get("status","active") != "active":
            continue
        try:
            item, err = extract_skysports_article_live_content(url)
        except Exception as e:
            item, err = None, f"unexpected: {e}"
        if err or not item or not item.get("live_updates"):
            rec["last_seen_at"] = now_kst_str()
        else:
            picked, anchor2 = diff_live_updates(
                item["live_updates"],
                {"last_ts": rec.get("last_ts",""), "last_keys": rec.get("last_keys", [])}
            )
            if picked:
                rec["title"] = item.get("title","") or rec.get("title","")
                rec["author"]= item.get("author","") or rec.get("author","")
                rec["live_updates"].extend(picked)
                rec["time"] = anchor2["last_ts"]
                rec["last_ts"] = anchor2["last_ts"]
                rec["last_keys"] = anchor2["last_keys"]
                rec["last_seen_at"] = now_kst_str()

                # 이번 배치 Δ (임베딩 입력)
                delta_items.append({
                    "url": url,
                    "title": rec["title"],
                    "author": rec["author"],
                    "type": "live",
                    "time": anchor2["last_ts"],       # 하드필터 호환
                    "live_updates": picked
                })
            else:
                rec["last_seen_at"] = now_kst_str() # 오래된 뉴스 비교 위해 사용

        # 7일 무갱신 → archive
        last_dt = to_kst_dt(rec.get("last_ts",""))
        if last_dt and (datetime.now(tz=ZoneInfo("Asia/Seoul")) - last_dt).days >= 7:
            rec["status"] = "ended"
            archive[url] = rec
            active.pop(url, None)

    return delta_items

# ---------- 메인 ----------
def main():
    # links_new 가 없거나 비어도 → 라이브 폴링은 항상 수행해야 함
    links_new = []
    if os.path.exists(LINKS_NEW_FILE):
        try:
            with open(LINKS_NEW_FILE, "r", encoding="utf-8") as f:
                links_new = json.load(f) or []
        except Exception:
            links_new = []

    out = []
    success_urls = []
    errors = []  # (url, reason)

    # A) 일반 기사 Δ (라이브 URL은 여기서 제외)
    for url in links_new:
        try:
            if "football/live-blog" in url:
                # 라이브는 process_live()에서 증분 처리
                success_urls.append(url)  # 등록 성공으로 간주
                continue
            item, err = extract_skysports_article_content(url)
        except Exception as e:
            item, err = None, f"unexpected exception: {e}"

        if item is not None and err is None:
            out.append(item)
            success_urls.append(url)
        else:
            errors.append((url, err or "unknown error"))

    # Δ 저장(일반 기사만)
    with open(DELTA_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 실패 링크 포함 links_new.json은 전부 비움
    with open(LINKS_NEW_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    # 에러 로그 남기기
    log_errors(errors)

    # B) 라이브 증분 + 7일 아카이브
    active, archive = load_live_maps()
    live_delta_items = process_live(active, archive)
    save_json(LIVE_DELTA_FILE, live_delta_items)
    save_live_maps(active, archive)

    print(f"크롤 완료: 일반 성공 {len([u for u in success_urls if 'live-blog' not in u])} / 실패 {len(errors)} → {DELTA_FILE}")
    print(f"[LIVE] Δ={len(live_delta_items)} 저장 → {LIVE_DELTA_FILE} | active={len(active)}, archive={len(archive)}")

if __name__ == "__main__":
    main()
