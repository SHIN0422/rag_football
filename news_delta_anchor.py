from __future__ import annotations
import argparse, json, os, time
from dataclasses import dataclass
from typing import List, Optional, Set
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

# Anchor-based "new links only" fetcher for SkySports Football
# Outputs:
# - links_new.json (delta only, ORDER: oldest -> newest within this batch)
# - links.json     (cumulative deduped set)
# - state_links.json {anchor_url (HEAD/NEWEST), recent_urls, last_run_at}

LISTING_URL_TPL = "https://www.skysports.com/listing/basket/11095/{page}?sortOrder=publishDate&_xhr"
HEADERS = {"User-Agent": "Mozilla/5.0"}
STATE_FILE = "state_links.json" # 가장 최근에 저장한 링크 저장
CUMULATIVE_LINKS = "links.json" # 누적 링크
DELTA_LINKS = "links_new.json" # 새로 발견한 링크
ALLOW_PREFIXES = ("/football/news", "/football/live-blog")
LIVE_ACTIVE_FILE = "live_active.json" # 라이브 업데이트 뉴스 저장용

# LinkState 객체 생성용
@dataclass
class LinkState:
    anchor_url: Optional[str] = None      # 지난번 실행에서 가장 최신 뉴스 링크
    recent_urls: List[str] = None         # 최근에 본 뉴스 URL들
    last_run_at: float = 0.0              # 마지막 실행 시각

    def to_dict(self):
        return {
            "anchor_url": self.anchor_url,
            "recent_urls": self.recent_urls or [],
            "last_run_at": self.last_run_at,
        }

    @classmethod
    def from_file(cls, path: str) -> "LinkState":
        if not os.path.exists(path):
            return cls(anchor_url=None, recent_urls=[], last_run_at=0.0)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            anchor_url=data.get("anchor_url"),
            recent_urls=data.get("recent_urls", []),
            last_run_at=data.get("last_run_at", 0.0),
        )

# 세션 객체 생성
def new_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s

# 뉴스 링크를 뽑아내서 저장
def parse_listing_items(html_fragment: str) -> List[str]:
    soup = BeautifulSoup(html_fragment, "html.parser")
    urls: List[str] = []
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if any(h.startswith(p) for p in ALLOW_PREFIXES):
            if h.startswith("/"):
                h = "https://www.skysports.com" + h
            urls.append(h)
    # de-dupe while preserving order
    seen = set(); out = []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out  # ORDER: newest -> older as they appear in listing

# 뉴스 링크 목록을 가져와 parse_listing_items에 전달
def fetch_listing_page(sess: requests.Session, page: int) -> List[str]:
    urls = []
    try:
        # XHR 방식
        url = LISTING_URL_TPL.format(page=page)
        r = sess.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        html = data.get("items", "")
        urls = parse_listing_items(html)
        print(f"[DBG] page={page}, urls={len(urls)}")
    except Exception:
        pass

    return urls

# 앵커랑 비교해서 뉴스 가져옴
def collect_until_anchor(batch: int, max_pages: int, state: LinkState) -> List[str]:
    """
    Scan from the listing HEAD (newest -> older) until:
      - we collect `batch` items, or
      - we hit the saved anchor_url (the newest from the last run).
    Returns a list in the SAME ORDER as listing (newest -> older).
    """
    sess = new_session()
    collected: List[str] = []
    found_anchor = False

    for page in range(1, max_pages + 1):
        try:
            urls = fetch_listing_page(sess, page)  # newest -> older
        except Exception as e:
            print(f"[WARN] listing fetch failed for page {page}: {e}")
            break

        if not urls:
            continue

        for u in urls:
            if state.anchor_url and u == state.anchor_url:
                found_anchor = True
                break
            if u not in collected:
                collected.append(u)
                if len(collected) >= batch:
                    break

        if len(collected) >= batch or found_anchor:
            break

    if state.anchor_url and not found_anchor and collected:
        # We didn't encounter last-run head within max_pages.
        # Still OK: we advance using what we fetched (newer items).
        print("[INFO] Anchor not found within max_pages; proceeding with newest items.")

    # LIMIT to at most `batch` (defensive)
    if len(collected) > batch:
        collected = collected[:batch]

    # IMPORTANT:
    # collected is NEWEST -> OLDER.
    # For ingestion, it's nicer to process OLDER -> NEWEST:
    # But the ANCHOR must be the NEWEST (head).
    links_for_ingest = list(reversed(collected))  # OLDEST -> NEWEST
    head_newest = collected[0] if collected else None

    return links_for_ingest, head_newest  # tuple

# 새로 가져온 뉴스 링크를 기존 링크와 합침
def update_cumulative(links_new: List[str]) -> int:
    existing: Set[str] = set()
    if os.path.exists(CUMULATIVE_LINKS):
        try:
            with open(CUMULATIVE_LINKS, "r", encoding="utf-8") as f:
                existing = set(json.load(f))
        except Exception:
            existing = set()
    merged = list(existing.union(links_new))
    with open(CUMULATIVE_LINKS, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return len(merged)


def _load_json(path, default):
    if not os.path.exists(path): return default
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return default

def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def _now_kst_str():
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(tz=ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10, help="How many new links to fetch per run")
    ap.add_argument("--max-pages", type=int, default=40, help="How deep to scan the listing to find the anchor")
    args = ap.parse_args()

    state = LinkState.from_file(STATE_FILE)

    links_new, head_newest = collect_until_anchor(args.batch, args.max_pages, state) # 새 뉴스 수집

    # Write delta (ORDER: oldest -> newest)
    with open(DELTA_LINKS, "w", encoding="utf-8") as f:
        json.dump(links_new, f, ensure_ascii=False, indent=2)

    if not links_new:
        print("새 링크 없음 → links_new.json 비워둠.")
        return

    # ★ 신규 라이브 URL을 레지스트리에 등록
    live_active = _load_json(LIVE_ACTIVE_FILE, {})
    changed = False
    for url in links_new:
        if "/football/live-blog/" in url and (url not in live_active):
            live_active[url] = {
                "url": url, "title": "", "author": "",
                "time": "", "live_updates": [],
                "status": "active", "last_ts": "", "last_keys": [],
                "last_seen_at": _now_kst_str(),
            }
            changed = True
    if changed:
        _save_json(LIVE_ACTIVE_FILE, live_active)

    # Update cumulative set
    total = update_cumulative(links_new)

    # Anchor MUST be the HEAD/NEWEST item we observed this run
    if head_newest:
        state.anchor_url = head_newest

    # Maintain a small ring of recent URLs (optional, for diagnostics/recovery)
    ring = state.recent_urls or []
    ring.extend(links_new)
    state.recent_urls = ring[-200:]
    state.last_run_at = time.time()

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"{len(links_new)}개 새 링크 → links_new.json (oldest→newest)")
    print(f"누적 링크 {total}개 → links.json")
    print(f"앵커(HEAD/NEWEST) 갱신: {state.anchor_url}")

if __name__ == "__main__":
    main()
