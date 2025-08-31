# imbedding.py
# ------------------------------------------------------------
# - articles_new.json(일반 Δ) + live_new.json(라이브 Δ) 임베딩
# - 일반: URL 단위 업서트(기존 동일 URL 삭제 후 추가)
# - 라이브: 업데이트 1건 = 1청크 기본, 길면 분할 / id = url#upd-{md5} 로 업서트
# - 완료 후 Δ 파일 비움, articles_content.json 누적 유지(일반 기사)
# ------------------------------------------------------------

import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
from pathlib import Path
import hashlib

hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
persist_directory = str((Path(__file__).resolve().parent / "news_chroma_db"))

ART_DELTA_FILE   = "articles_new.json"          # 일반 Δ 입력
LIVE_DELTA_FILE  = "live_new.json"              # 라이브 Δ 입력(이번 배치 추가분만)
CUMULATIVE_FILE  = "articles_content.json"      # 임베딩 성공 시 병합 대상(일반 기사)

# 임베딩 후 새로운 내용 누적 파일에 병합
def merge_delta_into_cumulative(delta_path: str, cumulative_path: str):
    """임베딩이 끝난 Δ(일반 기사)를 누적 파일에 병합(동일 URL은 Δ가 덮어씀)."""
    delta = []
    if os.path.exists(delta_path):
        try:
            with open(delta_path, 'r', encoding='utf-8') as f:
                delta = json.load(f)
        except Exception:
            delta = []

    if not delta:
        return

    existing = []
    if os.path.exists(cumulative_path):
        try:
            with open(cumulative_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = []

    by_url = {it.get("url"): it for it in existing if isinstance(it, dict) and it.get("url")}
    for it in delta:
        u = it.get("url")
        if u:
            by_url[u] = it

    tmp = cumulative_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(by_url.values()), f, ensure_ascii=False, indent=4)
    os.replace(tmp, cumulative_path)

def _upd_hash(u: dict) -> str:
    base = (u.get("title","") + "\n" + u.get("text","")).strip()
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:12]

def create_and_update_embeddings():
    # 1) ChromaDB 클라이언트/컬렉션
    db = Chroma(persist_directory=persist_directory,
                embedding_function=hf_embeddings,
                collection_name="news_collection")

    # 2) Δ JSON 로드
    arts = []
    lives = []
    if os.path.exists(ART_DELTA_FILE):
        try:
            with open(ART_DELTA_FILE, 'r', encoding='utf-8') as f:
                arts = json.load(f) or []
        except Exception:
            arts = []
    if os.path.exists(LIVE_DELTA_FILE):
        try:
            with open(LIVE_DELTA_FILE, 'r', encoding='utf-8') as f:
                lives = json.load(f) or []
        except Exception:
            lives = []

    if not arts and not lives:
        print("새롭게 추가할 기사가 없습니다.")
        return True

    processed_urls = set()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    added = 0

    # 3) 일반 기사 Δ (URL 업서트)
    documents_to_add = []
    ids_to_add = []
    for article in arts:
        url = article.get('url')
        if not url or url in processed_urls:
            continue
        processed_urls.add(url)

        if "content" not in article:
            continue
        text = article.get("content", "")

        metadata = {
            "url": url,
            "title": article.get('title', ''),
            "author": article.get('author', ''),
            "time": article.get('time', ''),
            "type": "article"
        }

        doc = Document(page_content=text, metadata=metadata)
        chunks = text_splitter.split_documents([doc])

        # 기존 동일 URL 문서 제거(업서트)
        try:
            db.delete(where={"url": url})
        except Exception:
            pass
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{url}-p{i:03d}"
            documents_to_add.append(chunk)
            ids_to_add.append(chunk_id)

    if documents_to_add:
        print(f"[ART] add {len(documents_to_add)} chunks")
        db.add_documents(documents=documents_to_add, ids=ids_to_add)
        added += len(documents_to_add)
    else:
        print("[ART] no chunks to add")

    # 4) 라이브 Δ (업데이트 단위 업서트)
    for live_item in lives:
        url = live_item.get("url")
        title = live_item.get("title","")
        author= live_item.get("author","")
        updates = live_item.get("live_updates") or []
        for u in updates:
            up_text = (u.get("title","") + "\n" + u.get("text","")).strip()
            if not up_text:
                continue
            up_time = u.get("time","")
            h = _upd_hash(u)
            base_id = f"{url}#upd-{h}"

            # 길면 분할, 아니면 1청크
            parts = [up_text]
            if len(up_text) > 1800:  # 문자 길이 기준 대략 분기
                parts = [c.page_content for c in text_splitter.split_documents([Document(page_content=up_text)])]

            for j, part in enumerate(parts):
                cid = base_id if len(parts) == 1 else f"{base_id}-{j:02d}"
                doc = Document(
                    page_content=part,
                    metadata={
                        "url": url,
                        "title": title,
                        "author": author,
                        "type": "live",
                        "time": up_time,          # 하드 필터 호환
                        "update_time": up_time
                    }
                )
                # 업서트(수정 대비 동일 id 삭제 후 추가)
                try:
                    db.delete(ids=[cid])
                except Exception:
                    pass
                db.add_documents([doc], ids=[cid])
                added += 1

    db.persist()
    print(f"[ingest] added={added}")

    # 5) Δ 파일 비움 + 일반 기사 누적 병합
    if os.path.exists(ART_DELTA_FILE):
        merge_delta_into_cumulative(ART_DELTA_FILE, CUMULATIVE_FILE)
        with open(ART_DELTA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)
    if os.path.exists(LIVE_DELTA_FILE):
        with open(LIVE_DELTA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)

    print(f"누적 파일 갱신 완료 → {CUMULATIVE_FILE} / Δ 초기화 완료 → {ART_DELTA_FILE}, {LIVE_DELTA_FILE}")
    return True

if __name__ == "__main__":
    ok = create_and_update_embeddings()
