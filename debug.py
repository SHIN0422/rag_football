# test_rag_pipeline.py

import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from collections import defaultdict
import math
import re
from datetime import datetime as dt, timedelta
import openai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# --- rag.py에서 가져온 유틸리티 함수 및 설정 ---

# 설정 파일 로드
try:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except Exception:
    config = {}

# LLM & 임베딩 (전역)
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
persist_directory = "./news_chroma_db"

# (옵션) 크로스 인코더
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(config.get("reranker_model", "BAAI/bge-reranker-base"), 
                            max_length=int(config.get("reranker_max_length", 512)))
except Exception:
    CrossEncoder = None
    reranker = None
    print("❌ 크로스 인코더 로드 실패 또는 미설치. 재랭킹 기능을 건너뜁니다.")


# 단어 사전에 있는 팀의 한국 이름을 영어로 변경
def translate_query(query: str, dictionary: dict) -> str:
    if not query or not dictionary:
        return query
    for kor, eng in dictionary.items():
        query = query.replace(kor, eng)
    return query

# 여러 개의 리스트가 있으면 각각의 등수에 대해 계산에 더한 총 점수로 정렬 (RRF 점수 반환 추가)
def rrf_fuse(result_lists, k=36, C=60):
    scores, pick = defaultdict(float), {}
    for results in result_lists:
        for rank, d_with_score in enumerate(results):
            if isinstance(d_with_score, tuple):
                d = d_with_score[0]
            else:
                d = d_with_score
            key = d.page_content
            scores[key] += 1.0 / (C + rank + 1)
            pick.setdefault(key, d)
    merged_with_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged_docs = []
    for key, score in merged_with_scores[:k]:
        merged_docs.append((pick[key], score))
    return merged_docs

# 한국어 쿼리를 영어로 번역
def gpt_translate_korean_to_english(query: str, model="gpt-4o-mini") -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following Korean football question into English for use in a document search engine. Be concise."),
        ("human", "{q}")
    ])
    chain = prompt | ChatOpenAI(model=model, temperature=0) | StrOutputParser()
    return chain.invoke({"q": query})


# 크로스 인코더로 점수를 매겨 재정렬
def rerank_with_cross_encoder(query: str, docs, top_n=12, batch_size=16):
    if not docs or reranker is None:
        return docs[:top_n]
    
    # RRF 융합 문서에서 Document 객체만 추출
    doc_objects = [d for d, s in docs]

    pairs = [[query, d.page_content] for d in doc_objects]
    try:
        scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    except Exception:
        return doc_objects[:top_n]
    
    ranked = sorted(zip(doc_objects, scores), key=lambda x: x[1], reverse=True)
    return [(d, s) for d, s in ranked[:top_n]]

# --- RAG 파이프라인 실행 및 로그 기록 함수 ---
def run_test_and_log(question: str):
    """
    RAG 검색 및 재랭킹 파이프라인을 실행하고 결과를 로그 파일에 저장합니다.
    """
    # 로그 파일 이름에 타임스탬프 추가
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"rag_test_log_{timestamp}.json"
    
    try:
        # DB 및 리트리버 초기화
        db = Chroma(persist_directory=persist_directory, embedding_function=hf_embeddings, collection_name="news_collection")
        
        # BM25 인덱스 생성
        raw_docs = db.get(include=["documents", "metadatas"])
        all_docs = [Document(page_content=c, metadata=m) for c, m in zip(raw_docs.get("documents", []), raw_docs.get("metadatas", []))]
        bm25_global = BM25Retriever.from_documents(all_docs)
        bm25_global.k = 20
        
        # 1) 질의 전처리
        query_dict_translated = translate_query(question, config.get("translation_dict", {})).lower()
        bm25_query_eng = gpt_translate_korean_to_english(query_dict_translated)
        
        print(f"1. 원본 질문: {question}")
        print(f"2. 단어사전 변형 질문: {query_dict_translated}")
        print(f"3. BM25용 영어 질문: {bm25_query_eng}")

        # 4) 리트리벌 (벡터 + BM25)
        # 벡터 DB 검색 (점수 포함)
        vector_docs_with_scores = db.similarity_search_with_score(query_dict_translated, k=20)
        bm_docs = bm25_global.invoke(bm25_query_eng)

        # 5) RRF 융합 (점수 포함)
        candidates_with_rrf_scores = rrf_fuse([vector_docs_with_scores, bm_docs], k=36, C=60)
        
        # 6) 재랭킹
        ranked_docs_with_scores = rerank_with_cross_encoder(question, candidates_with_rrf_scores, top_n=int(config.get("rrf_k", 12)))
        
        # 로그 데이터 구성
        log_data = {
            "query_original": question,
            "query_dictionary_translated": query_dict_translated,
            "query_english_for_bm25": bm25_query_eng,
            "vector_search_results": [{"title": d.metadata.get("title", "N/A"), "url": d.metadata.get("url", "N/A"), "content_preview": d.page_content[:100], "score": float(s)} for d, s in vector_docs_with_scores],
            "bm25_search_results": [{"title": d.metadata.get("title", "N/A"), "url": d.metadata.get("url", "N/A"), "content_preview": d.page_content[:100]} for d in bm_docs],
            "rrf_fused_candidates": [{"title": d.metadata.get("title", "N/A"), "url": d.metadata.get("url", "N/A"), "content_preview": d.page_content[:100], "rrf_score": float(s)} for d, s in candidates_with_rrf_scores],
            "final_ranked_documents": []
        }
        
        for i, (doc, score) in enumerate(ranked_docs_with_scores):
            log_data["final_ranked_documents"].append({
                "rank": i + 1,
                "rerank_score": float(score),
                "title": doc.metadata.get("title", "N/A"),
                "url": doc.metadata.get("url", "N/A"),
                "time": doc.metadata.get("time", "N/A"),
                "content_preview": doc.page_content[:200]
            })

        # 로그 파일 저장
        log_path = Path(f"./debug/{log_filename}")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 테스트 완료! 결과가 '{log_filename}' 파일에 저장되었습니다.")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

# --- 실행 부분 ---
if __name__ == "__main__":
    test_question = f"맨시티 관련 소식 알려줘"
    run_test_and_log(test_question)