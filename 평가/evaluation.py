import json
import os
import re
import pickle
import math
from collections import defaultdict
from pathlib import Path
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- LangChain 및 주요 라이브러리 Import ---
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    from langchain.retrievers import BM25Retriever

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None
    print("[Warning] sentence-transformers가 설치되지 않았습니다. Reranker가 비활성화됩니다.")

import openai

# --- 설정 및 전역 변수 초기화 ---
load_dotenv()

# LLM & 임베딩
llm = ChatOpenAI(model="gpt-4o-mini")
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 현재 파일 위치 기준으로 경로 설정
BASE_DIR = Path(__file__).resolve().parent
persist_directory = str(BASE_DIR / "news_chroma_db")
bm25_index_path = BASE_DIR / "bm25_index.pkl"

# 설정 파일 로드 (config.json)
try:
    with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except Exception:
    config = {}

# 팀 이름 변환 사전
translation_dict = config.get("translation_dict", {})

# 전역 리트리버 및 랭커 변수
vector_retriever = None
bm25_global = None
bm25_doc_count = 0
bm25_all_docs = []
reranker = None
parser = StrOutputParser()


# --- 1. rag.py의 핵심 기능 함수들 (원본 유지) ---
# (translate_query, rrf_fuse, gpt_translate_korean_to_english, 
#  build_global_bm25, init_reranker_from_config, rerank_with_cross_encoder,
#  _detect_news_category, _build_chain_for_news, create_rag_chain)
# (이전 코드와 동일한 함수들이므로 여기서는 생략합니다)

def translate_query(query: str, dictionary: dict) -> str:
    if not query or not dictionary:
        return query
    for kor, eng in dictionary.items():
        query = query.replace(kor, eng)
    return query

def rrf_fuse(result_lists, k=36, C=60):
    scores, pick = defaultdict(float), {}
    for results in result_lists: 
        for rank, d in enumerate(results):
            key = d.page_content
            scores[key] += 1.0 / (C + rank + 1)
            pick.setdefault(key, d)
    merged = [pick[key] for key, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return merged[:k]

def gpt_translate_korean_to_english(query: str, model="gpt-4o-mini") -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following Korean football question into English for use in a document search engine. Be concise."),
        ("human", "{q}")
    ])
    chain = prompt | ChatOpenAI(model=model, temperature=0) | StrOutputParser()
    return chain.invoke({"q": query})

def build_global_bm25():
    global bm25_global, bm25_doc_count, bm25_all_docs
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=hf_embeddings,
        collection_name="news_collection",
    )
    try:
        all_db_ids = set(db.get(include=[])['ids'])
    except Exception:
        all_db_ids = set()
    old_docs = []
    processed_ids = set()
    if bm25_index_path.exists():
        try:
            with open(bm25_index_path, "rb") as f:
                saved_data = pickle.load(f)
                old_docs = saved_data.get('docs', [])
                processed_ids = saved_data.get('ids', {d.metadata.get('id') for d in old_docs})
            print(f"[bm25] {len(processed_ids)}개의 기존 문서 정보를 파일에서 로드했습니다.")
        except Exception:
            print("[bm25] 인덱스 파일 로드 실패. 새로 생성합니다.")
            old_docs = []
            processed_ids = set()
    new_doc_ids = list(all_db_ids - processed_ids)
    if not new_doc_ids:
        print("[bm25] 새로운 문서가 없습니다. 인덱스가 최신 상태입니다.")
        if bm25_global is None and old_docs:
            bm25_global = BM25Retriever.from_documents(old_docs)
            bm25_global.k = int(config.get("bm25_k", 20))
            bm25_all_docs = old_docs
            bm25_doc_count = len(old_docs)
        return len(processed_ids)
    print(f"[bm25] {len(new_doc_ids)}개의 새로운 문서를 DB에서 가져옵니다.")
    new_docs_data = db.get(ids=new_doc_ids, include=["documents", "metadatas"])
    new_docs = [
        Document(page_content=c, metadata=m)
        for c, m in zip(new_docs_data.get("documents", []), new_docs_data.get("metadatas", []))
        if c
    ]
    final_docs = old_docs + new_docs
    print(f"[bm25] 총 {len(final_docs)}개 문서로 인덱스를 재생성합니다...")
    bm25 = BM25Retriever.from_documents(final_docs)
    bm25.k = int(config.get("bm25_k", 20))
    bm25_global = bm25
    bm25_all_docs = final_docs
    bm25_doc_count = len(final_docs)
    try:
        with open(bm25_index_path, "wb") as f:
            pickle.dump({'docs': final_docs, 'ids': all_db_ids}, f)
        print(f"[bm25] 최신 인덱스 정보를 파일에 저장했습니다: {bm25_index_path}")
    except Exception as e:
        print(f"[bm25] 인덱스 파일 저장 실패: {e}")
    return bm25_doc_count

def init_reranker_from_config(cfg: dict):
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

def rerank_with_cross_encoder(query: str, docs, top_n=12, batch_size=16):
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

def _detect_news_category(q: str) -> str:
    prompt = f"""
        당신은 사용자의 축구 '뉴스' 질문을 분석하여 핵심 의도를 파악하는 AI입니다.
        질문의 의도를 다음 6가지 카테고리 중 하나로만 분류하세요.
        - transfer: 이적설, 영입, 방출, 재계약 관련 질문
        - injury: 선수의 부상, 징계, 컨디션 문제 관련 질문
        - preview: 앞으로 열릴 경기에 대한 예측, 관전 포인트, 예상 라인업 관련 질문
        - review: 이미 끝난 경기의 결과, 하이라이트, 분석, 결정적 장면 관련 질문
        - performance: 특정 선수나 팀의 최근 활약상, 폼, 스탯, 평가 관련 질문
        - general: 위의 5가지에 해당하지 않는 모든 일반적인 정보 질문
        **다른 설명 없이, 아래 6개의 단어 중 하나만 출력해야 합니다.**
        transfer / injury / preview / review / performance / general
        사용자 질문: "{q}"
        분류:
        """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().lower()
        valid_categories = {"transfer", "injury", "preview", "review", "performance", "general"}
        return result if result in valid_categories else "general"
    except Exception as e:
        print(f"뉴스 유형 분류 중 오류 발생: {e}")
        return "general"

def _build_chain_for_news(category: str):
    system_message = ""
    if category == "transfer":
        system_message = (
            "당신은 이적 시장 전문가입니다. 제공된 최신 뉴스 기사들을 바탕으로, "
            "사용자가 질문한 이적설의 핵심 내용을 사실 기반으로 요약해야 합니다. "
            "특히 **'선수 이름', '관련 구단', '예상 이적료/조건', '루머의 출처나 신뢰도'**에 초점을 맞춰 답변을 구조화하세요. "
            "주어진 뉴스 본문의 내용만을 사용하여 답변을 생성해야 합니다."
        )
    elif category == "injury":
        system_message = (
            "당신은 구단의 공식 의료팀처럼 보고하는 AI입니다. 주어진 뉴스들을 근거로, "
            "사용자가 질문한 선수의 상태에 대해 명확하고 간결하게 보고해야 합니다. "
            "주어진 뉴스 본문의 내용만을 사용하여 답변을 생성해야 합니다."
            "**'선수 이름', '부상 부위 및 심각도', '예상 결장 기간 또는 복귀 시점'**을 중심으로 정리하여 답변하세요."
        )
    else: # general 및 기타
        system_message = (
            "당신은 친절한 축구 전문 AI 챗봇입니다. "
            "제공된 최신 뉴스 기사들을 바탕으로 사용자의 질문에 대해 가장 관련성 높은 정보를 찾아 명확하게 요약하여 답변해주세요. "
            "주어진 뉴스 본문의 내용만을 사용하여 답변을 생성해야 합니다."
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "아래는 질문에 답변하는 데 필요한 뉴스 기사들의 내용입니다.\n\n---\n{context}\n---\n\n이 내용을 바탕으로 다음 질문에 답변해주세요:\n{input}")
    ])
    return prompt | llm | parser

def create_rag_chain():
    global vector_retriever, bm25_global, reranker, config
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=hf_embeddings,
        collection_name="news_collection",
    )
    if config.get("use_mmr", True):
        vector_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": int(config.get("mmr_k", 20)),
                "fetch_k": int(config.get("mmr_fetch_k", 80)),
                "lambda_mult": float(config.get("mmr_lambda", 0.7)),
            },
        )
    else:
        vector_retriever = db.as_retriever(search_kwargs={"k": int(config.get("k", 20))})
    build_global_bm25()
    init_reranker_from_config(config)
    try:
        count = db._collection.count()
    except Exception:
        count = 0
    return f"준비 완료 / DB 문서 수: {count} / BM25 문서 수: {bm25_doc_count}"


# --- 2. RAGAs 평가 로직 (수정됨) ---

def load_testset(filepath="ragas_dataset.jsonl"):
    questions = []
    dataset_path = BASE_DIR / filepath
    if not dataset_path.exists():
        print(f"오류: {dataset_path} 파일을 찾을 수 없습니다.")
        return []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "reason" not in data and data.get("question"):
                    questions.append(data["question"])
            except json.JSONDecodeError:
                continue
    unique_questions = list(set(questions))
    print(f"'{filepath}'에서 {len(unique_questions)}개의 고유한 질문을 로드했습니다.")
    sample_size = min(len(unique_questions), 30)
    return unique_questions[:sample_size]

def get_rag_results(question: str, model_type: str):
    q_preprocessed = translate_query(question, translation_dict).lower()
    q_translated = gpt_translate_korean_to_english(q_preprocessed)
    vector_docs = vector_retriever.invoke(q_translated)
    bm_docs = bm25_global.invoke(q_translated) if bm25_global is not None else []
    final_docs = []
    TOP_K = int(config.get("rrf_k", 10))
    if model_type == "baseline_vector":
        final_docs = vector_docs[:TOP_K]
    elif model_type == "baseline_bm25":
        final_docs = bm_docs[:TOP_K]
    elif model_type == "hybrid_rrf":
        candidates = rrf_fuse(
            [vector_docs, bm_docs],
            k=int(config.get("rrf_candidates_k", 20)),
            C=int(config.get("rrf_C", 60)),
        )
        final_docs = candidates[:TOP_K]
    elif model_type == "final_rrf_rerank":
        candidates = rrf_fuse(
            [vector_docs, bm_docs],
            k=int(config.get("rrf_candidates_k", 20)),
            C=int(config.get("rrf_C", 60)),
        )
        final_docs = rerank_with_cross_encoder(
            question, 
            candidates,
            top_n=TOP_K,
            batch_size=int(config.get("reranker_batch_size", 16)),
        )
    context_str = "\n\n".join(d.page_content for d in final_docs)
    category = _detect_news_category(question)
    rag_chain = _build_chain_for_news(category)
    try:
        answer = rag_chain.invoke({"context": context_str, "input": question})
    except Exception as e:
        print(f"LLM 답변 생성 오류 (질문: {question[:20]}...): {e}")
        answer = "답변 생성 중 오류가 발생했습니다."
    contexts = [d.page_content for d in final_docs]
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }

def run_evaluation_custom_relevancy_only():
    """
    (★ 수정) 4개 모델 평가를 10회 반복하고,
    '매 실행마다' 중간 요약 파일('모델', '평균점수')만 저장하며
    '10회 종료 후' 최종 평균 요약 파일도 1개 저장
    """
    
    # 0. RAG 시스템 초기화 (1회 실행)
    print("RAG 시스템 초기화 시작...")
    init_status = create_rag_chain()
    print(f"RAG 시스템 초기화 완료: {init_status}")

    # 1. 평가 데이터셋 로드 (1회 실행)
    questions = load_testset("ragas_dataset.jsonl")
    if not questions:
        print("평가할 질문 데이터가 없습니다.")
        return

    models_to_test = [
        "baseline_vector",
        "baseline_bm25",
        "hybrid_rrf",
        "final_rrf_rerank"
    ]
    
    # 3. GPT 기반 'Answer Relevancy' 커스텀 평가 정의 (1회 실행)
    print("\n--- 커스텀 평가 프롬프트 정의 ---")
    
    relevancy_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert evaluator. Rate the relevance of the generated answer to the given question."),
        ("user", """
        [Question]: {question}
        [Answer]: {answer}

        Rate how relevant the [Answer] is to the [Question].
        Respond ONLY with a number between 0 and 1:
        - 0 = Completely irrelevant, or a refusal like "I don't know".
        - 1 = Perfectly relevant and directly answers the question.
        
        Output only the number.
        """)
    ])
    
    relevancy_chain = relevancy_prompt | llm | StrOutputParser()
    
    # --- (★ 수정된 평가 로직) ---

    # 4. (신규) 10회 실행 점수를 누적할 딕셔너리 초기화 (최종 요약용)
    all_run_scores = {model_name: [] for model_name in models_to_test}

    num_runs = 10
    print(f"총 {len(questions)}개의 질문으로 RAG 평가를 {num_runs}회 반복합니다.")

    # 5. (신규) 10회 반복 루프 시작
    for i in range(1, num_runs + 1):
        print(f"\n--- [ {i} / {num_runs} 번째 실행 ] ---")
        
        results_data = {}
        
        # (★ 신규) '이번 회차'의 요약 결과를 담을 리스트
        final_scores_THIS_RUN = []

        # 5-1. 모든 모델에 대해 '답변' 생성
        for model_name in models_to_test:
            print(f"  [{model_name}] 모델의 결과 생성 중 (실행 {i})")
            model_results = []
            for q in tqdm(questions, desc=f"  Processing {model_name} (Run {i})", leave=False):
                model_results.append(get_rag_results(q, model_name))
            results_data[model_name] = model_results

        # 5-2. 각 모델의 결과를 커스텀 평가
        for model_name, results_list in results_data.items():
            print(f"  [{model_name}] 모델의 커스텀 관련성(Relevancy) 점수 계산 중 (실행 {i})")
            
            relevancy_scores = []

            for item in tqdm(results_list, desc=f"  Evaluating {model_name} (Run {i})", leave=False):
                question = item["question"]
                answer = item["answer"]
                score = 0.5 # 기본값

                try:
                    score_str = relevancy_chain.invoke({
                        "question": question,
                        "answer": answer
                    }).strip()
                    score = float(re.findall(r"\d*\.?\d+", score_str)[0])
                except Exception as e:
                    print(f"    커스텀 평가 오류 (질문: {question[:10]}...): {e}")
                    score = 0.0 # 평가 실패 시 0점
                
                relevancy_scores.append(score)
                
                # (★ 삭제) '이번 회차' 상세 리스트에 추가하는 로직 제거
            
            # 5-3. (신규) '이번 회차' 평균 점수 계산 및 '최종' 요약 리스트에 누적
            mean_relevancy_this_run = np.mean(relevancy_scores)
            all_run_scores[model_name].append(mean_relevancy_this_run) # 최종 요약용
            final_scores_THIS_RUN.append({ # 이번 회차 요약용
                "Model": model_name,
                "Custom_Relevancy": mean_relevancy_this_run
            })
            print(f"  [{model_name}] (실행 {i}) 평균 점수: {mean_relevancy_this_run:.4f}")
        
        # --- (★ 신규) 1회차 실행이 끝날 때마다 중간 저장 ---
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 1. (★ 삭제) '이번 회차' 상세 파일 저장 로직 제거
        
        # 2. '이번 회차' 요약 파일 저장 (요청대로 '모델'과 '점수'만 포함)
        summary_df = pd.DataFrame(final_scores_THIS_RUN).set_index("Model")
        summary_filename = f"RUN_{i}_custom_relevancy_SUMMARY_{timestamp}.csv"
        summary_df.to_csv(summary_filename)
        print(f"\n★ [{i}회차] 요약본을 {summary_filename} 에 저장했습니다.")
        # --- (★ 중간 저장 로직 끝) ---

    # --- (★ 10회 루프 종료 후 최종 집계) ---
    print("\n\n--- 📊 10회 실행 최종 평균 계산 ---")
    
    final_average_scores = []
    
    # 6. 10회 실행의 최종 평균 계산
    for model_name, score_list in all_run_scores.items():
        final_average = np.mean(score_list)
        final_average_scores.append({
            "Model": model_name,
            "Average_of_10_Runs": final_average,
            "All_10_Scores": str(score_list) # 참고용으로 10회 점수 목록도 포함
        })

    summary_df = pd.DataFrame(final_average_scores).set_index("Model")
    
    print("\n--- 📊 10회 실행 최종 평균 요약 ---")
    print(summary_df)
    
    # 7. (신규) 10회 평균 요약본을 '하나의 파일'로 저장
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_filename = f"FINAL_10_RUN_AVERAGE_SUMMARY_{timestamp}.csv"
    
    summary_df.to_csv(summary_filename)
    print(f"\n★ 10회 실행 평균 요약본을 {summary_filename} 에 저장했습니다.")
    print("모든 평가가 완료되었습니다.")
    # --- (★ 평가 로직 종료) ---


# --- 3. 스크립트 실행 ---
if __name__ == "__main__":
    run_evaluation_custom_relevancy_only()