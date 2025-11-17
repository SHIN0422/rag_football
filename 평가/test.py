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
from datasets import Dataset
from ragas import evaluate
# ★ 1. context_relevancy를 import 목록에서 제외
from ragas.metrics import (
    faithfulness,
    answer_relevancy
)
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
    # langchain_community가 우선
    from langchain_community.retrievers import BM25Retriever
except Exception:
    from langchain.retrievers import BM25Retriever  # fallback

# (옵션) 크로스 인코더 — 없으면 자동 폴백
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None
    print("[Warning] sentence-transformers가 설치되지 않았습니다. Reranker가 비활성화됩니다.")

import openai

# --- 설정 및 전역 변수 초기화 ---
load_dotenv()

# LLM & 임베딩 (rag.py와 동일하게 설정)
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


# --- 1. rag.py의 핵심 기능 함수들 (복사) ---
# 이 함수들은 test.py가 평가를 수행하기 위해 필요합니다.

def translate_query(query: str, dictionary: dict) -> str:
    """간단한 단어 경계 치환(사전 없으면 원문 그대로)"""
    if not query or not dictionary:
        return query
    for kor, eng in dictionary.items():
        query = query.replace(kor, eng)
    return query

def rrf_fuse(result_lists, k=36, C=60):
    """Reciprocal Rank Fusion: 여러 리스트의 등수를 합산해 상위 k개 선택"""
    scores, pick = defaultdict(float), {}
    for results in result_lists: 
        for rank, d in enumerate(results):
            key = d.page_content
            scores[key] += 1.0 / (C + rank + 1)
            pick.setdefault(key, d)
    merged = [pick[key] for key, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return merged[:k]

def gpt_translate_korean_to_english(query: str, model="gpt-4o-mini") -> str:
    """GPT를 이용한 한->영 번역"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following Korean football question into English for use in a document search engine. Be concise."),
        ("human", "{q}")
    ])
    chain = prompt | ChatOpenAI(model=model, temperature=0) | StrOutputParser()
    return chain.invoke({"q": query})

def build_global_bm25():
    """
    (rag.py와 동일) 인덱스 파일 로드 및 DB와 비교하여 업데이트
    """
    global bm25_global, bm25_doc_count, bm25_all_docs

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=hf_embeddings,
        collection_name="news_collection",
    )

    # 1. 기존 인덱스 파일 로드
    old_docs = []
    processed_ids = set()
    if bm25_index_path.exists():
        try:
            with open(bm25_index_path, "rb") as f:
                saved_data = pickle.load(f)
                old_docs = saved_data.get('docs', [])
                processed_ids = saved_data.get('ids', {d.metadata.get('id') for d in old_docs})
            print(f"[bm25] {len(processed_ids)}개의 기존 문서 정보를 파일에서 로드했습니다.")
        except Exception as e:
            print(f"[bm25] 인덱스 파일 로드 실패: {e}. 새로 생성합니다.")
            old_docs = []
            processed_ids = set()

    # 2. ChromaDB에서 현재 ID 목록 가져오기
    try:
        all_db_ids = set(db.get(include=[])['ids'])
        if not all_db_ids:
            print("[bm25] DB에 문서가 없습니다.")
            return 0
    except Exception as e:
        print(f"[bm25] DB에서 ID를 가져오는 데 실패했습니다: {e}")
        return len(old_docs)

    # 3. 새로운 문서 ID 찾기
    new_doc_ids = list(all_db_ids - processed_ids)

    # 4. 새로운 문서가 없으면 기존 인덱스 사용
    if not new_doc_ids:
        print("[bm25] 새로운 문서가 없습니다. 인덱스가 최신 상태입니다.")
        if bm25_global is None and old_docs:
            bm25_global = BM25Retriever.from_documents(old_docs)
            bm25_global.k = int(config.get("bm25_k", 20))
            bm25_all_docs = old_docs
            bm25_doc_count = len(old_docs)
        return len(processed_ids)

    # 5. 새로운 문서만 DB에서 가져오기
    print(f"[bm25] {len(new_doc_ids)}개의 새로운 문서를 DB에서 가져옵니다.")
    new_docs_data = db.get(ids=new_doc_ids, include=["documents", "metadatas"])
    new_docs = [
        Document(page_content=c, metadata=m)
        for c, m in zip(new_docs_data.get("documents", []), new_docs_data.get("metadatas", []))
        if c
    ]
    
    # 6. 기존 + 신규 문서로 새 인덱스 생성
    final_docs = old_docs + new_docs
    print(f"[bm25] 총 {len(final_docs)}개 문서로 인덱스를 재생성합니다...")
    bm25 = BM25Retriever.from_documents(final_docs)
    bm25.k = int(config.get("bm25_k", 20))
    
    # 7. 전역 변수 업데이트 및 파일 저장
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
    """크로스 인코더 초기화 (rag.py와 동일)"""
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
    """크로스 인코더 재랭크 (rag.py와 동일)"""
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
    """(rag.py와 동일) GPT를 이용한 뉴스 질문 카테고리 분류"""
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
        # v1.x openai 구문 사용 (rag.py의 코드와 다를 수 있으나, 최신 버전 기준)
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
    """(rag.py와 동일) 카테고리별 동적 프롬프트 생성"""
    system_message = ""
    # (카테고리별 system_message 정의 - rag.py에서 복사)
    if category == "transfer":
        system_message = (
            "당신은 이적 시장 전문가입니다. 제공된 최신 뉴스 기사들을 바탕으로, "
            "사용자가 질문한 이적설의 핵심 내용을 사실 기반으로 요약해야 합니다. "
            "특히 **'선수 이름', '관련 구단', '예상 이적료/조건', '루머의 출처나 신뢰도'**에 초점을 맞춰 답변을 구조화하세요. "
            "추측은 최소화하고, 기사에 언급된 내용만으로 답변하세요."
            "주어진 뉴스 본문의 내용만을 사용하여 답변을 생성해야 합니다. 본문에 명시적으로 언급되지 않은 정보는 절대 추가하거나 추론해서는 안 됩니다."
        )
    elif category == "injury":
        system_message = (
            "당신은 구단의 공식 의료팀처럼 보고하는 AI입니다. 주어진 뉴스들을 근거로, "
            "사용자가 질문한 선수의 상태에 대해 명확하고 간결하게 보고해야 합니다. "
            "주어진 뉴스 본문의 내용만을 사용하여 답변을 생성해야 합니다. 본문에 명시적으로 언급되지 않은 정보는 절대 추가하거나 추론해서는 안 됩니다."
            "**'선수 이름', '부상 부위 및 심각도', '예상 결장 기간 또는 복귀 시점'**을 중심으로 정리하여 답변하세요."
        )
    # ... (preview, review, performance 카테고리 메시지 - 생략되었지만 동일하게 복사) ...
    else: # general 및 기타
        system_message = (
            "당신은 친절한 축구 전문 AI 챗봇입니다. "
            "제공된 최신 뉴스 기사들을 바탕으로 사용자의 질문에 대해 가장 관련성 높은 정보를 찾아 명확하게 요약하여 답변해주세요. "
            "항상 객관적인 사실에 기반하여 정보를 전달해야 합니다."
            "주어진 뉴스 본문의 내용만을 사용하여 답변을 생성해야 합니다. 본문에 명시적으로 언급되지 않은 정보는 절대 추가하거나 추론해서는 안 됩니다."
        )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "아래는 질문에 답변하는 데 필요한 뉴스 기사들의 내용입니다.\n\n---\n{context}\n---\n\n이 내용을 바탕으로 다음 질문에 답변해주세요:\n{input}")
    ])
    
    # llm과 parser는 전역 변수 사용
    return prompt | llm | parser

def create_rag_chain():
    """
    (rag.py와 동일)
    평가에 필요한 모든 전역 리트리버 (Vector, BM25, Reranker)를 초기화합니다.
    """
    global vector_retriever, bm25_global, reranker, config

    # 1. 벡터 DB 로드
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=hf_embeddings,
        collection_name="news_collection",
    )

    # 2. 벡터 리트리버 설정
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

    # 3. BM25 구축/초기화
    build_global_bm25()

    # 4. (옵션) 재랭커 준비
    init_reranker_from_config(config)

    # 5. 상태 메시지 반환
    try:
        count = db._collection.count()
    except Exception:
        count = 0
    return f"준비 완료 / DB 문서 수: {count} / BM25 문서 수: {bm25_doc_count}"


# --- 2. RAGAs 평가 로직 (기존 test.py 코드) ---

def load_testset(filepath="ragas_dataset.jsonl"):
    """'satisfied' 피드백('reason'이 없는) 질문만 로드"""
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
    
    # 중복 제거 후 30개 샘플링
    unique_questions = list(set(questions))
    print(f"'{filepath}'에서 {len(unique_questions)}개의 고유한 질문을 로드했습니다.")
    
    # ★★★ 논문을 위해 테스트셋 크기를 30개로 고정 (데이터가 적으면 그 이하)
    sample_size = min(len(unique_questions), 30)
    return unique_questions[:sample_size]

def get_rag_results(question: str, model_type: str):
    """
    지정된 model_type에 따라 검색 및 답변 생성을 수행하고,
    RAGAs 평가에 필요한 딕셔너리를 반환합니다.
    """
    
    # 1. 질의 전처리
    q_preprocessed = translate_query(question, translation_dict).lower()
    q_translated = gpt_translate_korean_to_english(q_preprocessed)
    # 2. 리트리벌 (항상 둘 다 실행)
    vector_docs = vector_retriever.invoke(q_translated)
    bm_docs = bm25_global.invoke(q_translated) if bm25_global is not None else []
    
    # 3. 모델 타입별 후보군(final_docs) 확정
    final_docs = []
    
    # RAG 논문에서 K값(검색 개수)을 통일하는 것이 중요합니다.
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
        final_docs = candidates[:TOP_K] # Reranker 없이 상위 K개
        
    elif model_type == "final_rrf_rerank":
        candidates = rrf_fuse(
            [vector_docs, bm_docs],
            k=int(config.get("rrf_candidates_k", 20)),
            C=int(config.get("rrf_C", 60)),
        )
        final_docs = rerank_with_cross_encoder(
            question, # 재랭커는 원본 한글 질문 사용
            candidates,
            top_n=TOP_K,
            batch_size=int(config.get("reranker_batch_size", 16)),
        )

    # 4. 답변 생성
    context_str = "\n\n".join(d.page_content for d in final_docs)
    
    # 카테고리 분류 및 동적 프롬프트 체인 사용
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

def run_evaluation():
    """메인 평가 함수"""
    
    # 0. RAG 시스템 초기화 (Vector, BM25, Reranker 로드)
    print("RAG 시스템 초기화 시작...")
    init_status = create_rag_chain()
    print(f"RAG 시스템 초기화 완료: {init_status}")

    # 1. 평가 데이터셋 로드
    questions = load_testset("ragas_dataset.jsonl")
    if not questions:
        print("평가할 질문 데이터가 없습니다. RAGAS_DATASET_FILE을 확인하세요.")
        return

    print(f"총 {len(questions)}개의 질문으로 RAGAs 평가를 시작합니다.")
    
    models_to_test = [
        "baseline_vector",
        "baseline_bm25",
        "hybrid_rrf",
        "final_rrf_rerank"
    ]
    
    results_data = {}

    # 2. 모든 모델에 대해 '답변' 및 '문맥' 생성 (LLM 호출)
    for model_name in models_to_test:
        print(f"\n--- [{model_name}] 모델의 결과 생성 중 ---")
        model_results = []
        # tqdm으로 진행률 표시
        for q in tqdm(questions, desc=f"Processing {model_name}"):
            model_results.append(get_rag_results(q, model_name))
        results_data[model_name] = model_results

    # 3. 각 모델의 결과를 RAGAs로 평가
    final_scores = []
    
    for model_name, results_list in results_data.items():
        print(f"\n--- [{model_name}] 모델의 RAGAs 점수 계산 중 ---")
        
        # RAGAs가 요구하는 Dataset 형식으로 변환
        eval_dataset = Dataset.from_list(results_list)
        
        # ★ 2. context_relevancy를 metrics 리스트에서 제외
        result = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy
            ],
            llm=llm
        )

        # === (신규) GPT 기반 할루시네이션 평가 추가 ===
                # === (신규) GPT 기반 할루시네이션 평가 추가 ===
        print(f"--- {model_name} GPT Hallucination 평가 중 ---")
        hallucination_scores = []
        coverage_scores = []
        consistency_scores = []
        fluency_scores = []

        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert evaluator of factual accuracy for AI-generated answers."),
            ("user", """You are given:
                [Question]: {question}
                [Answer]: {answer}
                [Context]: {context}

                Does the answer contain any hallucination or information that is **not supported** by the context?
                Respond ONLY with a number between 0 and 1:
                - 0 = Fully factual (no hallucination)
                - 1 = Completely hallucinated (entirely unsupported)
                If partially hallucinated, respond with an intermediate value like 0.3 or 0.6.
                Output only the number.""")
            ])
        hallucination_chain = hallucination_prompt | llm | StrOutputParser()

        # 🔽 여기에 추가
        coverage_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert evaluator of how well an answer uses context evidence."),
            ("user", """Question: {question}
                Answer: {answer}
                Context: {context}

                Rate how effectively the answer uses key information from the context.
                Respond ONLY with a number 0–1 (0 = not at all, 1 = fully uses context).""")
            ])
        coverage_chain = coverage_prompt | llm | StrOutputParser()

        consistency_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert evaluator of logical and factual consistency."),
            ("user", """Question: {question}
                Answer: {answer}
                Context: {context}

                Does the answer contradict itself or the provided context?
                0 = inconsistent, 1 = fully consistent.
                Respond ONLY with a number between 0 and 1.""")
            ])
        consistency_chain = consistency_prompt | llm | StrOutputParser()

        fluency_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert evaluating the fluency and readability of text."),
            ("user", """Answer: {answer}

                Rate the fluency, grammar, and clarity of this answer.
                Respond ONLY with a number between 0 and 1 (0 = poor, 1 = excellent).""")
            ])
        fluency_chain = fluency_prompt | llm | StrOutputParser()

        # === 평가 루프 ===
        for item in tqdm(results_list, desc=f"Hallucination Eval for {model_name}"):
            try:
                score_str = hallucination_chain.invoke({
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": "\n".join(item["contexts"])
                }).strip()
                score = float(re.findall(r"\d*\.?\d+", score_str)[0])
            except Exception:
                score = 0.5
            hallucination_scores.append(score)

            # --- 추가된 3가지 지표 ---
            try:
                coverage_str = coverage_chain.invoke({
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": "\n".join(item["contexts"])
                }).strip()
                coverage_score = float(re.findall(r"\d*\.?\d+", coverage_str)[0])
            except Exception:
                coverage_score = 0.5
            coverage_scores.append(coverage_score)

            try:
                consistency_str = consistency_chain.invoke({
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": "\n".join(item["contexts"])
                }).strip()
                consistency_score = float(re.findall(r"\d*\.?\d+", consistency_str)[0])
            except Exception:
                consistency_score = 0.5
            consistency_scores.append(consistency_score)

            try:
                fluency_str = fluency_chain.invoke({
                    "answer": item["answer"]
                }).strip()
                fluency_score = float(re.findall(r"\d*\.?\d+", fluency_str)[0])
            except Exception:
                fluency_score = 0.5
            fluency_scores.append(fluency_score)

        # === 평균 계산 후 저장 ===
        mean_hallucination = np.mean(hallucination_scores)
        mean_coverage = np.mean(coverage_scores)
        mean_consistency = np.mean(consistency_scores)
        mean_fluency = np.mean(fluency_scores)

        
        result_df = result.to_pandas()
        print(f"--- {model_name} 개별 결과 ---")
        print(result_df)
        
        # ★ 3. context_relevancy를 결과 저장에서 제외
        final_scores.append({
            "Model": model_name,
            "Faithfulness": result["faithfulness"],
            "Answer Relevancy": result["answer_relevancy"],
            "GPT_Hallucination": mean_hallucination,
            "Context_Coverage": mean_coverage,
            "Consistency": mean_consistency,
            "Fluency": mean_fluency
        })

    # 4. 최종 결과 테이블 출력
    summary_df = pd.DataFrame(final_scores).set_index("Model")
    print("\n\n--- 📊 최종 RAGAs 평가 요약 (논문 표 1) ---")
    print(summary_df)
    

    from datetime import datetime

    # 현재 시간 문자열 생성 (예: 2025-11-14_03-25-10)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"ragas_evaluation_summary5_{timestamp}.csv"
    # 5. 파일로 저장
    #summary_df.to_csv("ragas_evaluation_summary.csv")
    summary_df.to_csv(filename, index=False)
    print("결과를 ragas_evaluation_summary.csv 파일로 저장했습니다.")


# --- 3. 스크립트 실행 ---
if __name__ == "__main__":
    # API 키가 .env 파일에 없다면 여기서 수동 설정
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    run_evaluation()