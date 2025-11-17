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
# RAGAs 평가 지표
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

import openai

# --- 설정 및 전역 변수 초기화 ---
load_dotenv()

# LLM & 임베딩
llm = ChatOpenAI(model="gpt-4o-mini")
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 현재 파일 위치 기준으로 경로 설정
BASE_DIR = Path(__file__).resolve().parent
persist_directory = str(BASE_DIR / "news_chroma_db")

# 설정 파일 로드 (config.json)
try:
    with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except Exception:
    config = {}

# 팀 이름 변환 사전
translation_dict = config.get("translation_dict", {})

# 전역 리트리버 및 파서 변수
vector_retriever = None
parser = StrOutputParser()

# --- 1. RAG 핵심 기능 함수들 (이전과 동일) ---

def translate_query(query: str, dictionary: dict) -> str:
    """간단한 단어 경계 치환(사전 없으면 원문 그대로)"""
    if not query or not dictionary:
        return query
    for kor, eng in dictionary.items():
        query = query.replace(kor, eng)
    return query

def gpt_translate_korean_to_english(query: str, model="gpt-4o-mini") -> str:
    """GPT를 이용한 한->영 번역"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following Korean football question into English for use in a document search engine. Be concise."),
        ("human", "{q}")
    ])
    chain = prompt | ChatOpenAI(model=model, temperature=0) | StrOutputParser()
    return chain.invoke({"q": query})

def _build_general_chain():
    """
    AI 카테고리 분류 없이, 항상 일반적인 프롬프트만 사용하는 체인을 생성합니다.
    """
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
    
    return prompt | llm | parser

def create_rag_chain_simple():
    """
    평가에 필요한 벡터 리트리버만 초기화합니다.
    """
    global vector_retriever, config

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

    # 3. 상태 메시지 반환
    try:
        count = db._collection.count()
    except Exception:
        count = 0
    return f"준비 완료 / DB 문서 수: {count} (Vector Retriever만 활성화)"


# --- 2. RAGAs 평가 로직 (수정/단순화) ---

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
    
    unique_questions = list(set(questions))
    print(f"'{filepath}'에서 {len(unique_questions)}개의 고유한 질문을 로드했습니다.")
    
    sample_size = min(len(unique_questions), 30)
    return unique_questions[:sample_size]

def get_rag_results_simple(question: str):
    """
    오직 '벡터 검색 + 일반 프롬프트' RAG만 수행합니다.
    """
    
    # 1. 질의 전처리
    q_preprocessed = translate_query(question, translation_dict).lower()
    q_translated = gpt_translate_korean_to_english(q_preprocessed)
    
    # 2. 리트리벌 (Vector만)
    vector_docs = vector_retriever.invoke(q_translated)
    
    # 3. 후보군 확정
    TOP_K = int(config.get("rrf_k", 10)) 
    final_docs = vector_docs[:TOP_K]

    # 4. 답변 생성 (일반 프롬프트 체인 사용)
    context_str = "\n\n".join(d.page_content for d in final_docs)
    
    rag_chain = _build_general_chain()
    
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

def run_evaluation_simple():
    """(수정) '벡터 검색 + 일반 프롬프트' 모델만 RAGAs로 평가"""
    
    # 0. RAG 시스템 초기화 (Vector만)
    print("RAG 시스템 초기화 시작...")
    init_status = create_rag_chain_simple()
    print(f"RAG 시스템 초기화 완료: {init_status}")

    # 1. 평가 데이터셋 로드
    questions = load_testset("ragas_dataset.jsonl")
    if not questions:
        print("평가할 질문 데이터가 없습니다. RAGAS_DATASET_FILE을 확인하세요.")
        return

    print(f"총 {len(questions)}개의 질문으로 RAGAs 평가를 시작합니다.")
    
    model_name = "simple_vector_general_prompt"
    
    # 2. 모델 결과 생성 (LLM 호출)
    print(f"\n--- [{model_name}] 모델의 결과 생성 중 ---")
    model_results = []
    for q in tqdm(questions, desc=f"Processing {model_name}"):
        model_results.append(get_rag_results_simple(q))

    # 3. RAGAs로 평가
    print(f"\n--- [{model_name}] 모델의 RAGAs 점수 계산 중 ---")
    
    eval_dataset = Dataset.from_list(model_results)
    
    result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=llm
    )

    # --- (★ 수정된 부분 시작) ---
    
    # 4. 개별 상세 결과 출력
    # result.to_pandas()는 'question', 'answer', 'contexts', 'faithfulness', 'answer_relevancy' 등을 포함
    result_df = result.to_pandas()
    print(f"\n\n--- 📊 [{model_name}] 개별 상세 결과 ---")
    
    # 터미널에 출력할 때 'contexts' 열은 너무 길어서 제외하고, 주요 컬럼만 선택
    display_columns = ['question', 'answer', 'faithfulness', 'answer_relevancy']
    
    # RAGAs 버전에 따라 'contexts'가 없을 수도 있으니 확인
    all_display_columns = [col for col in display_columns if col in result_df.columns]
    
    # 보기 좋게 주요 컬럼만 출력
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', 60):
        print(result_df[all_display_columns])
    
    # 5. 평균 점수도 별도로 출력
    print("\n\n--- 📊 RAGAs 평가 요약 (평균) ---")
    print(f"Model: {model_name}")
    print(f"Faithfulness (Avg): {result['faithfulness']}")
    print(f"Answer Relevancy (Avg): {result['answer_relevancy']}")

    # 6. (수정) 파일로 저장 (평균 요약이 아닌, 개별 상세 결과를 저장)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 파일 이름 변경 (DETAILED 명시)
    filename = f"ragas_simple_vector_eval_DETAILED_{timestamp}.csv"
    
    # (수정) result_df (개별 결과)를 저장합니다.
    # CSV 파일이 엑셀에서 한글이 깨지지 않도록 'utf-8-sig' 인코딩 사용
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 개별 상세 결과({len(result_df)}개 항목)를 {filename} 파일로 저장했습니다.")
    print("엑셀에서 이 파일을 열어 'answer_relevancy'가 0.0인 항목의 'answer'를 확인해 보세요.")
    # --- (★ 수정된 부분 끝) ---


# --- 3. 스크립트 실행 ---
if __name__ == "__main__":
    run_evaluation_simple()