# evaluate_llm_as_judge.py
from langchain_openai import ChatOpenAI
import json

# OpenAI API 키 설정 필요

def evaluate_by_judge(question: str, answer: str, context: str) -> dict:
    """
    GPT-4o를 심판으로 사용하여 답변의 정확성과 완전성을 평가합니다.
    """
    prompt = f"""
    당신은 제공된 '뉴스 본문'만을 근거로 '생성된 답변'을 평가하는 엄격한 축구 사실 확인 전문가입니다.

    [사용자 질문]
    {question}

    [뉴스 본문]
    {context}

    [생성된 답변]
    {answer}

    ---
    [평가 지시]
    1. '생성된 답변'이 '뉴스 본문'에 언급된 사실과 일치하는지 평가하여 '정확성(accuracy)' 점수를 100점 만점으로 매겨주세요. 본문에 없는 내용을 조금이라도 언급했다면 점수를 크게 깎아야 합니다.
    2. '생성된 답변'이 '사용자 질문'에 대해 '뉴스 본문'이 담고 있는 핵심 정보를 충분히 포함하고 있는지 평가하여 '완전성(completeness)' 점수를 100점 만점으로 매겨주세요.
    3. 평가에 대한 근거를 '이유(reason)'로 명확하게 서술해주세요.
    4. 반드시 JSON 형식으로만 응답해주세요. (예: {{"accuracy": 100, "completeness": 90, "reason": "..."}})
    """
    try:
        judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = judge_llm.invoke(prompt)
        return json.loads(response.content)
    except Exception as e:
        return {"accuracy": 0, "completeness": 0, "reason": f"평가 중 오류 발생: {e}"}

# 이 함수를 평가 데이터셋의 각 항목에 대해 실행하면 됩니다.