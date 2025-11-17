# # evaluate_auto.py
# import os
# from dotenv import load_dotenv
# from datasets import load_dataset
# from ragas import evaluate
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
# )
# import pandas as pd

# # ★★★ (추가) .env 파일에서 API 키를 불러와 환경변수로 설정 ★★★
# load_dotenv()

# # 1. 평가 데이터셋 로드
# dataset = load_dataset('json', data_files='ragas_dataset.jsonl', split='train')

# # 2. ground_truth가 필요 없는 지표만 정의
# metrics = [
#     faithfulness,
#     answer_relevancy,
# ]

# # 3. RAGAs 평가 실행
# # 이제 코드가 환경변수에서 API 키를 자동으로 찾아 사용합니다.
# result = evaluate(dataset, metrics=metrics)

# # 4. 결과 출력 및 저장
# df = result.to_pandas()
# print(df)
# df.to_csv("ragas_auto_evaluation_results.csv", index=False, encoding="utf-8-sig")

# print("\n✅ RAGAs 자동 평가가 완료되었습니다.")

# evaluate_auto.py
import os
from dotenv import load_dotenv
from datasets import load_dataset
from test import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
import pandas as pd
# ★★★ (추가) 평가에 사용할 언어 모델을 직접 불러옵니다 ★★★
from langchain_openai import ChatOpenAI

# .env 파일에서 API 키를 불러와 환경변수로 설정
load_dotenv()

# ★★★ (추가) 평가를 수행할 LLM을 gpt-4o로 명시적으로 지정합니다 ★★★
# 이 모델은 다국어 이해 능력이 뛰어나 한국어 평가에 적합합니다.
eval_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# 1. 평가 데이터셋 로드
dataset = load_dataset('json', data_files='ragas_dataset.jsonl', split='train')

# 2. ground_truth가 필요 없는 지표만 정의
metrics = [
    faithfulness,
    answer_relevancy,
]

# 3. RAGAs 평가 실행
# ★★★ (수정) evaluate 함수에 llm 인자를 추가하여 지정한 모델을 사용하도록 합니다 ★★★
result = evaluate(dataset, metrics=metrics, llm=eval_llm)

# 4. 결과 출력 및 저장
df = result.to_pandas()
print(df)
df.to_csv("ragas_auto_evaluation_results.csv", index=False, encoding="utf-8-sig")

print("\n✅ RAGAs 자동 평가가 완료되었습니다.")