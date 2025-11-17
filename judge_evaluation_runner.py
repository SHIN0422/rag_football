# judge_evaluation_runner.py
import pandas as pd
from evaluate_llm_as_judge import evaluate_by_judge # 함수 임포트

# 1. 평가할 데이터 로드
dataset = pd.read_json(path_or_buf='ragas_dataset.jsonl', lines=True)
results = []

# 2. 각 데이터에 대해 심판 AI 평가 실행
for index, row in dataset.iterrows():
    print(f"[{index + 1}/{len(dataset)}] 평가 중...")
    # contexts 리스트를 하나의 문자열로 합칩니다.
    context_str = "\n\n".join(row['contexts'])

    # 심판 AI에게 평가 요청
    eval_result = evaluate_by_judge(row['question'], row['answer'], context_str)

    # 원본 데이터와 평가 결과를 합쳐서 저장
    combined_result = {**row.to_dict(), **eval_result}
    results.append(combined_result)

# 3. 최종 결과를 CSV 파일로 저장
df_results = pd.DataFrame(results)
df_results.to_csv("judge_evaluation_results.csv", index=False, encoding="utf-8-sig")
print("✅ LLM-as-a-Judge 평가가 완료되었습니다.")