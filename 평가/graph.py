import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import re
import traceback

def clean_and_parse_list(s):
    """
    "np.float64()" 래퍼를 포함할 수 있는 리스트의 문자열 표현을 파싱합니다.
    """
    if not isinstance(s, str):
        return np.nan
    
    # np.float64() 래퍼를 제거하여 표준 리스트 문자열로 정리
    # 정규 표현식을 사용하여 더 강력하게 제거
    s_cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', s)
    
    try:
        # 정리된 문자열 평가
        result = ast.literal_eval(s_cleaned)
        if isinstance(result, list):
            # 모든 요소를 float으로 변환
            return [float(x) for x in result]
        else:
            # 리스트가 아닌 경우 (예: 단일 숫자) NaN 반환
            return np.nan
    except (ValueError, SyntaxError, TypeError):
        # 파싱 실패 시 NaN 반환
        return np.nan

print("스크립트 실행 시작...")

try:
    # 1. 데이터 불러오기
    df = pd.read_csv("ragas_evaluation_summary5.csv")
    print(f"'{df.shape[0]}'개의 행이 있는 'ragas_evaluation_summary5.csv' 파일을 불러왔습니다.")

    # 2. 데이터 파싱
    # 'Faithfulness'와 'Answer Relevancy' 열의 문자열을 숫자 리스트로 변환
    df['Faithfulness_list'] = df['Faithfulness'].apply(clean_and_parse_list)
    df['Answer_Relevancy_list'] = df['Answer Relevancy'].apply(clean_and_parse_list)
    print("'Faithfulness' 및 'Answer Relevancy' 열 파싱 완료.")

    # 3. 고유 모델 목록 가져오기
    models = df['Model'].unique()
    print(f"분석할 모델: {', '.join(models)}")

    plot_files = []

    # 4. 각 모델에 대해 별도의 플롯 생성
    for model in models:
        print(f"'{model}' 모델에 대한 플롯 생성 중...")
        
        try:
            # 해당 모델의 데이터 행 추출 (iloc[0] 사용)
            model_data_row = df[df['Model'] == model].iloc[0]
            
            # 파싱된 점수 리스트 가져오기
            faith_scores = model_data_row['Faithfulness_list']
            ans_rel_scores = model_data_row['Answer_Relevancy_list']

            # 데이터 유효성 검사 (리스트가 아니거나 비어있는 경우 건너뛰기)
            if not isinstance(faith_scores, list) or not isinstance(ans_rel_scores, list):
                print(f"'{model}' 모델에 대한 유효한 점수 리스트를 찾을 수 없습니다. 이 모델은 건너뜁니다.")
                continue

            if not faith_scores or not ans_rel_scores:
                print(f"'{model}' 모델에 점수 데이터가 비어있습니다. 이 모델은 건너뜁니다.")
                continue

            # X축 (단계) 생성
            steps_faith = range(len(faith_scores))
            steps_ans_rel = range(len(ans_rel_scores))

            # 5. 플롯 생성
            plt.figure(figsize=(12, 7))
            
            # Faithfulness 점선 그래프 (파란색, 'o' 마커)
            plt.plot(steps_faith, faith_scores, marker='o', linestyle='--', label='Faithfulness')
            
            # Answer Relevancy 점선 그래프 (주황색, 's' 마커)
            plt.plot(steps_ans_rel, ans_rel_scores, marker='s', linestyle='--', label='Answer Relevancy')
            
            plt.title(f'모델별 점수: {model}', fontsize=16)
            plt.xlabel('평가 단계 (Evaluation Step)', fontsize=12)
            plt.ylabel('점수 (Score)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.7) # 그리드 추가
            plt.tight_layout() # 레이아웃 최적화
            
            # 6. 플롯 저장
            # 파일 이름 생성 (모델 이름을 기반으로)
            safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '', model).replace('_', '-')
            filename = f"{safe_filename}_scores_plot.png"
            plt.savefig(filename)
            plot_files.append(filename)
            print(f"'{filename}'으로 플롯 저장 완료.")

        except Exception as e:
            # 특정 모델 처리 중 오류 발생 시 로깅
            print(f"'{model}' 모델 플롯 생성 중 오류 발생: {e}")
            traceback.print_exc()

    print("\n모든 플롯 생성이 완료되었습니다.")
    print(f"생성된 파일: {', '.join(plot_files)}")

except FileNotFoundError:
    print("오류: 'ragas_evaluation_summary5.csv' 파일을 찾을 수 없습니다.")
except Exception as e:
    # 스크립트 전반적인 오류 발생 시 로깅
    print(f"스크립트 실행 중 예기치 않은 오류가 발생했습니다: {e}")
    traceback.print_exc()