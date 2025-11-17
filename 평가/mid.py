import pandas as pd
import numpy as np
import re
import ast
import traceback
import glob
from datetime import datetime

# --- 파싱 함수 (Answer Relevancy용) ---
def robust_parse_list(s):
    """
    "np.float64()" 래퍼가 포함된 리스트 문자열을 안전하게 파싱합니다.
    (0을 포함한 모든 숫자를 올바르게 추출합니다.)
    """
    if pd.isna(s):
        return []
    
    s_cleaned = str(s)
    
    # "np.float64(NUMBER)"를 "NUMBER"로 변경
    s_cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', s_cleaned)
    
    try:
        # ast.literal_eval을 사용해 안전하게 파이썬 리스트로 변환
        result = ast.literal_eval(s_cleaned)
        
        if isinstance(result, list):
            return [float(x) for x in result]
        else:
            return []
    except (ValueError, SyntaxError, TypeError):
        return []
# ---

# 1. 파일 패턴 및 출력 파일명 정의
file_pattern = "ragas_evaluation_summary5_*.csv"
output_base_name = "answer_relevancy_stats" # 새 출력 파일 기본 이름 (중간값 + 0 개수)

print(f"'{file_pattern}' 패턴의 파일 처리를 시작합니다.")

try:
    # 2. CSV 파일 검색 (glob)
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"오류: '{file_pattern}' 패턴에 맞는 파일을 찾을 수 없습니다.")
        raise FileNotFoundError(f"'{file_pattern}' 패턴에 맞는 파일이 없습니다.")

    print(f"총 {len(file_list)}개의 파일을 찾았습니다:")
    for f in file_list:
        print(f"- {f}")

    # 3. 모든 CSV 파일을 읽어 하나의 DataFrame으로 합치기
    all_dataframes = []
    for file in file_list:
        df_single = pd.read_csv(file)
        df_single['Source_File'] = file 
        all_dataframes.append(df_single)

    df = pd.concat(all_dataframes, ignore_index=True)

    print(f"\n총 {len(df)}개의 행으로 모든 파일이 성공적으로 통합되었습니다.")

    # 4. 'Answer Relevancy' 파싱 및 통계 계산
    if 'Answer Relevancy' in df.columns:
        print("\n'Answer Relevancy' 파싱 중...")
        df['Answer_Relevancy_list'] = df['Answer Relevancy'].apply(robust_parse_list)
        
        # 중간값(Median) 계산
        df['Answer_Relevancy_Median'] = df['Answer_Relevancy_list'].apply(lambda x: np.median(x) if x else np.nan)
        print("'Answer Relevancy' 중간값(Median) 계산 완료.")
        
        # *** [새로 추가된 부분] 0의 개수 계산 ***
        # 리스트(x)에서 0.0 (float)의 개수를 셉니다.
        df['Answer_Relevancy_Zero_Count'] = df['Answer_Relevancy_list'].apply(lambda x: x.count(0.0) if x else 0)
        print("'Answer Relevancy' 0 개수 계산 완료.")
    
    else:
        print("경고: 'Answer Relevancy' 열을 찾을 수 없습니다.")


    # 5. 새 CSV 저장을 위한 최종 열 목록 준비
    
    columns_to_save = []
    
    if 'Model' in df.columns:
        columns_to_save.append('Model')
    
    if 'Answer_Relevancy_Median' in df.columns:
        columns_to_save.append('Answer_Relevancy_Median')
    
    # *** [새로 추가된 부분] 0 개수 열을 저장 목록에 추가 ***
    if 'Answer_Relevancy_Zero_Count' in df.columns:
        columns_to_save.append('Answer_Relevancy_Zero_Count')
    
    if 'Source_File' in df.columns:
         columns_to_save.append('Source_File')

    # 6. 최종 DataFrame 생성 및 저장
    df_to_save = df[columns_to_save]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{output_base_name}_{timestamp}.csv"
    
    df_to_save.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n새 파일 '{filename}' 생성 완료.")
    print("저장된 데이터 (상위 5개):")
    print(df_to_save.head())

except FileNotFoundError as e:
    print(f"오류: {e}")
except KeyError as e:
    print(f"오류: {e}. CSV 파일의 열 이름을 확인하세요.")
except Exception as e:
    print(f"스크립트 실행 중 예기치 않은 오류가 발생했습니다: {e}")
    traceback.print_exc()