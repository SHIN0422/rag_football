import pandas as pd
import numpy as np
import re
import ast
import traceback
import glob
from datetime import datetime # datetime 임포트

# 1. 파일 패턴 및 출력 파일명 정의
file_pattern = "ragas_evaluation_summary5_*.csv"
output_base_name = "combined_average_faith_answer" # 새 출력 파일 기본 이름

# 2. 수정된 파싱 함수 (오류 수정된 버전)
def robust_parse_list(s):
    """
    "np.float64()" 래퍼를 포함하거나 포함하지 않는 
    리스트의 문자열 표현을 안전하게 파싱합니다.
    """
    if pd.isna(s):
        return []
    
    s_cleaned = str(s)
    
    # "np.float64(NUMBER)"를 "NUMBER"로 변경
    # 이것이 '64'가 숫자로 잡히던 문제를 해결합니다.
    s_cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', s_cleaned)
    
    try:
        # ast.literal_eval을 사용해 안전하게 파이썬 리스트로 변환
        result = ast.literal_eval(s_cleaned)
        
        if isinstance(result, list):
            # 모든 요소를 float으로 변환
            return [float(x) for x in result]
        else:
            return []
    except (ValueError, SyntaxError, TypeError):
        return []

print(f"'{file_pattern}' 패턴의 파일 처리를 시작합니다.")

try:
    # 3. CSV 파일 검색 (glob)
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"오류: '{file_pattern}' 패턴에 맞는 파일을 찾을 수 없습니다.")
        # 파일이 없으면 여기서 중단
        raise FileNotFoundError(f"'{file_pattern}' 패턴에 맞는 파일이 없습니다.")

    print(f"총 {len(file_list)}개의 파일을 찾았습니다:")
    for f in file_list:
        print(f"- {f}")

    # 4. 모든 CSV 파일을 읽어 하나의 DataFrame으로 합치기
    all_dataframes = []
    for file in file_list:
        df_single = pd.read_csv(file)
        # 어떤 파일에서 데이터가 왔는지 추적하기 위해 'Source_File' 열 추가
        df_single['Source_File'] = file 
        all_dataframes.append(df_single)

    # 리스트에 있는 모든 DataFrame을 위아래로 합치기
    df = pd.concat(all_dataframes, ignore_index=True)

    print(f"\n총 {len(df)}개의 행으로 모든 파일이 성공적으로 통합되었습니다.")
    print("\n통합된 CSV 파일 원본 (상위 5개):")
    print(df.head())

    # 5. 'Faithfulness'와 'Answer Relevancy' 열에 *수정된* 함수 적용
    if 'Faithfulness' in df.columns and 'Answer Relevancy' in df.columns:
        df['Faithfulness_list'] = df['Faithfulness'].apply(robust_parse_list)
        print("\n'Faithfulness' 열 파싱 완료.")
        
        df['Answer_Relevancy_list'] = df['Answer Relevancy'].apply(robust_parse_list)
        print("'Answer Relevancy' 열 파싱 완료.")
    else:
        raise KeyError("필수 열('Faithfulness' 또는 'Answer Relevancy')이 CSV에 존재하지 않습니다.")

    # 6. 각각 평균 계산
    print("\n평균 계산 중...")
    df['Faithfulness_Avg'] = df['Faithfulness_list'].apply(lambda x: np.mean(x) if x else np.nan)
    df['Answer_Relevancy_Avg'] = df['Answer_Relevancy_list'].apply(lambda x: np.mean(x) if x else np.nan)
    
    print("평균 계산 완료.")
    
    # 7. 새 CSV 저장을 위한 데이터프레임 준비
    columns_to_save = ['Faithfulness_Avg', 'Answer_Relevancy_Avg']
    
    # 'Model' 열이 있는지 확인
    if 'Model' in df.columns:
        columns_to_save.insert(0, 'Model')
    
    # 'Source_File' 열 추가
    if 'Source_File' in df.columns:
         columns_to_save.append('Source_File')

    df_to_save = df[columns_to_save]

    # 8. 새 CSV 저장 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{output_base_name}_{timestamp}.csv"
    
    df_to_save.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n새 파일 '{filename}' 생성 완료.")
    print("저장된 데이터 (수정됨):")
    print(df_to_save)

except FileNotFoundError:
    print(f"오류: '{file_pattern}' 파일을 찾을 수 없습니다.")
except KeyError as e:
    print(f"오류: {e}. CSV 파일의 열 이름을 확인하세요.")
except Exception as e:
    print(f"스크립트 실행 중 예기치 않은 오류가 발생했습니다: {e}")
    traceback.print_exc()