import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

print("스크립트 실행 시작...")

try:
    # 1. 데이터 불러오기
    df = pd.read_csv("ragas_evaluation_summary5.csv")
    print(f"'{df.shape[0]}'개의 행이 있는 'ragas_evaluation_summary5.csv' 파일을 불러왔습니다.")

    # 2. 플롯을 생성할 열 목록
    columns_to_plot = ['Emb_AQ', 'Emb_AC', 'GPT_Hallucination']
    plot_files = []

    # 3. 각 열에 대해 개별 막대 그래프 생성
    for col in columns_to_plot:
        print(f"'{col}'에 대한 막대 그래프 생성 중...")
        
        # 데이터 유효성 검사 (NaN 값이 있는지 확인)
        if df[col].isnull().any():
            print(f"'{col}' 열에 NaN 값이 있습니다. NaN 값을 제외하고 그래프를 그립니다.")
            plot_data = df.dropna(subset=[col])
        else:
            plot_data = df.copy()
            
        # 값(value)을 기준으로 데이터 정렬
        # 막대 그래프에서 정렬된 순서로 표시하기 위함
        sorted_data = plot_data.sort_values(by=col)

        # 4. 막대 그래프 생성
        plt.figure(figsize=(10, 7))
        # 정렬된 데이터를 기반으로 막대 그래프 생성
        bars = plt.bar(sorted_data['Model'], sorted_data[col], color='skyblue')
        
        plt.title(f'모델별 {col} 점수', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('점수 (Score)', fontsize=12)
        plt.xticks(rotation=45, ha='right') # X축 레이블이 겹치지 않도록 회전
        plt.grid(axis='y', linestyle=':', alpha=0.7) # Y축 그리드 추가
        
        # 각 막대 위에 정확한 값 표시
        for bar in bars:
            yval = bar.get_height()
            # 소수점 4자리까지 표시
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center') 

        plt.tight_layout() # 레이아웃 최적화
        
        # 5. 플롯 저장
        filename = f"{col}_barplot.png"
        plt.savefig(filename)
        plot_files.append(filename)
        print(f"'{filename}'으로 플롯 저장 완료.")

    print("\n모든 막대 그래프 생성이 완료되었습니다.")
    print(f"생성된 파일: {', '.join(plot_files)}")

except FileNotFoundError:
    print("오류: 'ragas_evaluation_summary5.csv' 파일을 찾을 수 없습니다.")
except KeyError as e:
    # 데이터에 해당 열이 없는 경우 오류 처리
    print(f"오류: '{e}' 열을 찾을 수 없습니다. CSV 파일의 열 이름을 확인하세요.")
except Exception as e:
    print(f"스크립트 실행 중 예기치 않은 오류가 발생했습니다: {e}")