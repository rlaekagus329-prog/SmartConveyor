import os
import pandas as pd
from sklearn.model_selection import train_test_split


def refresh_csv_from_cleaned_data(processed_root, output_dir):
    all_data = []
    # 실제 폴더를 뒤져서 살아남은 파일들만 수집
    for category in os.listdir(processed_root):
        cat_path = os.path.join(processed_root, category)
        if os.path.isdir(cat_path):
            files = os.listdir(cat_path)
            for f in files:
                all_data.append({'name': f, 'group': category})

    df = pd.DataFrame(all_data)
    # 다시 8:2로 분할
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['group'])

    train_df.to_csv(os.path.join(output_dir, "EdaTrain_Clean.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "EdaTest_Clean.csv"), index=False)
    print(f"✅ 정제된 데이터 기반 CSV 갱신 완료! 총 {len(df)}장")

# 경로 설정 후 실행
refresh_csv_from_cleaned_data(r"C:\SmartConveyor\data\raw\processed_images", r"C:\SmartConveyor\data\raw")