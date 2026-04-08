import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. 경로 및 설정
model_path = r"C:\SmartConveyor\smart_conveyor_B_final_90.keras"  # 저장한 모델 파일명
test_csv = r"C:\SmartConveyor\data\raw\EdaTest_Clean.csv"
image_dir = r"C:\SmartConveyor\data\raw\processed_images"
target_classes = ['55', '24', '205', '197', '46', '40', '60', '240']

# 2. 모델 로드
print("🚀 모델을 불러오는 중...")
model = tf.keras.models.load_model(model_path)

# 3. 테스트 데이터 준비
print("📊 테스트 데이터 로딩 중...")
test_df = pd.read_csv(test_csv)
y_true = []
y_pred = []

# 클래스명을 인덱스로 변환하는 맵핑
class_to_idx = {name: i for i, name in enumerate(target_classes)}

for _, row in test_df.iterrows():
    img_path = os.path.join(image_dir, str(row['group']), row['name'])
    try:
        # 이미지 전처리 (학습 때와 동일하게 224x224)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # 스케일링 확인 필요 (학습시 했다면 포함)
        img_array = np.expand_dims(img_array, axis=0)

        # 예측
        pred = model.predict(img_array, verbose=0)

        y_true.append(class_to_idx[str(row['group'])])
        y_pred.append(np.argmax(pred, axis=1)[0])
    except Exception as e:
        print(f"❌ 에러 발생 ({row['name']}): {e}")

# 4. 성능 평가 시각화 (혼동행렬 & F1-score)
def plot_final_report(y_true, y_pred, class_names):
    # (1) 혼동행렬 (숫자 중심)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # (2) F1-score 리포트 (히트맵 형태)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose().iloc[:-3, :3]

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Final F1-score Report (Test Set)")
    plt.show()

    print("\n--- 상세 리포트 ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

# 실행
plot_final_report(y_true, y_pred, target_classes)