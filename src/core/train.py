import shutil
import os
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


data_dir = r"C:\SmartConveyor\data\raw\processed_images"
csv_path = r"C:\SmartConveyor\data\raw\EdaTrain_Clean.csv"

target_classes = ['55', '24', '205', '197', '46', '40', '60', '240']

# 3. 데이터 로드
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    class_names=target_classes,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# 3-2. 검증 데이터 로드
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    class_names=target_classes,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# 최적화(4번)를 하기 전에 클래스 개수를 먼저 저장하세요!
num_classes = len(train_ds.class_names)
# 나중에 평가할 때 쓰기 위해 이름을 따로 보관해두는 것이 좋습니다.
target_classes = train_ds.class_names #실제 데이터셋 객체가 들고 있는 진짜 이름표를 변수에 저장

# 4. 성능 최적화 (이게 핵심!)
# 셔플링과 프리페치(미리 읽어오기)를 통해 GPU가 쉬지 않게 합니다.
AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) - 램딸려서 캐쉬 못씀
train_ds = train_ds.shuffle(100).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)



# 5. 모델 빌드 (클래스 개수는 데이터셋에 맞게 자동 추출)
model = build_smart_conveyor_model(num_classes)

# 6. 조기 종료 설정 (patience=5)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# 6. 고점에 다다랐을 때 학습률을 낮춰서 정교하게 학습 (65% 벽 뚫기용)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001,verbose=1)


######### 평가
def plot_learning_curves(history):
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

def evaluate_model_performance(model, val_ds, class_names):
    print("\n[시스템] 상세 성능 평가 시작...")
    y_true, y_pred = [], []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.show()

    return y_true, y_pred

# f1-score
def plot_f1_score_report(y_true, y_pred, class_names):
    # 1. 리포트를 딕셔너리 형태로 받아서 DataFrame으로 변환
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # 2. 'accuracy', 'macro avg', 'weighted avg' 행은 제외하고 카테고리별 점수만 필터링
    # f1-score 열만 선택해서 시각화합니다.
    df_f1 = df_report.iloc[:-3, :3] # 마지막 3행 제외, f1-score 포함 앞 3열만

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_f1, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Category-wise Precision, Recall, and F1-score")
    plt.show()

# --- [2] 메인 학습 루프 ---
if __name__ == "__main__":
    # 데이터 경로 설정 (코랩 환경에 맞춰 수정 필요)
    # data_dir = "/content/drive/MyDrive/SmartConveyor/data"

    # ... (데이터 로드 및 모델 빌드 부분 생략) ...

    # 학습 실행 1단계
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stop, reduce_lr]
    )
    # 학습 실행 2단계 봉인헤제
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True
            print(f"{layer.name}의 봉인이 해제되었습니다.")

    # 2. 아주 낮은 학습률로 다시 컴파일 (중요!)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. 추가 학습 진행 (Fine-tuning)
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20, # 전체 에포크를 늘려줍니다
        initial_epoch=history.epoch[-1], # 1단계가 끝난 지점부터 시작
        callbacks=[early_stop, reduce_lr]
    )

    # 4. 모델 저장
    model_name = 'smart_conveyor_B_final_90.keras'
    model.save(model_name)

    # --- [3] 결과 출력 (여기서 한 번만 호출!) ---
    plot_learning_curves(history)
    y_true, y_pred = evaluate_model_performance(model, val_ds, target_classes)
    plot_f1_score_report(y_true, y_pred, target_classes)

'''
담현님을 위한 체크포인트
target_names 추가: classification_report에 target_names=class_names를 넣어주면, 
리포트에 숫자가 아닌 실제 상품 카테고리명이 출력되어 훨씬 읽기 편해집니다.

verbose=0: 평가 시에 터미널이 predict 진행 표시로 도배되지 않도록 설정했습니다.

'''
'''
✅ 이 코드가 '현업급'인 이유 3가지
AUTOTUNE과 prefetch의 조화(cache제거) : 셔플링과 프리페치를 통해 병렬 처리 구조완성 -> 에포크당 300초 내외
CPU가 다음 데이터를 준비하는 동안 GPU가 현재 데이터를 학습하게 
만드는 병렬 처리 구조입니다. 에포크당 300초 내외로 끊기는 이유가 바로 이 설정 덕분입니다.



EarlyStopping의 restore_best_weights=True: 이게 정말 중요합니다. 
50번을 다 돌지 않더라도, 학습 중 **가장 성적이 좋았던 시점의 무게(Weight)**를 자동으로
복구해서 저장해주기 때문에 모델의 신뢰도가 훨씬 높습니다.

labels='inferred'와 class_names 지정: 폴더명을 자동으로 라벨로 인식하게 하되, 
target_classes로 순서를 강제했기 때문에 나중에 모델이 
'55'번 폴더를 '24'번으로 착각할 일 없이 정확하게 매칭됩니다.
'''
'''
"1단계 동결 학습 결과, 클래스 197은 f1-score 0.85의 높은 성능을 보였으나, 클래스 55와 60은 0.6 미만의 저조한 성적을 기록하며 전체 정확도가 0.67에 머물렀다. 이는 고정된 특징 추출기로는 상품 간 미세한 차이를 식별하는 데 한계가 있음을 시사한다. 따라서 전체 가중치를 미세 조정하는 Fine-tuning을 통해 모델의 식별력을 고도화하고자 한다."

'''