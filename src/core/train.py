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
target_classes = train_ds.class_names

# 4. 성능 최적화 (이게 핵심!)
# 셔플링과 프리페치(미리 읽어오기)를 통해 GPU가 쉬지 않게 합니다.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# 5. 모델 빌드 (클래스 개수는 데이터셋에 맞게 자동 추출)
model = build_smart_conveyor_model(num_classes)

# 6. 조기 종료 설정 (patience=5)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# 7. 고점에 다다랐을 때 학습률을 낮춰서 정교하게 학습
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
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    df_f1 = df_report.iloc[:-3, :3]

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_f1, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Category-wise Precision, Recall, and F1-score")
    plt.show()

# --- [2] 메인 학습 루프 ---
if __name__ == "__main__":

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

    # 2. 아주 낮은 학습률로 다시 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. 추가 학습 진행 (Fine-tuning)
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        initial_epoch=history.epoch[-1],
        callbacks=[early_stop, reduce_lr]
    )

    # 4. 모델 저장
    model_name = 'smart_conveyor_B_final_90.keras'
    model.save(model_name)

    # --- [3] 결과 출력 (여기서 한 번만 호출!) ---
    plot_learning_curves(history)
    y_true, y_pred = evaluate_model_performance(model, val_ds, target_classes)
    plot_f1_score_report(y_true, y_pred, target_classes)
