import shutil
import os
import pandas as pd
import tensorflow as tf
from src.core.evaluate import plot_learning_curves, evaluate_model_performance
from src.core.model import build_smart_conveyor_model
from keras.callbacks import EarlyStopping


data_dir = r"C:\SmartConveyor\data\raw\processed_images"
csv_path = r"C:\SmartConveyor\data\raw\EdaTrain.csv"

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
    patience=5,
    restore_best_weights=True
)

# 6. 고점에 다다랐을 때 학습률을 낮춰서 정교하게 학습 (65% 벽 뚫기용)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# 7. 학습 실행!
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # 50번을 다 안 돌아도 Early Stopping이 알아서 멈춰줄 거예요.
    callbacks=[early_stop, reduce_lr]
)

# 8. 모델 저장 (SmartConveyor 프로젝트의 심장)
model.save('smart_conveyor_epoch20v1.keras')
print("모델 저장 완료!")


# 9. 성능 평가
# 9-1. 학습 곡선 그리기 (history 객체 전달)
plot_learning_curves(history)

# 9-2. 상세 성능 평가
evaluate_model_performance(
    model_path='smart_conveyor_epoch20v1.keras',
    val_ds=val_ds,
    class_names=target_classes  # 위에서 저장해둔 리스트 사용
)

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
######### 평가
def plot_learning_curves(history):
    """
    1단계: 학습 과정(Training vs Validation) 시각화
    오버피팅(과적합) 여부를 판단하는 가장 중요한 지표입니다.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Loss 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def evaluate_model_performance(model_path, val_ds, class_names):
    """
    2단계: 최종 모델 성능 평가 (Confusion Matrix & Report)
    어떤 상품을 헷갈려 하는지 구체적으로 분석합니다.
    """
    print("\n[시스템] 모델 성능 평가를 시작합니다...")
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    # 데이터셋을 돌며 예측값 수집
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # 1. 정밀도, 재현율 등 리포트 출력
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 2. 혼동 행렬 시각화
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Product Classification Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    plot_learning_curves(history)
    evaluate_model_performance(model_path, val_ds, class_names)


