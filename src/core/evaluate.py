import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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