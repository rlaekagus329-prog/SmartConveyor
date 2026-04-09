import tensorflow as tf
from tensorflow.keras import layers, models

'''
# 최초 커스텀cnn 모델
def build_smart_conveyor_model(num_classes):
    model = models.Sequential([
        # --- [0단계] 데이터 증강 (Data Augmentation) ---
        # 학습 단계에서만 작동하며, 이미지를 무작위로 변형해 응용력을 높입니다.
        layers.RandomFlip("horizontal", input_shape=(224, 224, 3)), # 좌우 반전
        layers.RandomRotation(0.1), # 10% 내외로 회전
        layers.RandomZoom(0.1),     # 10% 내외로 확대/축소

        # --- [1단계] 특징 추출 ---
        # 첫 번째 블록
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 두 번째 블록 (점점 더 복잡한 패턴 학습)
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 세 번째 블록
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # --- [2단계] 분류 ---
        layers.GlobalAveragePooling2D(), # 플래튼 대신 사용
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # 과적합 방지를 위해 50%를 끕니다
        layers.Dense(num_classes, activation='softmax') # 최종 분류
    ])
#모델 컴파일, 머신러닝과 다른점
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
'''

'''
# 구조 강화 버전
def build_smart_conveyor_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),

        # 128 -> 256 -> 512까지 확장
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same', activation='relu'), # 층 추가
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'), # 뉴런 수 확장
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

'''


# 모델 변경 버전 - 모바일넷
def build_smart_conveyor_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False #  1단계 False, 2단계 TRUE

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # 테스트용: 모델 요약 출력
    my_model = build_smart_conveyor_model(num_classes=8)
    my_model.summary()
