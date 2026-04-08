import tensorflow as tf
from tensorflow.keras import layers, models

'''
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


# 모델 변경 버전
def build_smart_conveyor_model(num_classes):
    # 이미 학습된 베이스 모델 (특징 추출기)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = True #  False일단 기존 지식은 고정

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),

        base_model, # 수백만 장 학습한 뇌 장착

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


'''

'''

if __name__ == "__main__":
    # 테스트용: 모델 요약 출력
    my_model = build_smart_conveyor_model(num_classes=8)
    my_model.summary()

    '''
    물류 센터의 다양한 상품(10k 종류)을 구분해야 합니다. 따라서 층을 충분히 쌓되, 계산 효율을 위해 MaxPooling을 적절히 섞어줄 거예요.

1. 입력층 (Input Layer)
Size: (224, 224, 3)

설명: 컬러 이미지(RGB)를 받습니다. 15GB 데이터를 다루려면 메모리 관리를 위해 이미지를 224x224 정도로 줄여서 학습시키는 게 효율적입니다.

2. 특징 추출부 (Feature Extraction - Convolution)
Conv2D + ReLU: 이미지의 선, 면, 질감을 찾아냅니다.

MaxPooling2D: 중요한 정보만 남기고 이미지 크기를 줄여 연산 속도를 높입니다.

BatchNormalization: (전문가 팁!) 학습을 더 빠르고 안정적으로 만들어줍니다.

3. 분류부 (Classification - Dense)
Flatten: 2차원 특징 맵을 1차원으로 쫙 펼칩니다.

Dropout: (과적합 방지!) 모델이 특정 사진에만 집착하지 않도록 무작위로 노드를 끕니다. 수업 때 배운 validation_loss를 낮추는 핵심 기술이죠.

Dense (Output): 최종 상품 카테고리 개수만큼 노드를 설정합니다.
    
'''
'''
담현님을 위한 설계 포인트
num_classes 변수: 1만 개를 다 하려면 이 숫자가 10,000이 되어야 하지만, 일단 데이터가 다 받아지면 상위 몇 개 카테고리만 추려서 테스트할 예정이라 변수로 빼두었습니다.

BatchNormalization: 수업 시간에 가볍게 지나갔을 수도 있지만, 실무형 프로젝트에서는 학습 속도를 비약적으로 높여주는 효자 아이템입니다.

model.summary(): 코드를 실행하면 터미널에 모델의 층과 파라미터 개수가 표로 나옵니다. 이걸 찍어보는 습관이 아주 좋은 개발 습관이에요!
'''