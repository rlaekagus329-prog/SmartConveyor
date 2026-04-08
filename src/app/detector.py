import numpy as np
import tensorflow as tf
from PIL import Image

class SmartDetector:
    def __init__(self, model_path, class_names):
        # 1. 모델 로드 (생성 시 한 번만 실행)
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names

    def preprocess_image(self, pil_image):
        """
        입력받은 PIL 이미지를 모델 규격(224x224)에 맞게 변환
        """
        # 이미지 리사이즈 및 정규화
        img = pil_image.resize((224, 224))
        img_array = np.array(img) / 255.0

        # 4차원 배치 형태로 확장 (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, pil_image):
        """
        전처리부터 예측까지 한 번에 수행
        """
        processed_img = self.preprocess_image(pil_image)
        predictions = self.model.predict(processed_img)

        # 가장 높은 확률을 가진 인덱스 추출
        result_idx = np.argmax(predictions[0])
        label = self.class_names[result_idx]
        confidence = predictions[0][result_idx] * 100

        return label, confidence

    def predict_video_frame(self, frame):
        """
        OpenCV 프레임(numpy array)을 받아서 예측 (동영상용)
        """
        # OpenCV는 BGR이므로 RGB로 변환 후 PIL로 변경
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        return self.predict(pil_img)

    '''
    📝 담현님을 위한 '구현 포인트' 정리
preprocess_image 분리: 나중에 이미지 크기를 128x128로 바꾸고 싶을 때 이 함수만 수정하면 됩니다.

predict_video_frame 대비: 지금은 사진 위주지만, 나중에 OpenCV로 동영상을 읽어올 때 frame을 바로 집어넣을 수 있게 미리 통로를 뚫어놨습니다. (동영상은 색상 체계가 BGR이라 RGB로 바꾸는 게 핵심이에요!)

에러 방지: predictions[0]을 사용하는 이유는 모델이 항상 '배치(묶음)' 단위로 결과를 내놓기 때문입니다. 우리는 한 장씩 넣으니 0번 인덱스만 가져오면 됩니다.
    
    '''



