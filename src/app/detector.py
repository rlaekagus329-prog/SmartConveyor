import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from src.app.inventory_manager import InventoryManager


class SmartConveyorDetector:
    def __init__(self, model_path):
        print(f"🔄 모델 로딩 중: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.classes = ['55', '24', '205', '197', '46', '40', '60', '240']
        self.img_size = (224, 224)
        # --- [추가] 수량 카운터 초기화 ---
        self.inventory_manager = InventoryManager("inventory.csv")

    def detect(self, image_path):
        if not os.path.exists(image_path):
            return None, 0.0

        # 1. 이미지 로드 (224x224)
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array, verbose=0)
        score = np.max(predictions)
        class_idx = np.argmax(predictions)
        result_class = self.classes[class_idx]

        predictions = self.model.predict(img_array, verbose=0)

        # 💡 [진단용 로그] 이 값이 어떻게 나오는지 알려주세요!
        print(f"DEBUG - Raw Predictions: {predictions}")
        print(f"DEBUG - Argmax Index: {np.argmax(predictions)}")

        score = np.max(predictions)

        # --- [추가] 90% 이상의 확신도일 때만 수량 +1 ---
        if score >= 0.90:
            # CSV 파일의 해당 클래스 숫자를 +1 하고 파일로 저장함
            new_count = self.inventory_manager.add_item(result_class)
            print(f"📈 CSV 업데이트 완료: {result_class}번 (현재 {new_count}개)")
        else:
            print(f"⚠️ 확신도가 낮아({score*100:.1f}%) 수량을 추가하지 않았습니다.")

        return result_class, score

    def show_inventory(self):

        """현재까지 집계된 전체 수량 출력"""
        print("\n" + "="*10 + " 현재 재고 현황 " + "="*10)
        current_stock = self.inventory_manager.get_current_stock()
        for cls, count in current_stock.items():
            print(f"📦 클래스 {cls: >3} : {count}개")
        print("="*34)

# --- 메인 실행부 (사용자 입력 기능) ---
# --- 실행부 ---
if __name__ == "__main__":
    MODEL_FILE = "../../smart_conveyor_B_final_90.keras"
    detector = SmartConveyorDetector(MODEL_FILE)

    while True:
        user_input = input("\n📸 이미지 경로 입력 (종료 'q', 현황 's'): ").strip().strip('"')

        if user_input.lower() == 'q':
            detector.show_inventory() # 종료 전 최종 결과 보여주기
            break

        if user_input.lower() == 's':
            detector.show_inventory()
            continue

        label, confidence = detector.detect(user_input)

        if label:
            print(f"🔍 판별: {label} ({confidence*100:.2f}%)")