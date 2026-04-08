# 바코드 생성 및 관리
import barcode
from barcode.writer import ImageWriter
import os
from datetime import datetime

class BarcodeManager:
    def __init__(self, base_path="data/barcodes"):
        # 바코드가 저장될 기본 폴더 설정
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def generate_product_barcode(self, product_name, product_id=None):
        """
        상품명과 ID(또는 시간)를 조합해 고유한 바코드를 생성합니다.
        """
        # 1. 고유 식별자 생성 (ID가 없으면 현재 시간 기반으로 생성)
        if product_id is None:
            product_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # 2. 바코드에 들어갈 텍스트 구성 (예: PROD-SNACK-20260406)
        barcode_text = f"{product_name[:10].upper()}-{product_id}"

        try:
            # 3. Code128 형식의 바코드 객체 생성
            code128 = barcode.get_barcode_class('code128')
            my_barcode = code128(barcode_text, writer=ImageWriter())

            # 4. 파일명 설정 및 저장 (확장자는 ImageWriter가 자동으로 .png를 붙임)
            file_name = f"bc_{barcode_text}"
            file_path = os.path.join(self.base_path, file_name)

            # 저장 실행
            full_path = my_barcode.save(file_path)
            print(f"바코드 생성 완료: {full_path}")

            return full_path

        except Exception as e:
            print(f"바코드 생성 중 오류 발생: {e}")
            return None

    def clear_barcodes(self):
        """
        테스트 후 쌓인 바코드 파일들을 정리하는 유틸리티 기능
        """
        for file in os.listdir(self.base_path):
            file_path = os.path.join(self.base_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)


'''
📝 담현님을 위한 설계 포인트
클래스화 (Encapsulation): BarcodeManager라는 클래스로 묶어서 관리하면, main.py에서 manager.generate_product_barcode() 한 줄로 깔끔하게 호출할 수 있습니다.

datetime 활용: 실시간 컨베이어 벨트에서는 초 단위로 물건이 지나가기 때문에, timestamp를 사용해 파일 이름이 겹치지 않게 하는 것이 실무적인 포인트입니다.

ImageWriter 옵션: ImageWriter를 쓰면 바코드 밑에 사람이 읽을 수 있는 텍스트도 함께 출력되어 실제 물류 라벨처럼 보입니다.
'''