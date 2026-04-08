import pandas as pd
import os

class InventoryManager:
    def __init__(self, csv_path="inventory.csv"):
        self.csv_path = csv_path
        self.classes = ['55', '24', '205', '197', '46', '40', '60', '240']
        self.load_inventory()

    def load_inventory(self):
        """CSV 파일이 있으면 불러오고, 없으면 새로 만듭니다."""
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path, index_col='class_id')
            # CSV 데이터가 문자열일 수 있으므로 인덱스를 문자열로 통일
            self.df.index = self.df.index.astype(str)
        else:
            # 초기 데이터 생성
            data = {'count': [0] * len(self.classes)}
            self.df = pd.DataFrame(data, index=self.classes)
            self.df.index.name = 'class_id'
            self.save_to_csv()

    def add_item(self, class_id):
        """특정 클래스의 수량을 1 증가시키고 바로 저장합니다."""
        class_id = str(class_id)
        if class_id in self.df.index:
            self.df.at[class_id, 'count'] += 1
            self.save_to_csv()
            return self.df.at[class_id, 'count']
        return None

    def save_to_csv(self):
        """현재 상태를 CSV 파일로 저장합니다."""
        self.df.to_csv(self.csv_path)

    def get_current_stock(self):
        """전체 재고 현황을 딕셔너리로 반환합니다."""
        return self.df['count'].to_dict()