import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm import tqdm
from PIL import Image
import random
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import cv2
from ultralytics import YOLO


######### 데이터 분리
def organize_images_by_csv(data_dir, csv_path):
    df = pd.read_csv(csv_path)
    print(" 이미지 재분류 및 폴더 정리를 시작합니다...")

    # [수정 포인트] 하위 폴더 어디에 있든 파일명만 맞으면 다 찾아냅니다.
    # Recursive 옵션을 써서 모든 하위 폴더의 파일을 리스트업합니다.
    all_files = {os.path.basename(f): f for f in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True) if os.path.isfile(f)}

    count = 0
    for _, row in df.iterrows():
        img_name = row['name']
        category = str(row['group'])

        # 1. 대상 카테고리 폴더 생성
        cat_dir = os.path.join(data_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        # 2. 파일이 존재하는지 확인 (딕셔너리에서 검색)
        if img_name in all_files:
            src_path = all_files[img_name] # 현재 위치 (어느 폴더든 상관없음)
            dst_path = os.path.join(cat_dir, img_name) # 이동할 새 위치

            # 현재 위치와 이동할 위치가 같으면 건너뜁니다 (이미 잘 들어가 있는 경우)
            if src_path != dst_path:
                shutil.move(src_path, dst_path)
                count += 1
                if count % 1000 == 0:
                    print(f"{count}장 이동 완료...")

    print(f"찾은 전체 파일 개수: {len(all_files)}")
    print(f"CSV에서 읽은 첫 번째 파일명: '{df.iloc[0]['name']}'")
    print(f"실제 파일 리스트 중 하나: '{list(all_files.keys())[0]}'")
    print(f" 총 {count}장의 이미지 재분류 완료!")

##################
# EDA분석
##################
# 전체적인 데이터 분포 시각화
def visualize_distribution(csv_path):
    df = pd.read_csv(csv_path)
    column_name = 'group' if 'group' in df.columns else df.columns[1]

    plt.figure(figsize=(10, 5))
    df[column_name].value_counts().head(20).plot(kind='bar')
    plt.title("Top 20 Categories Distribution")
    plt.ylabel("Image Count")
    plt.show()

# 2. 이미지 해상도 및 가로세로 비율(Aspect Ratio) 분석
def analyze_image_specs(data_dir):
    image_specs = []

    # 지원하는 이미지 확장자
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    print(f"🚀 {data_dir} 내 이미지 분석 시작...")

    # 모든 하위 폴더를 뒤져서 이미지 정보 수집
    for root, dirs, files in os.walk(data_dir):
        for file in tqdm(files):
            if file.lower().endswith(valid_extensions):
                try:
                    img_path = os.path.join(root, file)
                    with Image.open(img_path) as img:
                        width, height = img.size
                        aspect_ratio = width / height
                        image_specs.append({
                            'width': width,
                            'height': height,
                            'aspect_ratio': round(aspect_ratio, 2),
                            'area': width * height # 이미지 크기(화소 수)
                        })
                except Exception as e:
                    print(f"파일 읽기 오류 ({file}): {e}")

    df = pd.DataFrame(image_specs)
    if df.empty:
        print("❌ 분석할 이미지를 찾지 못했습니다.")
        return None

    # --- 시각화 ---
    plt.figure(figsize=(15, 5))

    # 1. 너비와 높이의 분포 (Joint Plot)
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='width', y='height', alpha=0.5)
    plt.title('Image Resolution Distribution')
    plt.xlabel('Width')
    plt.ylabel('Height')

    # 2. 가로세로 비율 분포 (Histogram)
    plt.subplot(1, 3, 2)
    sns.histplot(df['aspect_ratio'], bins=30, kde=True, color='orange')
    plt.axvline(1.0, color='red', linestyle='--', label='Square (1:1)')
    plt.title('Aspect Ratio Distribution')
    plt.legend()

    # 3. 해상도(Area) 분포
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df['area'], color='lightgreen')
    plt.title('Image Area (Total Pixels)')

    plt.tight_layout()
    plt.show()
    # 통계 요약 출력
    print("\n" + "="*30)
    print("📊 이미지 통계 요약")
    print("="*30)
    print(df.describe().loc[['mean', 'min', '50%', 'max']])
    print(f"\n최빈 가로세로 비율: {df['aspect_ratio'].mode()[0]}")

    return df

# 3. 노이즈 샘플링, Grid View로 출력해서 눈으로 확인
def visualize_grid_sampling(data_dir, num_classes=10, row=3, col=5):
    """
    각 카테고리에서 무작위 이미지를 샘플링하여 Grid View로 출력합니다.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 1. 모든 카테고리(폴더명) 가져오기
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not categories:
        print("❌ 분석할 카테고리(폴더)를 찾지 못했습니다.")
        return

    # 2. 분석할 카테고리 수 제한 (너무 많으면 Grid가 터져요)
    selected_categories = random.sample(categories, min(len(categories), num_classes))
    print(f"🚀 {len(selected_categories)}개 카테고리에서 노이즈 샘플링 시작...")

    fig, axes = plt.subplots(len(selected_categories), col, figsize=(col * 3, len(selected_categories) * 3))

    # 만약 한 카테고리만 뽑았을 때 axes가 2차원이 아니게 되는 것 방지
    if len(selected_categories) == 1:
        axes = axes.reshape(1, -1)

    # 한글 폰트 설정 (필요시)
    plt.rcParams['font.family'] = 'Malgun Gothic'

    for i, category in enumerate(tqdm(selected_categories)):
        category_path = os.path.join(data_dir, category)
        all_images = [f for f in os.listdir(category_path) if f.lower().endswith(valid_extensions)]

        if len(all_images) < col:
            print(f"⚠️ {category} 카테고리에 이미지가 너무 적습니다. ({len(all_images)}장)")
            sample_images = all_images
        else:
            sample_images = random.sample(all_images, col)

        for j in range(col):
            # 행(i)과 열(j)에 맞는 격자(ax) 선택
            ax = axes[i, j]
            if j < len(sample_images):
                img_path = os.path.join(category_path, sample_images[j])
                try:
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis('off')
                    if j == 0:
                        ax.set_title(f"Label: {category}", fontsize=12, weight='bold')
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                    ax.axis('off')
            else:
                # 이미지가 부족할 경우 빈칸으로 표시
                ax.axis('off')

    plt.tight_layout()
    plt.show()

################
## 최종 데이터셋 전처리
################
def letterbox_image(image, size=(224, 224)):
    """이미지 비율을 유지하며 패딩을 넣어 리사이징 (Letterboxing)"""
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0)) # 검은색 배경
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def preprocess_and_split_data(raw_data_dir, output_dir, target_classes, apply_yolo=True):
    # 1. 저장 경로 생성
    processed_root = os.path.join(output_dir, "processed_images")
    os.makedirs(processed_root, exist_ok=True)

    all_data = []

    print(f"🚀 {len(target_classes)}개 카테고리 최종 전처리 시작...")

    # [추가] YOLO 모델 로드 (루프 밖에서 한 번만 로드하여 속도 최적화)
    if apply_yolo:
        print("🤖 YOLOv8 모델을 로딩 중입니다 (배경 제거용)...")
        yolo_model = YOLO('yolov8n.pt')

    for category in target_classes:
        src_cat_path = os.path.join(raw_data_dir, category)
        dst_cat_path = os.path.join(processed_root, category)
        os.makedirs(dst_cat_path, exist_ok=True)

        if not os.path.exists(src_cat_path):
            print(f"⚠️ {category} 원본 폴더를 찾을 수 없어 건너뜁니다.")
            continue

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
        img_files = [f for f in os.listdir(src_cat_path) if f.endswith(valid_extensions)]

        # 데이터 균형 조정 (최대 1000장)
        if len(img_files) > 1000:
            img_files = np.random.choice(img_files, 1000, replace=False)

        for img_name in tqdm(img_files, desc=f"Processing Class {category}"):
            try:
                img_path = os.path.join(src_cat_path, img_name)

                if apply_yolo:
                    # OpenCV로 이미지 읽기 (YOLO는 cv2 포맷을 사용)
                    img_cv2 = cv2.imread(img_path)

                    # YOLO 탐지 실행
                    results = yolo_model(img_cv2, verbose=False)
                    boxes = results[0].boxes

                    if len(boxes) > 0:
                        # 신뢰도가 가장 높은 첫 번째 객체의 좌표 (x1, y1, x2, y2)
                        b = boxes[0].xyxy[0].cpu().numpy().astype(int)

                        # 방어 로직: 자를 좌표가 이미지 크기를 벗어나지 않도록 보정
                        y1, y2 = max(0, b[1]), min(img_cv2.shape[0], b[3])
                        x1, x2 = max(0, b[0]), min(img_cv2.shape[1], b[2])

                        crop_img = img_cv2[y1:y2, x1:x2]
                    else:
                        # YOLO가 객체를 못 찾으면 원본 이미지 그대로 사용 (Fallback)
                        crop_img = img_cv2

                    # cv2(BGR)를 PIL(RGB)로 변환 (다음 단계인 letterbox를 위해)
                    img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                else:
                    # YOLO를 안 쓸 경우 기존처럼 PIL로 바로 읽기
                    img_pil = Image.open(img_path).convert('RGB')

                # 2. Letterboxing 처리 (비율 유지하며 224x224 빈 공간 검은색 패딩)
                proc_img = letterbox_image(img_pil, size=(224, 224))

                # 3. 결과 저장
                proc_img.save(os.path.join(dst_cat_path, img_name))

                all_data.append({'name': img_name, 'group': category})
            except Exception as e:
                print(f"❌ {img_name} 처리 중 에러: {e}")

    # 2. 데이터프레임 생성 및 분할
    df = pd.DataFrame(all_data)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['group'])

    # 3. CSV 저장
    train_df.to_csv(os.path.join(output_dir, "EdaTrain.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "EdaTest.csv"), index=False)

    print("\n" + "="*30)
    print(f"✅ 전처리 완료 (YOLO Crop 적용: {apply_yolo})!")
    print(f"📁 저장 위치: {output_dir}")
    print(f"📊 Train: {len(train_df)}장 / Test: {len(test_df)}장")
    print("="*30)

########### 실행부
if __name__ == "__main__":

    # 1. 경로 설정
    data_dir = r"C:\SmartConveyor\data\raw\train"
    csv_path = r"C:\SmartConveyor\data\raw\train.csv"

    # 분류
    print("=== [1단계] 데이터 재분류 프로세스 시작 ===")
    organize_images_by_csv(data_dir, csv_path)

    # EDA
    print("\n=== [2단계] 데이터 분포 분석 (EDA) ===")
    visualize_distribution(csv_path)

    print("\n=== [3단계] 이미지 규격 및 비율 분석 ===")
    analyze_image_specs(data_dir)

    print("\n=== [4단계] 노이즈 샘플링 시각화 ===")

    # 전처리

    data_dir = r"C:\SmartConveyor\data\raw\train"
    out_dir = r"C:\SmartConveyor\data\raw"
    final_categories = ['55', '24', '205', '197', '46', '40', '60', '240']

    preprocess_and_split_data(data_dir, out_dir, final_categories, apply_yolo=True)
