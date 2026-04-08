import streamlit as st
from PIL import Image
import os

# 우리가 만든 모듈들 불러오기
from src.app.detector import SmartDetector
from src.app.barcode_mgr import BarcodeManager

# 1. 초기 설정 및 리소스 로드
st.set_page_config(page_title="SmartConveyor AI", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def initialize_system():
    # 학습된 모델 경로와 클래스 명칭 (데이터셋에 맞게 추후 수정)
    model_path = 'models/smart_conveyor_v1.keras'
    # Products 10k의 실제 카테고리 리스트가 들어갈 자리입니다.
    class_names = ['Product_A', 'Product_B', 'Food_C', 'Damaged']

    detector = SmartDetector(model_path, class_names)
    barcode_gen = BarcodeManager()
    return detector, barcode_gen

# 시스템 가동
try:
    detector, barcode_gen = initialize_system()
except Exception as e:
    st.error(f"모델을 불러올 수 없습니다. 학습이 완료되었는지 확인하세요! 에러: {e}")
    st.stop()

# 2. 사이드바 UI
with st.sidebar:
    st.header("⚙️ System Control")
    st.info("SmartConveyor v1.0\nDeveloper: Kim Dam Hyun")
    if st.button("바코드 저장소 비우기"):
        barcode_gen.clear_barcodes()
        st.success("저장소가 정리되었습니다.")

# 3. 메인 화면 UI
st.title("🏭 SmartConveyor: 실시간 상품 분류 시스템")
st.write("---")

uploaded_file = st.file_uploader("컨베이어 벨트 위의 상품 사진을 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 화면을 두 칸으로 분할
    col_img, col_res = st.columns([1, 1])

    with col_img:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 입력된 상품 이미지", use_container_width=True)

    with col_res:
        st.subheader("🔍 분석 엔진 결과")

        # 4. 분석 실행
        with st.spinner('상품을 식별 중입니다...'):
            label, confidence = detector.predict(image)

        # 결과 표시 (정상/파손 분기 처리)
        if label == "Damaged":
            st.error(f"### 상태: [불량 감지] - {label}")
            st.write(f"**신뢰도:** {confidence:.2f}%")
            st.warning("⚠️ 파손된 상품입니다. 바코드를 생성하지 않고 라인을 정지합니다.")
        else:
            st.success(f"### 상품명: {label}")
            st.write(f"**신뢰도:** {confidence:.2f}%")

            # 5. 바코드 생성 및 출력
            st.write("---")
            st.subheader("🏷️ 자동 생성 바코드")
            barcode_path = barcode_gen.generate_product_barcode(label)

            if barcode_path:
                # 바코드 이미지 표시 (save() 결과에 .png가 붙으므로 확인 필요)
                st.image(barcode_path + ".png", caption=f"{label} 전용 물류 바코드")

                # 다운로드 버튼 추가 (현장 업무 편의성!)
                with open(barcode_path + ".png", "rb") as file:
                    st.download_button(
                        label="바코드 이미지 다운로드",
                        data=file,
                        file_name=f"barcode_{label}.png",
                        mime="image/png"
                    )

# 4. 하단 상태 표시줄
st.write("---")
st.caption("© 2026 SmartConveyor Project - Powered by TensorFlow & Streamlit")