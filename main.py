import streamlit as st
import pandas as pd
import os
from src.app.detector import SmartConveyorDetector
from src.app.inventory_manager import InventoryManager

# 1. 초기 설정
st.set_page_config(page_title="Smart Conveyor AI Dashboard", layout="wide")
CATEGORY_MAP = {
    '55': '지갑',
    '24': '바지',
    '205': '장신구',
    '197': '퍼즐',
    '46': '구두',
    '40': '운동화',
    '60': '캐리어',
    '240': '립스틱'
}

MODEL_PATH = "smart_conveyor_B_final_90.keras"
CSV_PATH = "inventory.csv"
# 세션상태관리
if 'detector' not in st.session_state:
    st.session_state.detector = SmartConveyorDetector(MODEL_PATH)
if 'inv_manager' not in st.session_state:
    st.session_state.inv_manager = InventoryManager(CSV_PATH)
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# 2. 메인 타이틀
st.title("🏭 Smart Conveyor 실시간 재고 관리")
st.markdown("---")

# 3. 실시간 재고 현황 섹션
st.subheader("📊 현재 재고 현황")
current_inv = st.session_state.inv_manager.get_current_stock()

# 💡 [수정] 표와 그래프용 데이터에 한글 이름을 입힘
inv_list = []
for code, count in current_inv.items():
    name = CATEGORY_MAP.get(code, f"미등록({code})") # 매핑 안되면 코드 노출
    inv_list.append({'품목명': name, '코드': code, '수량': count})

df_inv = pd.DataFrame(inv_list)
# 주요 지표 요약 (가장 많이 쌓인 품목 등)
total_items = df_inv['수량'].sum()
top_item_name = df_inv.loc[df_inv['수량'].idxmax()]['품목명'] if not df_inv.empty else "-"

m1, m2, m3 = st.columns(3)
m1.metric("총 입고 수량", f"{int(total_items)}개")
m2.metric("최다 입고 품목", top_item_name)
m3.metric("시스템 상태", "정상 (99.6%)", delta="Online")

# 그래프와 표를 나란히 배치
col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.bar_chart(df_inv.set_index('품목명')['수량'], color="#29b5e8")

with col_table:
    st.dataframe(df_inv[['품목명', '수량']], use_container_width=True, hide_index=True)
    if st.button("🔄 데이터 전체 초기화", type="secondary", use_container_width=True):
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
            st.session_state.inv_manager.load_inventory()
            st.rerun()

st.markdown("---")

# 4. [하단] 새로운 상품 분석 섹션
st.subheader("🔍 신규 상품 입고 분석")
col_input, col_res = st.columns([1, 1])

with col_input:
    uploaded_file = st.file_uploader("이미지를 업로드하면 AI가 분석을 시작합니다.", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with open("temp_input.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption='대기 중인 이미지', use_container_width=True)

with col_res:
    st.write("### 분석 결과 확인")

    # 분석 버튼
    if uploaded_file is not None:
        if st.button("🚀 AI 분석 실행", use_container_width=True, type="primary"):
            with st.spinner('AI 분석 중...'):
                label, score = st.session_state.detector.detect("temp_input.jpg")
                st.session_state.last_result = {"label": label, "score": score}

                if label and score >= 0.90:
                    st.session_state.inv_manager.add_item(label)
                    st.success(f"✅ 분석 완료: {label}번 상품 등록되었습니다!")
                    st.balloons()
                    st.rerun()
    # 이전 분석 결과 표시 공간
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.info(f"마지막 판별: **{res['label']}번** (신뢰도: {res['score']*100:.2f}%)")
    else:
        st.write("이미지를 업로드하고 분석 버튼을 눌러주세요.")