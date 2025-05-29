import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import time

# 커스텀 모듈 import
from database import BidDataManager, convert_to_won_format, format_won, format_joint_contract
from ai_search import init_chatbot, init_semantic_search
from langgraph_workflow import create_bid_search_workflow, create_hybrid_search_workflow
from ui_tabs import (
    show_live_bids_tab,
    show_semantic_search_tab, 
    add_langgraph_search_tab,
    add_chatbot_to_streamlit
)
from config import check_secrets

# Streamlit 페이지 설정
st.set_page_config(
    page_title="입찰 공고 서비스", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .process-log {
        background-color: #f0f2f6;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        border-left: 3px solid #667eea;
    }
    
    .db-result {
        background-color: #e8f4fd;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        border-left: 3px solid #2196f3;
    }
    
    .vector-result {
        background-color: #fff3e0;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        border-left: 3px solid #ff9800;
    }
    
    .api-result {
        background-color: #e8f5e8;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        border-left: 3px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# 전역 변수 및 초기화
@st.cache_resource
def initialize_managers():
    """매니저 클래스들 초기화"""
    # Secrets 확인
    check_secrets()
    
    bid_manager = BidDataManager()
    chatbot = init_chatbot()
    semantic_engine = init_semantic_search()
    
    return bid_manager, chatbot, semantic_engine

# 매니저 초기화
bid_manager, chatbot, semantic_engine = initialize_managers()

# 세션 상태 초기화
if "page" not in st.session_state:
    st.session_state["page"] = "home"

if "current_page" not in st.session_state:
    st.session_state["current_page"] = 0

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# 챗봇 질문 처리 함수
def process_question(question: str, chatbot_instance):
    """질문 처리 및 응답 생성"""
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_messages.append({"role": "user", "content": question})
    
    # 응답 생성
    with st.spinner("입찰 공고를 검색하고 분석 중입니다..."):
        # 벡터 DB 검색
        search_results = chatbot_instance.search_vector_db(question)
        
        # GPT 응답 생성
        response = chatbot_instance.get_gpt_response(question, search_results)
    
    # 응답 표시
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_messages.append({"role": "assistant", "content": response})

# 메인 페이지 라우팅
page = st.session_state.get("page", "home")

if page == "home":
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI 기반 입찰 공고 통합 서비스</h1>
        <p>실시간 입찰 정보부터 AI 분석까지, 스마트한 입찰 관리 솔루션</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📢 실시간 입찰 공고", 
        "🔍 AI 검색", 
        "🚀 LangGraph AI 검색", 
        "🤖 AI 도우미"
    ])
    
    # Tab 1: 실시간 입찰 공고
    with tab1:
        show_live_bids_tab(bid_manager)
    
    # Tab 2: AI 검색 (시맨틱 검색)
    with tab2:
        show_semantic_search_tab(semantic_engine, bid_manager)
        
    # Tab 3: LangGraph AI 검색
    with tab3:
        add_langgraph_search_tab(bid_manager)
    
    # Tab 4: AI 도우미 (챗봇)
    with tab4:
        add_chatbot_to_streamlit(chatbot, process_question)

# 상세 페이지
elif page == "detail":
    if st.button("⬅️ 목록으로 돌아가기"):
        st.session_state["page"] = "home"
        st.rerun()

    if "selected_live_bid" in st.session_state:
        row = st.session_state["selected_live_bid"]
        
        # 데이터 처리
        if isinstance(row, dict):
            # raw 데이터에서 정보 추출 (Supabase의 JSONB 구조)
            raw_data = row.get('raw', {}) if 'raw' in row else row
            
            마감일 = raw_data.get('bidClseDate') or row.get('bidClseDate')
            마감시간 = raw_data.get('bidClseTm') or row.get('bidClseTm')
            게시일 = raw_data.get('bidNtceDate') or row.get('bidNtceDate')
            
            # 날짜 처리
            try:
                if 마감일 and len(str(마감일)) == 8:
                    마감일 = pd.to_datetime(마감일, format='%Y%m%d')
                    마감일_표시 = 마감일.strftime("%Y년 %m월 %d일")
                else:
                    마감일_표시 = "공고 참조"
            except:
                마감일_표시 = "공고 참조"
            
            try:
                if 게시일 and len(str(게시일)) == 8:
                    게시일 = pd.to_datetime(게시일, format='%Y%m%d')
                    게시일_표시 = 게시일.strftime("%Y년 %m월 %d일")
                else:
                    게시일_표시 = "정보 없음"
            except:
                게시일_표시 = "정보 없음"

            마감시간_표시 = 마감시간 if 마감시간 else "공고 참조"

        # 상세 정보 표시
        st.markdown(
            f"""
            <div style="
                background-color: #e0f2f7; 
                padding: 25px 20px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            ">
                <h2 style="color: #0056b3; margin-top: 0px; margin-bottom: 10px; font-weight: bold; font-size: 2.2em;">
                    {raw_data.get('bidNtceNm') or row.get('bidNtceNm', '공고명 없음')}
                </h2>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                        📊 구분: {raw_data.get('bidNtceSttusNm') or row.get('bidNtceSttusNm', '정보 없음')}
                    </span>
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                        🏢 수요기관: {raw_data.get('dmndInsttNm') or row.get('dmndInsttNm', '정보 없음')}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                        📅 게시일: {게시일_표시}
                    </span>                   
                </div>
                    <span style="font-size: 1.2em; font-weight: 600; color: #333;">
                            ⏳ 공고마감일: {마감일_표시} {마감시간_표시}
                    </span>
                    <div style="font-size: 1.5em; font-weight: bold; color: #007bff; text-align: right;">
                        💰 금액: {format_won(str(raw_data.get('asignBdgtAmt') or row.get('asignBdgtAmt', '0')))}
                    </div>
            </div>
            """, unsafe_allow_html=True
        )

        # 추가 정보 섹션
        col1, col2, col3 = st.columns([1,1,1])       
        
        with col1:
            공동수급 = raw_data.get('cmmnReciptMethdNm') or row.get('cmmnReciptMethdNm')
            지역제한 = raw_data.get('rgnLmtYn') or row.get('rgnLmtYn')
            참가가능지역 = raw_data.get('prtcptPsblRgnNm') or row.get('prtcptPsblRgnNm')
            
            st.markdown(
                f"""
                <div style="
                background-color: #f0fdf4;
                border: 1px solid #e5e5e5;
                padding: 25px 20px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                height: 300px; 
            ">
                    <h4 style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">공동수급 • 지역제한</h4>
                    <hr style="border: 1px solid #e5e5e5; margin-top: 10px; margin-bottom: 10px;">
                    <div style="margin-bottom: 10px;">
                        <span style="font-size: 16px; font-weight: bold; color: #333;">🤝 공동수급</span><br>
                        <span style="font-size: 18px; font-weight: 500; color: #000;">
                        {format_joint_contract(공동수급)}</span>
                    </div>
                    <div>
                        <span style="font-size: 16px; font-weight: bold; color: #333;">📍 지역제한</span><br>
                        <span style="font-size: 18px; font-weight: 500; color: #000;">
                            {참가가능지역 if 지역제한 == 'Y' and 참가가능지역 else '없음'}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )          
            
        with col2:
            업종명 = raw_data.get('bidprcPsblIndstrytyNm') or row.get('bidprcPsblIndstrytyNm')
            st.markdown(
                f"""
                <div style="
                background-color: #fff9e6; 
                border: 1px solid #e5e5e5;
                padding: 25px 20px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                height: 300px; 
            ">
                    <h4 style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">🚫업종 제한</h4>
                    <hr style="border: 1px solid #e5e5e5; margin-top: 10px; margin-bottom: 10px;">
                    <p style="font-size: 18px; font-weight: bold; overflow-y: auto; max-height: 90px;">
                        {"<br>".join([f"{i+1}. {item.strip()}" for i,
                                    item in enumerate(str(업종명).split(',')) if str(item).strip()]) 
                                    if 업종명 and str(업종명).strip() != "" else '공문서참조'}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div style="
                background-color: #f0f8ff; 
                border: 1px solid #e5e5e5;
                padding: 25px 20px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                height: 300px; 
                ">
                    <h4 style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">💡 기타 정보</h4>
                    <hr style="border: 1px solid #e5e5e5; margin-top: 10px; margin-bottom: 10px;">
                    <p style="font-size: 16px;">
                        <strong>공고번호:</strong> {raw_data.get('bidNtceNo') or row.get('bidNtceNo', 'N/A')}<br>
                        <strong>공고기관:</strong> {raw_data.get('ntceInsttNm') or row.get('ntceInsttNm', 'N/A')}<br>
                        <strong>분류:</strong> {raw_data.get('bsnsDivNm') or row.get('bsnsDivNm', 'N/A')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # GPT 요약 표시
        bid_no = raw_data.get('bidNtceNo') or row.get('bidNtceNo')
        if bid_no:
            summary_text, created_at, summary_type = bid_manager.get_bid_summary(bid_no)

            created_info = ""
            if created_at:
                type_label = "상세문서 기반" if summary_type == "hwp_based" else "기본정보 기반"
                created_info = f" ({type_label}, 생성일: {created_at})"

            st.markdown(
                f"""
                <div style="
                    background-color: #f0f8ff; 
                    border-left: 5px solid #4682b4; 
                    padding: 15px;
                    margin-top: 10px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                ">
                <div>
                    <span style="font-size: 16px; font-weight: bold; color: #333;">AI 상세요약{created_info}</span><br>   
                    <hr style="border: 1px solid #e5e5e5; margin-top: 10px; margin-bottom: 10px;">         
                </div>
                    <p style="font-size: 16px; font-weight: 500;">{summary_text}</p>
                </div>
                """, unsafe_allow_html=True
            )

if __name__ == "__main__":
    # 자동 새로고침 (선택사항)
    st_autorefresh(interval=300000, key="datarefresh")  # 5분마다 새로고침
    pass
