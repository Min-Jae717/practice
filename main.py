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

    if "selected_
