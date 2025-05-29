import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json

# 설정 파일 import
from config import get_app_config, check_secrets

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
</style>
""", unsafe_allow_html=True)

# 데이터베이스 연결 클래스 (간소화)
class SimpleBidManager:
    def __init__(self):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            config = get_app_config()
            self.connection = psycopg2.connect(
                config.database.url,
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
        except Exception as e:
            st.error(f"데이터베이스 연결 실패: {e}")
            self.connection = None
    
    def get_live_bids(self, limit=50):
        """실시간 입찰 공고 조회"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            query = """
            SELECT 
                bidNtceNo,
                raw->>'bidNtceNm' as bidNtceNm,
                raw->>'ntceInsttNm' as ntceInsttNm,
                raw->>'bsnsDivNm' as bsnsDivNm,
                raw->>'asignBdgtAmt' as asignBdgtAmt,
                raw->>'bidNtceDate' as bidNtceDate,
                raw->>'bidClseDate' as bidClseDate,
                raw
            FROM bids_live
            ORDER BY created_at DESC
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            st.error(f"데이터 조회 오류: {e}")
            return []
    
    def search_bids(self, keyword):
        """키워드로 입찰 공고 검색"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            query = """
            SELECT 
                bidNtceNo,
                raw->>'bidNtceNm' as bidNtceNm,
                raw->>'ntceInsttNm' as ntceInsttNm,
                raw->>'bsnsDivNm' as bsnsDivNm,
                raw->>'asignBdgtAmt' as asignBdgtAmt,
                raw
            FROM bids_live
            WHERE raw->>'bidNtceNm' ILIKE %s
            ORDER BY created_at DESC
            LIMIT 20
            """
            cursor.execute(query, (f"%{keyword}%",))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            st.error(f"검색 오류: {e}")
            return []

# AI 챗봇 클래스 (1단계 - LangChain만 사용)
class SimpleAIChatbot:
    def __init__(self):
        try:
            from langchain_openai import ChatOpenAI
            config = get_app_config()
            self.llm = ChatOpenAI(
                api_key=config.openai.api_key,
                model=config.openai.model,
                temperature=0.7
            )
        except Exception as e:
            st.error(f"AI 챗봇 초기화 실패: {e}")
            self.llm = None
    
    def get_response(self, question: str, bid_data: list) -> str:
        """간단한 AI 응답 생성"""
        if not self.llm:
            return "AI 서비스를 사용할 수 없습니다."
        
        try:
            # 컨텍스트 구성
            context = ""
            if bid_data:
                context = "관련 입찰 공고:\n"
                for i, bid in enumerate(bid_data[:3]):
                    context += f"{i+1}. {bid.get('bidntcenm', '제목없음')} - {bid.get('ntceinsttm', '기관없음')}\n"
            
            prompt = f"""
사용자 질문: {question}

{context}

위 정보를 바탕으로 입찰 공고에 대해 도움이 되는 답변을 해주세요.
답변은 친절하고 간결하게 해주세요.
"""
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"응답 생성 중 오류 발생: {e}"

# 유틸리티 함수
def convert_to_won_format(amount):
    """금액을 원 단위로 포맷팅"""
    try:
        if not amount:
            return "공고 참조"
        
        amount = float(str(amount).replace(",", ""))
        if amount >= 100000000:
            return f"{amount/100000000:.1f}억원"
        elif amount >= 10000:
            return f"{amount/10000:.1f}만원"
        else:
            return f"{int(amount):,}원"
    except:
        return "공고 참조"

# 매니저 초기화
@st.cache_resource
def init_managers():
    check_secrets()
    bid_manager = SimpleBidManager()
    chatbot = SimpleAIChatbot()
    return bid_manager, chatbot

# 챗봇 질문 처리
def process_question(question: str, chatbot: SimpleAIChatbot, bid_manager: SimpleBidManager):
    """질문 처리 및 응답 생성"""
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_messages.append({"role": "user", "content": question})
    
    # 응답 생성
    with st.spinner("AI가 답변을 생성하고 있습니다..."):
        # 관련 입찰 공고 검색
        search_results = bid_manager.search_bids(question)
        
        # AI 응답 생성
        response = chatbot.get_response(question, search_results)
    
    # 응답 표시
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_messages.append({"role": "assistant", "content": response})

# 메인 애플리케이션
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI 입찰 공고 검색 서비스</h1>
        <p>실시간 입찰 정보와 AI 상담을 한 번에!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 매니저 초기화
    bid_manager, chatbot = init_managers()
    
    # 탭 생성 (AI 챗봇 탭 추가)
    tab1, tab2, tab3 = st.tabs(["📢 실시간 입찰 공고", "🔍 검색", "🤖 AI 상담"])
    
    with tab1:
        st.subheader("📢 최신 입찰 공고")
        
        # 데이터 로드
        bids = bid_manager.get_live_bids()
        
        if bids:
            st.success(f"총 {len(bids)}건의 공고를 불러왔습니다.")
            
            # 결과 표시
            for i, bid in enumerate(bids):
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    col1.markdown(f"**{bid.get('bidntcenm', '제목 없음')}**")
                    col2.write(f"{bid.get('ntceinsttm', '기관명 없음')} | {bid.get('bsnsdivnm', '분류 없음')}")
                    col3.write(convert_to_won_format(bid.get('asignbdgtamt', 0)))
                    
                    if st.button("상세보기", key=f"detail_{i}"):
                        st.session_state.selected_bid = bid
                        st.session_state.show_detail = True
                        st.rerun()
                    
                    st.divider()
        else:
            st.warning("입찰 공고 데이터를 불러올 수 없습니다.")
    
    with tab2:
        st.subheader("🔍 입찰 공고 검색")
        
        # 검색 UI
        keyword = st.text_input("검색어를 입력하세요", placeholder="예: AI, 소프트웨어, 서버 등")
        
        if st.button("검색", type="primary"):
            if keyword:
                with st.spinner("검색 중..."):
                    results = bid_manager.search_bids(keyword)
                    
                    if results:
                        st.success(f"'{keyword}'에 대한 검색 결과: {len(results)}건")
                        
                        for i, result in enumerate(results):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                col1.markdown(f"**{result.get('bidntcenm', '제목 없음')}**")
                                col1.caption(f"{result.get('ntceinsttm', '기관명 없음')} | {result.get('bsnsdivnm', '분류 없음')}")
                                col2.write(convert_to_won_format(result.get('asignbdgtamt', 0)))
                                st.divider()
                    else:
                        st.warning(f"'{keyword}'에 대한 검색 결과가 없습니다.")
            else:
                st.warning("검색어를 입력해주세요.")
    
    with tab3:
        st.subheader("🤖 AI 입찰 상담")
        
        # 예시 질문 버튼
        st.markdown("**💡 예시 질문:**")
        example_questions = [
            "AI 관련 입찰 공고가 있나요?",
            "소프트웨어 개발 입찰의 특징은?",
            "최근 IT 입찰 동향은 어떤가요?"
        ]
        
        cols = st.columns(3)
        for idx, question in enumerate(example_questions):
            if cols[idx].button(question, key=f"example_{idx}"):
                st.session_state.pending_question = question
                st.rerun()
        
        if st.button("🔄 대화 초기화"):
            st.session_state.chat_messages = []
            st.rerun()
        
        # 세션 상태 초기화
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # 이전 대화 표시
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 예시 질문 처리
        if hasattr(st.session_state, 'pending_question'):
            question = st.session_state.pending_question
            del st.session_state.pending_question
            process_question(question, chatbot, bid_manager)
            st.rerun()
        
        # 사용자 입력
        if prompt := st.chat_input("입찰 관련 질문을 해주세요"):
            process_question(prompt, chatbot, bid_manager)
    
    # 상세보기 모달
    if st.session_state.get('show_detail', False):
        bid = st.session_state.get('selected_bid', {})
        
        st.markdown("---")
        st.subheader("📋 공고 상세 정보")
        
        raw_data = bid.get('raw', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**공고번호:** {bid.get('bidntceno', 'N/A')}")
            st.write(f"**공고명:** {bid.get('bidntcenm', 'N/A')}")
            st.write(f"**공고기관:** {bid.get('ntceinsttm', 'N/A')}")
            st.write(f"**분류:** {bid.get('bsnsdivnm', 'N/A')}")
        
        with col2:
            st.write(f"**예산:** {convert_to_won_format(bid.get('asignbdgtamt', 0))}")
            st.write(f"**게시일:** {raw_data.get('bidNtceDate', 'N/A')}")
            st.write(f"**마감일:** {raw_data.get('bidClseDate', 'N/A')}")
            st.write(f"**상태:** {raw_data.get('bidNtceSttusNm', 'N/A')}")
        
        if st.button("닫기"):
            st.session_state.show_detail = False
            st.rerun()

# 세션 상태 초기화
if 'show_detail' not in st.session_state:
    st.session_state.show_detail = False

if __name__ == "__main__":
    main()
