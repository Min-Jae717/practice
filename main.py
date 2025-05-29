import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np

# 설정 파일 import
from config import get_app_config, check_secrets

# Streamlit 페이지 설정
st.set_page_config(
    page_title="AI 입찰 공고 서비스", 
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
    .similarity-high { color: #28a745; font-weight: bold; }
    .similarity-medium { color: #ffc107; font-weight: bold; }
    .similarity-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# 데이터베이스 연결 클래스
class EnhancedBidManager:
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
               OR raw->>'ntceInsttNm' ILIKE %s
               OR raw->>'bidprcPsblIndstrytyNm' ILIKE %s
            ORDER BY created_at DESC
            LIMIT 30
            """
            cursor.execute(query, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            st.error(f"검색 오류: {e}")
            return []
    
    def get_semantic_chunks(self, limit=100):
        """벡터 검색용 데이터 조회"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            query = """
            SELECT 
                sc.content,
                sc.metadata,
                sc.bidNtceNo,
                bl.raw->>'bidNtceNm' as bidNtceNm,
                bl.raw->>'ntceInsttNm' as ntceInsttNm
            FROM semantic_chunks sc
            JOIN bids_live bl ON sc.bidNtceNo = bl.bidNtceNo
            ORDER BY sc.created_at DESC
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            # semantic_chunks 테이블이 없을 경우 빈 배열 반환
            return []

# 벡터 검색 클래스
class VectorSearchEngine:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.is_loaded = True
        except Exception as e:
            st.warning(f"벡터 모델 로드 실패: {e}")
            self.model = None
            self.is_loaded = False
    
    def encode_text(self, text):
        """텍스트를 벡터로 변환"""
        if not self.is_loaded:
            return None
        try:
            return self.model.encode([text])[0]
        except Exception as e:
            st.error(f"텍스트 인코딩 오류: {e}")
            return None
    
    def calculate_similarity(self, query_vector, doc_vectors):
        """코사인 유사도 계산"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([query_vector], doc_vectors)[0]
            return similarities
        except Exception as e:
            st.error(f"유사도 계산 오류: {e}")
            return []
    
    def semantic_search(self, query, bid_data, top_k=10):
        """시맨틱 검색 수행"""
        if not self.is_loaded or not bid_data:
            return []
        
        try:
            # 쿼리 벡터화
            query_vector = self.encode_text(query)
            if query_vector is None:
                return []
            
            # 문서 벡터화
            documents = []
            doc_vectors = []
            
            for bid in bid_data:
                # 검색 대상 텍스트 구성
                text = f"{bid.get('bidntcenm', '')} {bid.get('ntceinsttm', '')} {bid.get('bsnsdivnm', '')}"
                doc_vector = self.encode_text(text)
                
                if doc_vector is not None:
                    documents.append(bid)
                    doc_vectors.append(doc_vector)
            
            if not doc_vectors:
                return []
            
            # 유사도 계산
            similarities = self.calculate_similarity(query_vector, doc_vectors)
            
            # 결과 정렬
            results = []
            for i, sim in enumerate(similarities):
                if sim > 0.1:  # 최소 유사도 임계값
                    results.append({
                        'document': documents[i],
                        'similarity': float(sim)
                    })
            
            # 유사도 순으로 정렬
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            st.error(f"시맨틱 검색 오류: {e}")
            return []

# AI 챗봇 클래스 (향상된 버전)
class EnhancedAIChatbot:
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
    
    def get_response(self, question: str, search_results: list, semantic_results: list = None) -> str:
        """향상된 AI 응답 생성"""
        if not self.llm:
            return "AI 서비스를 사용할 수 없습니다."
        
        try:
            # 컨텍스트 구성
            context = "## 관련 입찰 공고 정보:\n\n"
            
            # 키워드 검색 결과
            if search_results:
                context += "### 키워드 검색 결과:\n"
                for i, bid in enumerate(search_results[:3]):
                    context += f"{i+1}. **{bid.get('bidntcenm', '제목없음')}**\n"
                    context += f"   - 기관: {bid.get('ntceinsttm', '기관없음')}\n"
                    context += f"   - 분류: {bid.get('bsnsdivnm', '분류없음')}\n\n"
            
            # 시맨틱 검색 결과
            if semantic_results:
                context += "### AI 유사도 검색 결과:\n"
                for i, result in enumerate(semantic_results[:3]):
                    bid = result['document']
                    similarity = result['similarity']
                    context += f"{i+1}. **{bid.get('bidntcenm', '제목없음')}** (유사도: {similarity:.1%})\n"
                    context += f"   - 기관: {bid.get('ntceinsttm', '기관없음')}\n"
                    context += f"   - 분류: {bid.get('bsnsdivnm', '분류없음')}\n\n"
            
            prompt = f"""
사용자 질문: {question}

{context}

위 입찰 공고 정보를 바탕으로 다음 가이드라인에 따라 답변해주세요:

1. 사용자 질문과 가장 관련성 높은 공고들을 우선적으로 설명
2. 각 공고의 특징과 장점을 간결하게 정리
3. 입찰 참여시 고려사항이나 팁이 있다면 조언
4. 추가 정보가 필요한 경우 어떤 부분을 더 확인해야 하는지 안내

답변은 친절하고 전문적으로, 3-5문장 정도로 간결하게 해주세요.
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

def format_similarity(similarity):
    """유사도 포맷팅"""
    if similarity >= 0.7:
        return f'<span class="similarity-high">{similarity:.1%}</span>'
    elif similarity >= 0.4:
        return f'<span class="similarity-medium">{similarity:.1%}</span>'
    else:
        return f'<span class="similarity-low">{similarity:.1%}</span>'

# 매니저 초기화
@st.cache_resource
def init_managers():
    check_secrets()
    bid_manager = EnhancedBidManager()
    vector_engine = VectorSearchEngine()
    chatbot = EnhancedAIChatbot()
    return bid_manager, vector_engine, chatbot

# 챗봇 질문 처리
def process_question(question: str, chatbot: EnhancedAIChatbot, bid_manager: EnhancedBidManager, vector_engine: VectorSearchEngine):
    """질문 처리 및 응답 생성 (향상된 버전)"""
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_messages.append({"role": "user", "content": question})
    
    # 응답 생성
    with st.spinner("AI가 다양한 방법으로 관련 공고를 찾고 있습니다..."):
        # 키워드 검색
        keyword_results = bid_manager.search_bids(question)
        
        # 시맨틱 검색
        semantic_results = []
        if vector_engine.is_loaded and keyword_results:
            semantic_results = vector_engine.semantic_search(question, keyword_results, top_k=5)
        
        # AI 응답 생성
        response = chatbot.get_response(question, keyword_results, semantic_results)
    
    # 응답 표시
    with st.chat_message("assistant"):
        st.markdown(response)
        
        # 검색 결과 요약 표시
        if semantic_results:
            with st.expander("🔍 AI 유사도 검색 결과 상세"):
                for i, result in enumerate(semantic_results):
                    bid = result['document']
                    similarity = result['similarity']
                    st.markdown(f"**{i+1}. {bid.get('bidntcenm', '제목없음')}**")
                    st.markdown(f"유사도: {format_similarity(similarity)}", unsafe_allow_html=True)
                    st.caption(f"기관: {bid.get('ntceinsttm', 'N/A')} | 분류: {bid.get('bsnsdivnm', 'N/A')}")
                    st.divider()
    
    st.session_state.chat_messages.append({"role": "assistant", "content": response})

# 메인 애플리케이션
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI 입찰 공고 검색 서비스</h1>
        <p>키워드 검색 + AI 시맨틱 검색 + 지능형 상담까지!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 매니저 초기화
    bid_manager, vector_engine, chatbot = init_managers()
    
    # 탭 생성 (AI 검색 탭 추가)
    tab1, tab2, tab3, tab4 = st.tabs(["📢 실시간 입찰 공고", "🔍 키워드 검색", "🎯 AI 시맨틱 검색", "🤖 AI 상담"])
    
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
        st.subheader("🔍 키워드 기반 검색")
        
        # 검색 UI
        keyword = st.text_input("검색어를 입력하세요", placeholder="예: AI, 소프트웨어, 서버 등")
        
        if st.button("검색", type="primary", key="keyword_search"):
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
        st.subheader("🎯 AI 시맨틱 검색")
        st.info("💡 AI가 의미를 이해하여 관련성 높은 공고를 찾아드립니다!")
        
        if not vector_engine.is_loaded:
            st.warning("⚠️ 벡터 검색 엔진이 로드되지 않았습니다. 키워드 검색을 이용해주세요.")
        else:
            # 검색 UI
            semantic_query = st.text_input("자연어로 검색하세요", 
                                         placeholder="예: 인공지능 관련 프로젝트, 클라우드 서버 구축사업 등")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                top_k = st.selectbox("결과 수", [5, 10, 15, 20], index=1)
            
            if st.button("🎯 AI 검색", type="primary", key="semantic_search"):
                if semantic_query:
                    with st.spinner("AI가 의미를 분석하여 관련 공고를 찾고 있습니다..."):
                        # 먼저 키워드 검색으로 후보 확보
                        candidates = bid_manager.search_bids(semantic_query)
                        
                        if candidates:
                            # 시맨틱 검색 수행
                            semantic_results = vector_engine.semantic_search(semantic_query, candidates, top_k=top_k)
                            
                            if semantic_results:
                                st.success(f"AI가 찾은 관련 공고: {len(semantic_results)}건")
                                
                                # 결과 표시
                                for i, result in enumerate(semantic_results):
                                    bid = result['document']
                                    similarity = result['similarity']
                                    
                                    with st.container():
                                        col1, col2, col3 = st.columns([0.5, 3, 1])
                                        
                                        # 유사도 표시
                                        col1.markdown(format_similarity(similarity), unsafe_allow_html=True)
                                        
                                        # 공고 정보
                                        col2.markdown(f"**{bid.get('bidntcenm', '제목 없음')}**")
                                        col2.caption(f"🏢 {bid.get('ntceinsttm', '기관명 없음')} | 📁 {bid.get('bsnsdivnm', '분류 없음')}")
                                        
                                        # 금액
                                        col3.write(convert_to_won_format(bid.get('asignbdgtamt', 0)))
                                        
                                        st.divider()
                            else:
                                st.warning("의미적으로 관련된 공고를 찾지 못했습니다.")
                        else:
                            st.warning("검색 결과가 없습니다. 다른 키워드를 시도해보세요.")
                else:
                    st.warning("검색어를 입력해주세요.")
    
    with tab4:
        st.subheader("🤖 AI 입찰 상담")
        st.info("💬 AI가 키워드 검색과 시맨틱 검색을 모두 활용하여 답변해드립니다!")
        
        # 예시 질문 버튼
        st.markdown("**💡 예시 질문:**")
        example_questions = [
            "AI 개발 관련 입찰 공고가 있나요?",
            "클라우드 서버 구축 프로젝트를 찾고 있어요",
            "소프트웨어 개발 입찰의 최근 동향은?",
            "빅데이터 분석 관련 공고 추천해주세요"
        ]
        
        cols = st.columns(2)
        for idx, question in enumerate(example_questions):
            if cols[idx % 2].button(question, key=f"example_{idx}"):
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
            process_question(question, chatbot, bid_manager, vector_engine)
            st.rerun()
        
        # 사용자 입력
        if prompt := st.chat_input("입찰 관련 질문을 자유롭게 해주세요"):
            process_question(prompt, chatbot, bid_manager, vector_engine)
    
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
