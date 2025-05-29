import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Optional
import operator

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
    .workflow-step {
        background: #f8f9fa;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #007bff;
        border-radius: 5px;
    }
    .workflow-success {
        background: #d4edda;
        border-left-color: #28a745;
    }
    .workflow-warning {
        background: #fff3cd;
        border-left-color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# LangGraph 상태 정의
class BidSearchState(TypedDict):
    """입찰 공고 검색 상태"""
    query: str
    keyword_results: List[Dict]
    semantic_results: List[Dict]
    api_results: List[Dict]
    combined_results: List[Dict]
    final_answer: str
    messages: Annotated[List[str], operator.add]
    error: Optional[str]

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

# 나라장터 API 검색 클래스
class APISearchEngine:
    def __init__(self):
        try:
            config = get_app_config()
            self.service_key = config.api.service_key
            self.base_url = config.api.base_url
            self.endpoints = {
                "용역": f"{self.base_url}/getBidPblancListInfoServc",
                "물품": f"{self.base_url}/getBidPblancListInfoThng"
            }
        except Exception as e:
            st.warning(f"API 설정 실패: {e}")
            self.service_key = None
    
    def search_api(self, query, limit=10):
        """나라장터 API 검색"""
        if not self.service_key:
            return []
        
        try:
            import requests
            from urllib.parse import urlencode
            
            # 최근 30일 데이터 검색
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            all_results = []
            
            for category, endpoint in self.endpoints.items():
                try:
                    params = {
                        'serviceKey': self.service_key,
                        'pageNo': 1,
                        'numOfRows': limit,
                        'type': 'json',
                        'inqryBgnDt': start_date.strftime('%Y%m%d') + '0000',
                        'inqryEndDt': end_date.strftime('%Y%m%d') + '2359',
                        'bidNtceNm': query
                    }
                    
                    response = requests.get(endpoint, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('response', {}).get('body', {}).get('items', [])
                        
                        if items:
                            for item in items:
                                item['source'] = f'API({category})'
                            all_results.extend(items)
                
                except Exception as e:
                    continue
            
            return all_results[:limit]
            
        except Exception as e:
            return []

# LangGraph 워크플로우 노드들
def keyword_search_node(state: BidSearchState) -> BidSearchState:
    """키워드 검색 노드"""
    try:
        bid_manager = EnhancedBidManager()
        results = bid_manager.search_bids(state["query"])
        
        state["keyword_results"] = results
        state["messages"].append(f"✅ 키워드 검색 완료: {len(results)}건")
        
    except Exception as e:
        state["error"] = f"키워드 검색 오류: {str(e)}"
        state["keyword_results"] = []
    
    return state

def semantic_search_node(state: BidSearchState) -> BidSearchState:
    """시맨틱 검색 노드"""
    try:
        vector_engine = VectorSearchEngine()
        
        if vector_engine.is_loaded and state["keyword_results"]:
            results = vector_engine.semantic_search(state["query"], state["keyword_results"], top_k=10)
            state["semantic_results"] = results
            state["messages"].append(f"✅ AI 시맨틱 검색 완료: {len(results)}건")
        else:
            state["semantic_results"] = []
            state["messages"].append("⚠️ 시맨틱 검색 불가 (모델 미로드 또는 데이터 없음)")
        
    except Exception as e:
        state["error"] = f"시맨틱 검색 오류: {str(e)}"
        state["semantic_results"] = []
    
    return state

def api_search_node(state: BidSearchState) -> BidSearchState:
    """API 검색 노드"""
    try:
        api_engine = APISearchEngine()
        
        # 기존 검색 결과가 부족할 때만 API 호출
        total_existing = len(state["keyword_results"]) + len(state["semantic_results"])
        
        if total_existing < 5:
            results = api_engine.search_api(state["query"], limit=10)
            state["api_results"] = results
            state["messages"].append(f"✅ 실시간 API 검색 완료: {len(results)}건")
        else:
            state["api_results"] = []
            state["messages"].append("ℹ️ 충분한 검색 결과로 API 검색 생략")
        
    except Exception as e:
        state["error"] = f"API 검색 오류: {str(e)}"
        state["api_results"] = []
    
    return state

def combine_results_node(state: BidSearchState) -> BidSearchState:
    """결과 통합 노드"""
    try:
        combined = {}
        
        # 키워드 검색 결과 추가
        for item in state["keyword_results"]:
            bid_no = item.get("bidntceno") or item.get("bidNtceNo")
            if bid_no:
                combined[bid_no] = {
                    **item,
                    "sources": ["키워드"],
                    "relevance_score": 3
                }
        
        # 시맨틱 검색 결과 추가/병합
        for result in state["semantic_results"]:
            item = result["document"]
            bid_no = item.get("bidntceno") or item.get("bidNtceNo")
            if bid_no:
                if bid_no in combined:
                    combined[bid_no]["sources"].append("AI시맨틱")
                    combined[bid_no]["relevance_score"] += result["similarity"] * 10
                    combined[bid_no]["similarity"] = result["similarity"]
                else:
                    combined[bid_no] = {
                        **item,
                        "sources": ["AI시맨틱"],
                        "relevance_score": result["similarity"] * 10,
                        "similarity": result["similarity"]
                    }
        
        # API 검색 결과 추가
        for item in state["api_results"]:
            bid_no = item.get("bidNtceNo")
            if bid_no:
                if bid_no in combined:
                    combined[bid_no]["sources"].append("실시간API")
                    combined[bid_no]["relevance_score"] += 2
                else:
                    combined[bid_no] = {
                        **item,
                        "sources": ["실시간API"],
                        "relevance_score": 2
                    }
        
        # 관련도 순 정렬
        combined_list = list(combined.values())
        combined_list.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        state["combined_results"] = combined_list[:15]
        state["messages"].append(f"✅ 결과 통합 완료: {len(state['combined_results'])}건")
        
    except Exception as e:
        state["error"] = f"결과 통합 오류: {str(e)}"
        state["combined_results"] = []
    
    return state

def generate_final_answer_node(state: BidSearchState) -> BidSearchState:
    """최종 답변 생성 노드"""
    try:
        from langchain_openai import ChatOpenAI
        config = get_app_config()
        
        llm = ChatOpenAI(
            api_key=config.openai.api_key,
            model=config.openai.model,
            temperature=0.7
        )
        
        if not state["combined_results"]:
            state["final_answer"] = f"'{state['query']}'에 대한 입찰 공고를 찾을 수 없습니다."
            return state
        
        # 컨텍스트 구성
        context = "## 검색된 입찰 공고 정보:\n\n"
        
        for i, result in enumerate(state["combined_results"][:5]):
            sources = ", ".join(result.get("sources", []))
            context += f"**{i+1}. {result.get('bidntcenm') or result.get('bidNtceNm', '제목없음')}**\n"
            context += f"- 기관: {result.get('ntceinsttm') or result.get('ntceInsttNm', '기관없음')}\n"
            context += f"- 분류: {result.get('bsnsdivnm') or result.get('bsnsDivNm', '분류없음')}\n"
            context += f"- 검색방법: {sources}\n"
            if 'similarity' in result:
                context += f"- AI 유사도: {result['similarity']:.1%}\n"
            context += f"- 관련도: {result.get('relevance_score', 0):.1f}점\n\n"
        
        prompt = f"""
사용자 질문: {state['query']}

{context}

위 검색 결과를 바탕으로 사용자에게 도움이 되는 답변을 작성해주세요.

답변 가이드라인:
1. 가장 관련성 높은 공고들을 우선 소개
2. 각 공고의 특징과 장점을 간결하게 설명
3. 입찰 참여 시 고려사항 조언
4. 추가 확인이 필요한 사항 안내

3-4문장으로 전문적이고 친절하게 답변해주세요.
"""
        
        response = llm.invoke(prompt)
        state["final_answer"] = response.content
        state["messages"].append("✅ AI 답변 생성 완료")
        
    except Exception as e:
        state["error"] = f"답변 생성 오류: {str(e)}"
        state["final_answer"] = "답변 생성에 실패했습니다."
    
    return state

# LangGraph 워크플로우 생성
@st.cache_resource
def create_workflow():
    """LangGraph 워크플로우 생성"""
    try:
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(BidSearchState)
        
        # 노드 추가
        workflow.add_node("keyword_search", keyword_search_node)
        workflow.add_node("semantic_search", semantic_search_node)
        workflow.add_node("api_search", api_search_node)
        workflow.add_node("combine_results", combine_results_node)
        workflow.add_node("generate_answer", generate_final_answer_node)
        
        # 엣지 설정
        workflow.set_entry_point("keyword_search")
        workflow.add_edge("keyword_search", "semantic_search")
        workflow.add_edge("semantic_search", "api_search")
        workflow.add_edge("api_search", "combine_results")
        workflow.add_edge("combine_results", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    except Exception as e:
        st.error(f"워크플로우 생성 실패: {e}")
        return None

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
    workflow = create_workflow()
    return bid_manager, vector_engine, chatbot, workflow

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

# LangGraph 워크플로우 실행 함수
def run_langgraph_workflow(query: str, workflow):
    """LangGraph 워크플로우 실행"""
    if not workflow:
        return None, ["❌ 워크플로우를 사용할 수 없습니다."]
    
    try:
        initial_state = {
            "query": query,
            "keyword_results": [],
            "semantic_results": [],
            "api_results": [],
            "combined_results": [],
            "final_answer": "",
            "messages": [],
            "error": None
        }
        
        final_state = workflow.invoke(initial_state)
        return final_state, final_state.get("messages", [])
        
    except Exception as e:
        return None, [f"❌ 워크플로우 실행 오류: {str(e)}"]

# 메인 애플리케이션
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI 입찰 공고 검색 서비스</h1>
        <p>키워드 + AI 시맨틱 + 실시간 API + LangGraph 워크플로우!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 매니저 초기화
    bid_manager, vector_engine, chatbot, workflow = init_managers()
    
    # 탭 생성 (LangGraph 탭 추가)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📢 실시간 입찰 공고", 
        "🔍 키워드 검색", 
        "🎯 AI 시맨틱 검색", 
        "⚡ LangGraph 고급 검색",
        "🤖 AI 상담"
    ])
    
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
                    
                    if st.button("상세보기", key=f"tab1_detail_{i}"):
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
        
        if st.button("검색", type="primary", key="tab2_keyword_search"):
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
            
            if st.button("🎯 AI 검색", type="primary", key="tab3_semantic_search"):
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
        st.subheader("⚡ LangGraph 기반 고급 검색")
        st.info("🔥 AI 워크플로우가 키워드 + 시맨틱 + API 검색을 자동으로 조합합니다!")
        
        # 검색 UI
        langgraph_query = st.text_input("고급 검색 질의", 
                                       placeholder="예: AI 개발 프로젝트 찾아줘, 서버 구축 관련 최신 입찰은?")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            show_workflow = st.checkbox("워크플로우 과정 표시", value=True)
        
        if st.button("⚡ LangGraph 검색", type="primary", key="tab4_langgraph_search"):
            if langgraph_query:
                with st.spinner("AI 워크플로우가 다중 검색을 수행하고 있습니다..."):
                    final_state, messages = run_langgraph_workflow(langgraph_query, workflow)
                    
                    if final_state:
                        # 워크플로우 과정 표시
                        if show_workflow and messages:
                            st.markdown("### 🔄 검색 워크플로우 과정")
                            for msg in messages:
                                if "✅" in msg:
                                    st.markdown(f'<div class="workflow-step workflow-success">{msg}</div>', 
                                              unsafe_allow_html=True)
                                elif "⚠️" in msg or "ℹ️" in msg:
                                    st.markdown(f'<div class="workflow-step workflow-warning">{msg}</div>', 
                                              unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="workflow-step">{msg}</div>', 
                                              unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # AI 답변 표시
                        if final_state.get("final_answer"):
                            st.markdown("### 🤖 AI 종합 분석 결과")
                            st.success(final_state["final_answer"])
                        
                        # 통합 검색 결과 표시
                        if final_state.get("combined_results"):
                            st.markdown(f"### 📋 통합 검색 결과 ({len(final_state['combined_results'])}건)")
                            
                            for i, result in enumerate(final_state["combined_results"][:10]):
                                with st.container():
                                    col1, col2, col3, col4 = st.columns([0.5, 2.5, 1.5, 1])
                                    
                                    # 순위
                                    col1.markdown(f"**{i+1}**")
                                    
                                    # 공고 정보
                                    title = result.get('bidntcenm') or result.get('bidNtceNm', '제목 없음')
                                    org = result.get('ntceinsttm') or result.get('ntceInsttNm', '기관명 없음')
                                    col2.markdown(f"**{title}**")
                                    col2.caption(f"🏢 {org}")
                                    
                                    # 검색 소스 및 점수
                                    sources = result.get("sources", [])
                                    score = result.get("relevance_score", 0)
                                    col3.write(f"🏷️ {', '.join(sources)}")
                                    col3.caption(f"관련도: {score:.1f}점")
                                    
                                    # 금액
                                    amount = result.get('asignbdgtamt') or result.get('asignBdgtAmt', 0)
                                    col4.write(convert_to_won_format(amount))
                                    
                                    # 유사도 표시 (있는 경우)
                                    if 'similarity' in result:
                                        col4.markdown(f"유사도: {format_similarity(result['similarity'])}", 
                                                    unsafe_allow_html=True)
                                    
                                    st.divider()
                        else:
                            st.warning("검색 결과가 없습니다.")
                    else:
                        st.error("워크플로우 실행에 실패했습니다.")
            else:
                st.warning("검색어를 입력해주세요.")
    
    with tab5:
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
            if cols[idx % 2].button(question, key=f"tab5_example_{idx}"):
                st.session_state.pending_question = question
                st.rerun()
        
        if st.button("🔄 대화 초기화", key="tab5_chat_reset"):
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
        
        if st.button("닫기", key="detail_close_btn"):
            st.session_state.show_detail = False
            st.rerun()

# 세션 상태 초기화
if 'show_detail' not in st.session_state:
    st.session_state.show_detail = False

if __name__ == "__main__":
    main()

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
