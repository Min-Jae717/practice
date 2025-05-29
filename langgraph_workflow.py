from typing import TypedDict, Annotated, List, Dict, Optional, Union
import operator
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import requests
from urllib.parse import urlencode, quote_plus

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from database import BidDataManager, VectorSearchManager, convert_to_won_format
from ai_search import SemanticSearchEngine

# OpenAI API 키
OPENAI_API_KEY = "your-openai-api-key"  # 실제 API 키로 변경

# 나라장터 API 설정
API_KEY = "your-narajangter-api-key"  # 실제 API 키로 변경
BASE_URL_COMMON = "http://apis.data.go.kr/1230000/ad/BidPublicInfoService"
API_ENDPOINTS = {
    "공사": f"{BASE_URL_COMMON}/getBidPblancListInfoCnstwk",
    "용역": f"{BASE_URL_COMMON}/getBidPblancListInfoServc", 
    "물품": f"{BASE_URL_COMMON}/getBidPblancListInfoThng",
    "외자": f"{BASE_URL_COMMON}/getBidPblancListInfoFrgcpt",
}

# ========== LangGraph 상태 정의 ==========
class BidSearchState(TypedDict):
    """입찰 공고 검색 상태"""
    query: str
    search_type: str
    category_filter: Optional[List[str]]
    date_range: Optional[tuple]
    supabase_results: List[Dict]
    vector_results: List[Dict]
    api_results: Dict[str, List[Dict]]
    combined_results: List[Dict]
    final_answer: str
    error: Optional[str]
    status_messages: Annotated[List[str], operator.add]
    quality_score: float
    expanded_query: str
    need_api_search: bool
    total_count: int

class HybridSearchState(TypedDict):
    """하이브리드 검색 프로세스의 상태"""
    query: str
    search_method: List[str]
    supabase_results: List[Dict]
    vector_results: List[Dict]
    api_results: Dict[str, List[Dict]]
    combined_results: List[Dict]
    final_results: List[Dict]
    summary: str
    messages: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    error: Optional[str]
    total_count: int
    need_api_search: bool

# ========== LangGraph 노드 함수들 ==========
def preprocess_query_node(state: BidSearchState) -> BidSearchState:
    """쿼리 전처리 노드"""
    try:
        query = state["query"].lower()
        
        # 입찰 관련 키워드 확장
        keyword_expansions = {
            "ai": ["인공지능", "AI", "머신러닝", "딥러닝"],
            "서버": ["서버", "서버구축", "서버시스템", "인프라"],
            "sw": ["소프트웨어", "SW", "S/W", "프로그램"],
            "hw": ["하드웨어", "HW", "H/W", "장비"],
            "시스템": ["시스템", "시스템구축", "정보시스템"],
            "개발": ["개발", "구축", "제작", "개발사업"]
        }
        
        expanded_terms = [query]
        for key, synonyms in keyword_expansions.items():
            if key in query:
                expanded_terms.extend(synonyms)
        
        state["expanded_query"] = " ".join(set(expanded_terms))
        state["status_messages"] = [f"✅ 쿼리 전처리 완료: {len(expanded_terms)}개 검색어"]
        
    except Exception as e:
        state["error"] = f"쿼리 전처리 오류: {str(e)}"
        
    return state

def search_supabase_node(state: BidSearchState) -> BidSearchState:
    """Supabase 검색 노드"""
    try:
        bid_manager = BidDataManager()
        
        # 키워드 검색
        categories = state.get("category_filter")
        start_date = None
        end_date = None
        
        if state.get("date_range"):
            start_date, end_date = state["date_range"]
            start_date = start_date.strftime("%Y%m%d") if start_date else None
            end_date = end_date.strftime("%Y%m%d") if end_date else None
        
        results = bid_manager.search_bids_by_keyword(
            state.get("expanded_query", state["query"]),
            categories,
            start_date,
            end_date
        )
        
        state["supabase_results"] = results
        state["status_messages"] = [f"✅ Supabase에서 {len(results)}개 공고 검색 완료"]
        
    except Exception as e:
        state["error"] = f"Supabase 검색 오류: {str(e)}"
        state["supabase_results"] = []
        
    return state

def search_vector_db_node(state: BidSearchState) -> BidSearchState:
    """벡터 DB 시맨틱 검색 노드"""
    try:
        semantic_engine = SemanticSearchEngine(OPENAI_API_KEY)
        
        # 시맨틱 검색 수행
        results = semantic_engine.search(
            state["query"], 
            num_results=30,
            similarity_threshold=0.3
        )
        
        # 결과 포맷팅
        vector_results = []
        for metadata, similarity in results:
            vector_results.append({
                "metadata": metadata,
                "similarity": similarity,
                "bidNtceNo": metadata.get('공고번호', 'N/A'),
                "bidNtceNm": metadata.get('공고명', 'N/A'),
                "ntceInsttNm": metadata.get('기관명', 'N/A')
            })
        
        state["vector_results"] = vector_results
        state["status_messages"] = [f"✅ 벡터 DB에서 {len(vector_results)}개 관련 문서 검색 완료"]
        
    except Exception as e:
        state["error"] = f"벡터 검색 오류: {str(e)}"
        state["vector_results"] = []
        
    return state

def combine_results_node(state: BidSearchState) -> BidSearchState:
    """결과 통합 노드"""
    try:
        supabase_results = state.get("supabase_results", [])
        vector_results = state.get("vector_results", [])
        combined_dict = {}

        # Supabase 결과 추가
        for item in supabase_results:
            bid_no = item.get("bidNtceNo")
            if bid_no:
                combined_dict[bid_no] = {
                    **item,
                    "source": "Supabase",
                    "relevance_score": 5.0
                }

        # 벡터 결과 추가/병합
        for item in vector_results:
            bid_no = item.get("bidNtceNo")
            if bid_no:
                if bid_no in combined_dict:
                    combined_dict[bid_no]["relevance_score"] += item["similarity"] * 10
                    combined_dict[bid_no]["similarity"] = item["similarity"]
                    combined_dict[bid_no]["source"] += ", Vector"
                else:
                    # Supabase에서 추가 정보 조회
                    bid_manager = BidDataManager()
                    bid_detail = bid_manager.get_bid_by_number(bid_no)
                    
                    if bid_detail:
                        raw_data = bid_detail.get('raw', {})
                        combined_dict[bid_no] = {
                            "bidNtceNo": bid_no,
                            "bidNtceNm": raw_data.get('bidNtceNm', 'N/A'),
                            "ntceInsttNm": raw_data.get('ntceInsttNm', 'N/A'),
                            "bsnsDivNm": raw_data.get('bsnsDivNm', 'N/A'),
                            "asignBdgtAmt": raw_data.get('asignBdgtAmt', 0),
                            "bidNtceDate": raw_data.get('bidNtceDate', ''),
                            "bidClseDate": raw_data.get('bidClseDate', ''),
                            "raw": raw_data,
                            "source": "Vector",
                            "relevance_score": item["similarity"] * 10,
                            "similarity": item["similarity"]
                        }

        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        state["combined_results"] = combined_results[:20]

        # 검색 품질 점수 계산
        if combined_results:
            avg_score = np.mean([r["relevance_score"] for r in combined_results[:10]])
            state["quality_score"] = float(avg_score)
        else:
            state["quality_score"] = 0.0

        state["status_messages"] = [f"✅ {len(state['combined_results'])}개 공고로 통합 완료"]

    except Exception as e:
        state["error"] = f"결과 통합 오류: {str(e)}"
        state["combined_results"] = []
        state["quality_score"] = 0.0

    return state

def enrich_with_summaries_node(state: BidSearchState) -> BidSearchState:
    """GPT 요약 추가 노드"""
    try:
        bid_manager = BidDataManager()
        
        # 각 공고에 대해 요약 정보 추가
        for result in state["combined_results"]:
            bid_no = result.get("bidNtceNo")
            if bid_no:
                summary, created_at, summary_type = bid_manager.get_bid_summary(bid_no)
                result["summary"] = summary
                result["summary_type"] = summary_type
                result["summary_created_at"] = created_at
        
        state["status_messages"] = ["✅ GPT 요약 정보 추가 완료"]
        
    except Exception as e:
        state["error"] = f"요약 정보 추가 오류: {str(e)}"
        
    return state

def generate_answer_node(state: BidSearchState) -> BidSearchState:
    """AI 답변 생성 노드"""
    try:
        if not state["combined_results"]:
            state["final_answer"] = f"'{state['query']}'에 대한 입찰 공고를 찾을 수 없습니다."
            return state
        
        # LangChain LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )
        
        # 컨텍스트 구성
        contexts = []
        for i, result in enumerate(state["combined_results"][:5]):
            bid_no = result.get("bidNtceNo", "N/A")
            title = result.get("bidNtceNm", "제목 없음")
            org = result.get("ntceInsttNm", "기관명 없음")
            summary = result.get("summary", "요약 없음")

            context_item = f"""
공고 {i+1}:
- 공고번호: {bid_no}
- 공고명: {title}
- 기관: {org}
- 요약: {summary}
"""
            contexts.append(context_item)

        context_text = "\n".join(contexts)

        # LangChain 체인 실행
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 입찰 공고 분석 전문가입니다."),
            ("human", f"다음은 입찰 공고 검색 결과입니다:\n{context_text}\n위 정보를 바탕으로 질문에 답해주세요: {state['query']}")
        ])

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "query": state["query"],
            "context": context_text
        })

        state["final_answer"] = answer
        state["status_messages"] = ["✅ AI 분석 답변 생성 완료!"]

    except Exception as e:
        state["error"] = f"AI 답변 생성 오류: {str(e)}"
        state["final_answer"] = "답변 생성에 실패했습니다."
    
    return state

def check_error(state: BidSearchState) -> str:
    """에러 체크 노드"""
    if state.get("error"):
        return "error"
    return "continue"

# ========== 하이브리드 검색 노드들 ==========
def search_supabase_hybrid_node(state: HybridSearchState) -> HybridSearchState:
    """Supabase 하이브리드 검색 노드"""
    query = state["query"]
    
    state["messages"].append(
        HumanMessage(content=f"Supabase 검색 시작: {query}")
    )
    
    try:
        bid_manager = BidDataManager()
        
        # 최근 30일 데이터 검색
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        results = bid_manager.search_bids_by_keyword(
            query, None, start_date, end_date
        )
        
        state["supabase_results"] = results
        state["search_method"].append("Supabase")
        state["messages"].append(
            AIMessage(content=f"Supabase에서 {len(results)}건 검색됨")
        )
        
    except Exception as e:
        state["messages"].append(
            AIMessage(content=f"Supabase 검색 오류: {str(e)}")
        )
        state["supabase_results"] = []
    
    return state

def search_vector_hybrid_node(state: HybridSearchState) -> HybridSearchState:
    """벡터 하이브리드 검색 노드"""
    query = state["query"]
    
    state["messages"].append(
        HumanMessage(content=f"VectorDB 시맨틱 검색 시작")
    )
    
    try:
        semantic_engine = SemanticSearchEngine(OPENAI_API_KEY)
        results = semantic_engine.search(query, num_results=30, similarity_threshold=0.03)
        
        vector_results = []
        for metadata, similarity in results:
            if similarity >= 0.03:
                vector_results.append({
                    "metadata": metadata,
                    "similarity": similarity,
                    "bidNtceNo": metadata.get('공고번호', 'N/A'),
                    "bidNtceNm": metadata.get('공고명', 'N/A'),
                    "ntceInsttNm": metadata.get('기관명', 'N/A')
                })
        
        state["vector_results"] = vector_results
        state["search_method"].append("VectorDB")
        state["messages"].append(
            AIMessage(content=f"VectorDB에서 {len(vector_results)}건 검색됨")
        )
        
    except Exception as e:
        state["messages"].append(
            AIMessage(content=f"VectorDB 검색 오류: {str(e)}")
        )
        state["vector_results"] = []
    
    return state

def check_need_api_hybrid_node(state: HybridSearchState) -> HybridSearchState:
    """API 검색 필요 여부 확인"""
    supabase_count = len(state["supabase_results"])
    vector_count = len(state["vector_results"])
    total_count = supabase_count + vector_count
    
    state["messages"].append(
        HumanMessage(content=f"검색 결과 확인: Supabase {supabase_count}건, VectorDB {vector_count}건")
    )
    
    if total_count < 10:
        state["need_api_search"] = True
        state["messages"].append(
            AIMessage(content=f"검색 결과 부족 ({total_count}건), API 검색 필요")
        )
    else:
        state["need_api_search"] = False
        state["messages"].append(
            AIMessage(content=f"충분한 검색 결과 ({total_count}건), API 검색 불필요")
        )
    
    return state

def fetch_naratang_api_hybrid_node(state: HybridSearchState) -> HybridSearchState:
    """나라장터 API 호출"""
    if not state["need_api_search"]:
        state["api_results"] = {}
        return state
    
    query = state["query"]
    
    state["messages"].append(
        HumanMessage(content=f"나라장터 API 실시간 검색 시작")
    )
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    all_results = {}
    api_total = 0
    
    for category, endpoint in API_ENDPOINTS.items():
        try:
            state["messages"].append(
                AIMessage(content=f"{category} 카테고리 API 호출 중...")
            )
            
            params = {
                'serviceKey': API_KEY,
                'pageNo': 1,
                'numOfRows': 20,
                'inqryDiv': 1,
                'type': 'json',
                'inqryBgnDt': start_date.strftime('%Y%m%d') + '0000',
                'inqryEndDt': end_date.strftime('%Y%m%d') + '2359',
                'bidNtceNm': query
            }
            
            query_string = urlencode(params, quote_via=quote_plus)
            url = f"{endpoint}?{query_string}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('response', {}).get('body', {}).get('items', [])
                
                if items:
                    all_results[category] = items
                    api_total += len(items)
                    state["messages"].append(
                        AIMessage(content=f"{category}: {len(items)}건 검색 완료")
                    )
                
        except Exception as e:
            state["messages"].append(
                AIMessage(content=f"{category} API 오류: {str(e)}")
            )
    
    state["api_results"] = all_results
    state["search_method"].append("API")
    state["messages"].append(
        AIMessage(content=f"API 검색 완료: 총 {api_total}건")
    )
    
    return state

def combine_hybrid_results_node(state: HybridSearchState) -> HybridSearchState:
    """하이브리드 검색 결과 통합"""
    state["messages"].append(
        HumanMessage(content="검색 결과 통합 중...")
    )
    
    combined_dict = {}
    
    # Supabase 결과 추가
    for item in state["supabase_results"]:
        bid_no = item.get("bidNtceNo")
        if bid_no:
            combined_dict[bid_no] = {
                **item,
                "source": "Supabase",
                "relevance_score": 5
            }
    
    # VectorDB 결과 추가/병합
    for item in state["vector_results"]:
        bid_no = item.get("bidNtceNo")
        if bid_no:
            if bid_no in combined_dict:
                combined_dict[bid_no]["relevance_score"] += item["similarity"] * 10
                combined_dict[bid_no]["source"] += ", VectorDB"
            else:
                # Supabase에서 추가 정보 조회
                bid_manager = BidDataManager()
                bid_detail = bid_manager.get_bid_by_number(bid_no)
                
                if bid_detail:
                    raw_data = bid_detail.get('raw', {})
                    combined_dict[bid_no] = {
                        "bidNtceNo": bid_no,
                        "bidNtceNm": raw_data.get('bidNtceNm', 'N/A'),
                        "ntceInsttNm": raw_data.get('ntceInsttNm', 'N/A'),
                        "source": "VectorDB",
                        "relevance_score": item["similarity"] * 10,
                        "raw": raw_data
                    }
    
    # API 결과 추가
    for category, items in state["api_results"].items():
        for item in items:
            bid_no = item.get("bidNtceNo")
            if bid_no:
                if bid_no in combined_dict:
                    combined_dict[bid_no]["relevance_score"] += 3
                    combined_dict[bid_no]["source"] += ", API"
                else:
                    combined_dict[bid_no] = {
                        **item,
                        "source": f"API({category})",
                        "relevance_score": 3,
                        "category": category
                    }
    
    combined_results = list(combined_dict.values())
    combined_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    state["combined_results"] = combined_results
    state["total_count"] = len(combined_results)
    
    state["messages"].append(
        AIMessage(content=f"통합 완료: 총 {len(combined_results)}개")
    )
    
    return state

def generate_hybrid_summary_node(state: HybridSearchState) -> HybridSearchState:
    """하이브리드 검색 요약 생성"""
    if not state["combined_results"]:
        state["summary"] = "검색 결과가 없습니다."
        state["final_results"] = []
        return state
    
    state["messages"].append(
        HumanMessage(content="AI 요약 생성 중...")
    )
    
    state["final_results"] = state["combined_results"][:20]
    
    # 소스별 통계
    source_stats = {}
    for item in state["final_results"]:
        sources = item.get("source", "").split(", ")
        for source in sources:
            source_stats[source] = source_stats.get(source, 0) + 1
    
    # 요약 생성
    summary_parts = [
        f"🔍 '{state['query']}' 검색 결과: 총 {state['total_count']}건",
        f"\n📊 검색 소스:"
    ]
    
    for source, count in source_stats.items():
        summary_parts.append(f"  • {source}: {count}건")
    
    # 상위 3개 공고 하이라이트
    summary_parts.append(f"\n🏆 상위 공고:")
    for i, item in enumerate(state["final_results"][:3], 1):
        summary_parts.append(
            f"{i}. {item.get('bidNtceNm', '제목 없음')[:40]}..."
            f" ({item.get('ntceInsttNm', 'N/A')}, {convert_to_won_format(item.get('asignBdgtAmt', 0))})"
        )
    
    state["summary"] = "\n".join(summary_parts)
    
    state["messages"].append(
        AIMessage(content="요약 생성 완료")
    )
    
    return state

def should_search_api_hybrid(state: HybridSearchState) -> str:
    """API 검색 필요 여부 판단"""
    if state["need_api_search"]:
        return "search_api"
    else:
        return "combine"

# ========== LangGraph 워크플로우 구성 ==========
def create_bid_search_workflow():
    """입찰 공고 검색 워크플로우 생성"""
    workflow = StateGraph(BidSearchState)
    
    # 노드 추가
    workflow.add_node("preprocess_query", preprocess_query_node)
    workflow.add_node("search_supabase", search_supabase_node)
    workflow.add_node("search_vector_db", search_vector_db_node)
    workflow.add_node("combine_results", combine_results_node)
    workflow.add_node("enrich_summaries", enrich_with_summaries_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # 엣지 추가
    workflow.set_entry_point("preprocess_query")
    
    # 조건부 엣지
    workflow.add_conditional_edges(
        "preprocess_query",
        check_error,
        {
            "continue": "search_supabase",
            "error": END
        }
    )
    
    workflow.add_edge("search_supabase", "search_vector_db")
    workflow.add_edge("search_vector_db", "combine_results")
    workflow.add_edge("combine_results", "enrich_summaries")
    workflow.add_edge("enrich_summaries", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

def create_hybrid_search_workflow():
    """하이브리드 검색 워크플로우 생성"""
    workflow = StateGraph(HybridSearchState)
    
    # 노드 추가
    workflow.add_node("search_supabase", search_supabase_hybrid_node)
    workflow.add_node("search_vector", search_vector_hybrid_node)
    workflow.add_node("check_need_api", check_need_api_hybrid_node)
    workflow.add_node("search_api", fetch_naratang_api_hybrid_node)
    workflow.add_node("combine", combine_hybrid_results_node)
    workflow.add_node("generate_summary", generate_hybrid_summary_node)
    
    # 엣지 설정
    workflow.set_entry_point("search_supabase")
    workflow.add_edge("search_supabase", "search_vector")
    workflow.add_edge("search_vector", "check_need_api")
    
    # 조건부 엣지
    workflow.add_conditional_edges(
        "check_need_api",
        should_search_api_hybrid,
        {
            "search_api": "search_api",
            "combine": "combine"
        }
    )
    
    workflow.add_edge("search_api", "combine")
    workflow.add_edge("combine", "generate_summary")
    workflow.add_edge("generate_summary", END)
    
    return workflow.compile()
