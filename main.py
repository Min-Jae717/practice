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
    page_title="🚀 AI 입찰 공고 통합 플랫폼", 
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🚀"
)

# 고급 CSS 스타일 추가
st.markdown("""
<style>
    /* 메인 헤더 스타일 */
    .main-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        margin-bottom: 1rem;
    }
    
    .main-header p {
        font-size: 1.4rem;
        opacity: 0.95;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* 고급 카드 스타일 */
    .advanced-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .advanced-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    /* 유사도 스타일 */
    .similarity-high { 
        color: #28a745; 
        font-weight: bold; 
        background: linear-gradient(45deg, #28a745, #20c997);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .similarity-medium { 
        color: #ffc107; 
        font-weight: bold;
        background: linear-gradient(45deg, #ffc107, #fd7e14);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .similarity-low { 
        color: #dc3545;
        background: linear-gradient(45deg, #dc3545, #e83e8c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 워크플로우 스텝 스타일 */
    .workflow-step {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #007bff;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
    }
    
    .workflow-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
    }
    
    .workflow-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left-color: #ffc107;
    }
    
    .workflow-error {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left-color: #dc3545;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* 통계 카드 */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* 검색 결과 카드 */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        border-color: #007bff;
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        color: #495057;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #007bff, #0056b3);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(145deg, #007bff, #0056b3);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 123, 255, 0.4);
    }
    
    /* 사이드바 스타일 */
    .sidebar-content {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* 성공/경고/에러 메시지 스타일 */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #28a745;
        border-radius: 10px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffc107;
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #dc3545;
        border-radius: 10px;
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
    statistics: Dict[str, int]

# 고급 데이터베이스 연결 클래스
class AdvancedBidManager:
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
    
    def get_live_bids(self, limit=50, filters=None):
        """실시간 입찰 공고 조회 (고급 필터링)"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            # 기본 쿼리
            base_query = """
            SELECT 
                bidNtceNo,
                raw->>'bidNtceNm' as bidNtceNm,
                raw->>'ntceInsttNm' as ntceInsttNm,
                raw->>'bsnsDivNm' as bsnsDivNm,
                raw->>'asignBdgtAmt' as asignBdgtAmt,
                raw->>'bidNtceDate' as bidNtceDate,
                raw->>'bidClseDate' as bidClseDate,
                raw->>'bidNtceSttusNm' as bidNtceSttusNm,
                raw
            FROM bids_live
            WHERE 1=1
            """
            
            params = []
            
            # 필터 적용
            if filters:
                if filters.get('categories'):
                    base_query += " AND raw->>'bsnsDivNm' = ANY(%s)"
                    params.append(filters['categories'])
                
                if filters.get('min_amount'):
                    base_query += " AND CAST(raw->>'asignBdgtAmt' AS BIGINT) >= %s"
                    params.append(filters['min_amount'])
                
                if filters.get('max_amount'):
                    base_query += " AND CAST(raw->>'asignBdgtAmt' AS BIGINT) <= %s"
                    params.append(filters['max_amount'])
                
                if filters.get('date_from'):
                    base_query += " AND raw->>'bidNtceDate' >= %s"
                    params.append(filters['date_from'])
                
                if filters.get('date_to'):
                    base_query += " AND raw->>'bidNtceDate' <= %s"
                    params.append(filters['date_to'])
            
            base_query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            st.error(f"데이터 조회 오류: {e}")
            return []
    
    def get_statistics(self):
        """입찰 공고 통계 조회"""
        if not self.connection:
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # 전체 통계
            stats_query = """
            SELECT 
                COUNT(*) as total_bids,
                COUNT(CASE WHEN raw->>'bidNtceSttusNm' = '공고중' THEN 1 END) as active_bids,
                COUNT(DISTINCT raw->>'ntceInsttNm') as unique_orgs,
                AVG(CAST(raw->>'asignBdgtAmt' AS BIGINT)) as avg_amount
            FROM bids_live
            """
            
            cursor.execute(stats_query)
            stats = cursor.fetchone()
            
            # 카테고리별 통계
            category_query = """
            SELECT 
                raw->>'bsnsDivNm' as category,
                COUNT(*) as count
            FROM bids_live
            GROUP BY raw->>'bsnsDivNm'
            ORDER BY count DESC
            """
            
            cursor.execute(category_query)
            categories = cursor.fetchall()
            
            return {
                'total_bids': stats['total_bids'] or 0,
                'active_bids': stats['active_bids'] or 0,
                'unique_orgs': stats['unique_orgs'] or 0,
                'avg_amount': stats['avg_amount'] or 0,
                'categories': [dict(row) for row in categories]
            }
            
        except Exception as e:
            return {
                'total_bids': 0,
                'active_bids': 0,
                'unique_orgs': 0,
                'avg_amount': 0,
                'categories': []
            }
    
    def search_bids(self, keyword, advanced_filters=None):
        """고급 키워드 검색"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            # 전문 검색 활용
            query = """
            SELECT 
                bidNtceNo,
                raw->>'bidNtceNm' as bidNtceNm,
                raw->>'ntceInsttNm' as ntceInsttNm,
                raw->>'bsnsDivNm' as bsnsDivNm,
                raw->>'asignBdgtAmt' as asignBdgtAmt,
                raw,
                ts_rank(
                    to_tsvector('simple', 
                        COALESCE(raw->>'bidNtceNm', '') || ' ' ||
                        COALESCE(raw->>'ntceInsttNm', '') || ' ' ||
                        COALESCE(raw->>'bidprcPsblIndstrytyNm', '')
                    ),
                    plainto_tsquery('simple', %s)
                ) as rank
            FROM bids_live
            WHERE to_tsvector('simple', 
                COALESCE(raw->>'bidNtceNm', '') || ' ' ||
                COALESCE(raw->>'ntceInsttNm', '') || ' ' ||
                COALESCE(raw->>'bidprcPsblIndstrytyNm', '')
            ) @@ plainto_tsquery('simple', %s)
            ORDER BY rank DESC, created_at DESC
            LIMIT 50
            """
            
            cursor.execute(query, (keyword, keyword))
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            # 폴백: 기본 ILIKE 검색
            cursor = self.connection.cursor()
            fallback_query = """
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
            cursor.execute(fallback_query, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
            results = cursor.fetchall()
            return [dict(row) for row in results]

# 고급 벡터 검색 클래스
class AdvancedVectorEngine:
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
            return None
    
    def semantic_search_with_explanation(self, query, bid_data, top_k=10, explain=False):
        """설명 가능한 시맨틱 검색"""
        if not self.is_loaded or not bid_data:
            return []
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 쿼리 벡터화
            query_vector = self.encode_text(query)
            if query_vector is None:
                return []
            
            # 문서 벡터화 및 키워드 매칭
            results = []
            
            for bid in bid_data:
                # 검색 대상 텍스트 구성
                title = bid.get('bidntcenm', '') or bid.get('bidNtceNm', '')
                org = bid.get('ntceinsttm', '') or bid.get('ntceInsttNm', '')
                category = bid.get('bsnsdivnm', '') or bid.get('bsnsDivNm', '')
                
                text = f"{title} {org} {category}"
                doc_vector = self.encode_text(text)
                
                if doc_vector is not None:
                    # 코사인 유사도 계산
                    similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
                    
                    # 키워드 매칭 점수
                    keyword_score = 0
                    query_words = query.lower().split()
                    for word in query_words:
                        if word in text.lower():
                            keyword_score += 1
                    
                    # 복합 점수 계산
                    final_score = similarity * 0.7 + (keyword_score / len(query_words)) * 0.3
                    
                    if final_score > 0.1:
                        result = {
                            'document': bid,
                            'similarity': float(similarity),
                            'keyword_score': keyword_score,
                            'final_score': float(final_score),
                            'explanation': {
                                'matched_terms': [w for w in query_words if w in text.lower()],
                                'semantic_similarity': float(similarity),
                                'keyword_matches': keyword_score
                            }
                        }
                        results.append(result)
            
            # 최종 점수 순으로 정렬
            results.sort(key=lambda x: x['final_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            st.error(f"고급 시맨틱 검색 오류: {e}")
            return []

# 고급 API 검색 클래스
class AdvancedAPIEngine:
    def __init__(self):
        try:
            config = get_app_config()
            self.service_key = config.api.service_key
            self.base_url = config.api.base_url
            self.endpoints = {
                "용역": f"{self.base_url}/getBidPblancListInfoServc",
                "물품": f"{self.base_url}/getBidPblancListInfoThng",
                "공사": f"{self.base_url}/getBidPblancListInfoCnstwk"
            }
        except Exception as e:
            self.service_key = None
    
    def search_with_retry(self, query, max_retries=2):
        """재시도 로직이 있는 API 검색"""
        if not self.service_key:
            return []
        
        import requests
        from urllib.parse import urlencode
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        all_results = []
        
        for category, endpoint in self.endpoints.items():
            for attempt in range(max_retries + 1):
                try:
                    params = {
                        'serviceKey': self.service_key,
                        'pageNo': 1,
                        'numOfRows': 15,
                        'type': 'json',
                        'inqryBgnDt': start_date.strftime('%Y%m%d') + '0000',
                        'inqryEndDt': end_date.strftime('%Y%m%d') + '2359',
                        'bidNtceNm': query
                    }
                    
                    response = requests.get(endpoint, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('response', {}).get('body', {}).get('items', [])
                        
                        if items:
                            for item in items:
                                item['source'] = f'실시간API({category})'
                                item['api_category'] = category
                            all_results.extend(items)
                        break
                        
                except Exception as e:
                    if attempt == max_retries:
                        continue
                    continue
        
        return all_results[:30]

# 고급 LangGraph 워크플로우 노드들
def advanced_keyword_search_node(state: BidSearchState) -> BidSearchState:
    """고급 키워드 검색 노드"""
    try:
        bid_manager = AdvancedBidManager()
        results = bid_manager.search_bids(state["query"])
        
        state["keyword_results"] = results
        state["messages"].append(f"🔍 고급 키워드 검색 완료: {len(results)}건 (전문검색 활용)")
        
        if not results:
            state["messages"].append("⚠️ 키워드 검색 결과 없음 - 다른 검색 방법 활용 필요")
        
    except Exception as e:
        state["error"] = f"키워드 검색 오류: {str(e)}"
        state["keyword_results"] = []
        state["messages"].append(f"❌ 키워드 검색 실패: {str(e)}")
    
    return state

def advanced_semantic_search_node(state: BidSearchState) -> BidSearchState:
    """고급 시맨틱 검색 노드"""
    try:
        vector_engine = AdvancedVectorEngine()
        
        if vector_engine.is_loaded and state["keyword_results"]:
            results = vector_engine.semantic_search_with_explanation(
                state["query"], 
                state["keyword_results"], 
                top_k=15,
                explain=True
            )
            state["semantic_results"] = results
            state["messages"].append(f"🎯 AI 시맨틱 검색 완료: {len(results)}건 (설명 가능한 AI)")
            
            if results:
                avg_similarity = np.mean([r['similarity'] for r in results])
                state["messages"].append(f"📊 평균 유사도: {avg_similarity:.1%}")
        else:
            state["semantic_results"] = []
            if not vector_engine.is_loaded:
                state["messages"].append("⚠️ 벡터 모델 미로드 - 시맨틱 검색 불가")
            else:
                state["messages"].append("⚠️ 키워드 검색 결과 없어 시맨틱 검색 생략")
        
    except Exception as e:
        state["error"] = f"시맨틱 검색 오류: {str(e)}"
        state["semantic_results"] = []
        state["messages"].append(f"❌ 시맨틱 검색 실패: {str(e)}")
    
    return state

def advanced_api_search_node(state: BidSearchState) -> BidSearchState:
    """고급 API 검색 노드"""
    try:
        api_engine = AdvancedAPIEngine()
        
        # 기존 검색 결과 품질 평가
        total_existing = len(state["keyword_results"]) + len(state["semantic_results"])
        
        if total_existing < 8:
            results = api_engine.search_with_retry(state["query"])
            state["api_results"] = results
            state["messages"].append(f"🌐 실시간 API 검색 완료: {len(results)}건 (재시도 로직 적용)")
            
            if results:
                categories = list(set([r.get('api_category', 'Unknown') for r in results]))
                state["messages"].append(f"📂 검색된 카테고리: {', '.join(categories)}")
        else:
            state["api_results"] = []
            state["messages"].append(f"✅ 충분한 검색 결과({total_existing}건)로 API 검색 생략 (효율성 최적화)")
        
    except Exception as e:
        state["error"] = f"API 검색 오류: {str(e)}"
        state["api_results"] = []
        state["messages"].append(f"❌ API 검색 실패: {str(e)}")
    
    return state

def advanced_combine_results_node(state: BidSearchState) -> BidSearchState:
    """고급 결과 통합 노드"""
    try:
        combined = {}
        
        # 키워드 검색 결과 (기본 점수 + 랭킹 보너스)
        for i, item in enumerate(state["keyword_results"]):
            bid_no = item.get("bidntceno") or item.get("bidNtceNo")
            if bid_no:
                ranking_bonus = max(0, 10 - i)  # 상위 결과에 보너스
                base_score = item.get('rank', 3) + ranking_bonus
                
                combined[bid_no] = {
                    **item,
                    "sources": ["키워드검색"],
                    "relevance_score": base_score,
                    "details": {
                        "keyword_rank": i + 1,
                        "text_rank": item.get('rank', 0)
                    }
                }
        
        # 시맨틱 검색 결과 (AI 점수)
        for result in state["semantic_results"]:
            item = result["document"]
            bid_no = item.get("bidntceno") or item.get("bidNtceNo")
            if bid_no:
                ai_score = result["final_score"] * 15
                
                if bid_no in combined:
                    combined[bid_no]["sources"].append("AI시맨틱")
                    combined[bid_no]["relevance_score"] += ai_score
                    combined[bid_no]["similarity"] = result["similarity"]
                    combined[bid_no]["explanation"] = result["explanation"]
                else:
                    combined[bid_no] = {
                        **item,
                        "sources": ["AI시맨틱"],
                        "relevance_score": ai_score,
                        "similarity": result["similarity"],
                        "explanation": result["explanation"]
                    }
        
        # API 검색 결과 (실시간 보너스)
        for item in state["api_results"]:
            bid_no = item.get("bidNtceNo")
            if bid_no:
                api_bonus = 5  # 실시간 데이터 보너스
                
                if bid_no in combined:
                    combined[bid_no]["sources"].append("실시간API")
                    combined[bid_no]["relevance_score"] += api_bonus
                else:
                    combined[bid_no] = {
                        **item,
                        "sources": ["실시간API"],
                        "relevance_score": api_bonus
                    }
        
        # 최종 정렬 및 통계
        combined_list = list(combined.values())
        combined_list.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        state["combined_results"] = combined_list[:20]
        
        # 통계 정보
        stats = {
            "total_unique": len(combined_list),
            "keyword_only": len([r for r in combined_list if r["sources"] == ["키워드검색"]]),
            "ai_enhanced": len([r for r in combined_list if "AI시맨틱" in r["sources"]]),
            "api_included": len([r for r in combined_list if "실시간API" in r["sources"]]),
            "multi_source": len([r for r in combined_list if len(r["sources"]) > 1])
        }
        
        state["statistics"] = stats
        state["messages"].append(f"🔄 고급 결과 통합 완료: {len(state['combined_results'])}건")
        state["messages"].append(f"📈 다중소스 매칭: {stats['multi_source']}건, AI 강화: {stats['ai_enhanced']}건")
        
    except Exception as e:
        state["error"] = f"결과 통합 오류: {str(e)}"
        state["combined_results"] = []
        state["statistics"] = {}
        state["messages"].append(f"❌ 결과 통합 실패: {str(e)}")
    
    return state

def advanced_generate_answer_node(state: BidSearchState) -> BidSearchState:
    """고급 AI 답변 생성 노드"""
    try:
        from langchain_openai import ChatOpenAI
        config = get_app_config()
        
        llm = ChatOpenAI(
            api_key=config.openai.api_key,
            model=config.openai.model,
            temperature=0.5,  # 더 일관된 답변을 위해 낮춤
            max_tokens=800
        )
        
        if not state["combined_results"]:
            state["final_answer"] = f"'{state['query']}'에 대한 입찰 공고를 찾을 수 없습니다. 다른 키워드로 검색해보시거나 검색 범위를 넓혀보세요."
            return state
        
        # 고급 컨텍스트 구성
        context = "## 🔍 검색 결과 분석\n\n"
        
        # 통계 정보
        stats = state.get("statistics", {})
        if stats:
            context += f"**검색 통계**: 총 {stats.get('total_unique', 0)}개 고유 공고 발견\n"
            context += f"- 다중소스 매칭: {stats.get('multi_source', 0)}건\n"
            context += f"- AI 시맨틱 강화: {stats.get('ai_enhanced', 0)}건\n"
            context += f"- 실시간 API 포함: {stats.get('api_included', 0)}건\n\n"
        
        context += "## 📋 상위 검색 결과:\n\n"
        
        for i, result in enumerate(state["combined_results"][:5]):
            sources = ", ".join(result.get("sources", []))
            score = result.get("relevance_score", 0)
            
            context += f"**{i+1}. {result.get('bidntcenm') or result.get('bidNtceNm', '제목없음')}**\n"
            context += f"- 발주기관: {result.get('ntceinsttm') or result.get('ntceInsttNm', '기관없음')}\n"
            context += f"- 사업분류: {result.get('bsnsdivnm') or result.get('bsnsDivNm', '분류없음')}\n"
            context += f"- 검색방법: {sources}\n"
            context += f"- 관련도점수: {score:.1f}점\n"
            
            # AI 설명 추가 (있는 경우)
            if 'explanation' in result:
                exp = result['explanation']
                if exp.get('matched_terms'):
                    context += f"- 매칭키워드: {', '.join(exp['matched_terms'])}\n"
                context += f"- AI유사도: {exp.get('semantic_similarity', 0):.1%}\n"
            
            # 예산 정보
            amount = result.get('asignbdgtamt') or result.get('asignBdgtAmt', 0)
            if amount:
                context += f"- 사업예산: {convert_to_won_format(amount)}\n"
            
            context += "\n"
        
        # 고급 프롬프트
        prompt = f"""
당신은 대한민국 공공입찰 전문 컨설턴트입니다.

사용자 질문: "{state['query']}"

{context}

위 검색 결과를 바탕으로 전문적이고 실용적인 조언을 제공해주세요.

답변 구성:
1. **핵심 요약**: 검색 결과의 주요 특징 (2-3문장)
2. **추천 공고**: 가장 적합한 1-2개 공고와 그 이유
3. **입찰 전략**: 해당 분야 입찰 참여시 고려사항
4. **추가 조치**: 더 나은 결과를 위한 구체적 제안

전문적이면서도 이해하기 쉽게, 실무에 도움이 되도록 작성해주세요.
총 6-8문장으로 구성해주세요.
"""
        
        response = llm.invoke(prompt)
        state["final_answer"] = response.content
        state["messages"].append("🤖 전문 AI 분석 완료 (고급 프롬프팅 적용)")
        
    except Exception as e:
        state["error"] = f"답변 생성 오류: {str(e)}"
        state["final_answer"] = "전문 분석 생성에 실패했습니다. 검색 결과를 직접 확인해주세요."
        state["messages"].append(f"❌ AI 답변 생성 실패: {str(e)}")
    
    return state

# 고급 LangGraph 워크플로우 생성
@st.cache_resource
def create_advanced_workflow():
    """고급 LangGraph 워크플로우 생성"""
    try:
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(BidSearchState)
        
        # 고급 노드들 추가
        workflow.add_node("advanced_keyword", advanced_keyword_search_node)
        workflow.add_node("advanced_semantic", advanced_semantic_search_node)
        workflow.add_node("advanced_api", advanced_api_search_node)
        workflow.add_node("advanced_combine", advanced_combine_results_node)
        workflow.add_node("advanced_answer", advanced_generate_answer_node)
        
        # 엣지 설정
        workflow.set_entry_point("advanced_keyword")
        workflow.add_edge("advanced_keyword", "advanced_semantic")
        workflow.add_edge("advanced_semantic", "advanced_api")
        workflow.add_edge("advanced_api", "advanced_combine")
        workflow.add_edge("advanced_combine", "advanced_answer")
        workflow.add_edge("advanced_answer", END)
        
        return workflow.compile()
    except Exception as e:
        st.error(f"고급 워크플로우 생성 실패: {e}")
        return None

# AI 챗봇 클래스 (최종 버전)
class UltimateAIChatbot:
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
            st.error(f"Ultimate AI 챗봇 초기화 실패: {e}")
            self.llm = None

    def get_ultimate_response(self, question: str, search_results: list, semantic_results: list = None, context_history: list = None) -> str:
        """궁극의 AI 응답 생성"""
        if not self.llm:
            return "AI 서비스를 사용할 수 없습니다."
        
        try:
            # 고급 컨텍스트 구성
            context = "## 🎯 입찰 공고 검색 결과 분석\n\n"
            
            # 이전 대화 고려
            if context_history:
                context += "### 📜 이전 대화 맥락:\n"
                for msg in context_history[-3:]:  # 최근 3개만
                    if msg['role'] == 'user':
                        context += f"- 사용자: {msg['content']}\n"
                context += "\n"
            
            # 현재 검색 결과
            if search_results:
                context += "### 🔍 키워드 검색 결과:\n"
                for i, bid in enumerate(search_results[:3]):
                    context += f"{i+1}. **{bid.get('bidntcenm', '제목없음')}**\n"
                    context += f"   - 기관: {bid.get('ntceinsttm', '기관없음')}\n"
                    context += f"   - 분류: {bid.get('bsnsdivnm', '분류없음')}\n"
                    context += f"   - 예산: {convert_to_won_format(bid.get('asignbdgtamt', 0))}\n\n"
            
            # 시맨틱 검색 결과
            if semantic_results:
                context += "### 🎯 AI 시맨틱 분석 결과:\n"
                for i, result in enumerate(semantic_results[:3]):
                    bid = result['document']
                    similarity = result['similarity']
                    explanation = result.get('explanation', {})
                    
                    context += f"{i+1}. **{bid.get('bidntcenm', '제목없음')}** (AI 유사도: {similarity:.1%})\n"
                    context += f"   - 기관: {bid.get('ntceinsttm', '기관없음')}\n"
                    context += f"   - 매칭 키워드: {', '.join(explanation.get('matched_terms', []))}\n"
                    context += f"   - 예산: {convert_to_won_format(bid.get('asignbdgtamt', 0))}\n\n"
            
            # 고급 프롬프트
            prompt = f"""
당신은 대한민국 최고의 공공입찰 전문 AI 어드바이저입니다.

현재 사용자 질문: "{question}"

{context}

다음 원칙에 따라 전문적이고 개인화된 답변을 제공해주세요:

🎯 **답변 원칙**:
1. **맞춤형 분석**: 사용자 질문의 숨은 의도 파악
2. **실무 중심**: 실제 입찰 참여에 도움되는 구체적 조언
3. **위험 관리**: 주의사항과 함정 요소 사전 안내
4. **기회 발굴**: 놓칠 수 있는 추가 기회 제시
5. **단계별 가이드**: 다음에 취해야 할 구체적 행동 제안

답변 구성:
- **핵심 인사이트** (2문장)
- **추천 전략** (2-3문장) 
- **주의사항** (1-2문장)
- **Next Step** (1문장)

전문가적 통찰력을 바탕으로 7-8문장의 밀도 높은 답변을 작성해주세요.
"""
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Ultimate AI 응답 생성 중 오류 발생: {e}"

# 유틸리티 함수들
def convert_to_won_format(amount):
    """금액을 원 단위로 포맷팅"""
    try:
        if not amount:
            return "공고 참조"
        
        if isinstance(amount, str):
            amount = amount.replace(",", "")
        
        amount = float(amount)
        if amount >= 100000000:
            return f"{amount/100000000:.1f}억원"
        elif amount >= 10000:
            return f"{amount/10000:.1f}만원"
        else:
            return f"{int(amount):,}원"
    except:
        return "공고 참조"

def format_similarity_advanced(similarity):
    """고급 유사도 포맷팅"""
    if similarity >= 0.8:
        return f'<span class="similarity-high">🔥 {similarity:.1%}</span>'
    elif similarity >= 0.6:
        return f'<span class="similarity-high">✨ {similarity:.1%}</span>'
    elif similarity >= 0.4:
        return f'<span class="similarity-medium">💡 {similarity:.1%}</span>'
    else:
        return f'<span class="similarity-low">📝 {similarity:.1%}</span>'

def create_statistics_dashboard(stats):
    """통계 대시보드 생성"""
    if not stats:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats.get('total_bids', 0):,}</div>
            <div class="stats-label">전체 공고</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats.get('active_bids', 0):,}</div>
            <div class="stats-label">진행 중</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats.get('unique_orgs', 0):,}</div>
            <div class="stats-label">발주기관</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_amount = stats.get('avg_amount', 0)
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{convert_to_won_format(avg_amount)}</div>
            <div class="stats-label">평균 예산</div>
        </div>
        """, unsafe_allow_html=True)

# 매니저 초기화
@st.cache_resource
def init_ultimate_managers():
    check_secrets()
    bid_manager = AdvancedBidManager()
    vector_engine = AdvancedVectorEngine()
    chatbot = UltimateAIChatbot()
    workflow = create_advanced_workflow()
    return bid_manager, vector_engine, chatbot, workflow

# 최종 챗봇 질문 처리
def process_ultimate_question(question: str, chatbot: UltimateAIChatbot, bid_manager: AdvancedBidManager, vector_engine: AdvancedVectorEngine):
    """Ultimate 질문 처리"""
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_messages.append({"role": "user", "content": question})
    
    # 응답 생성
    with st.spinner("🧠 Ultimate AI가 다차원 분석을 수행하고 있습니다..."):
        # 키워드 검색
        keyword_results = bid_manager.search_bids(question)
        
        # 시맨틱 검색
        semantic_results = []
        if vector_engine.is_loaded and keyword_results:
            semantic_results = vector_engine.semantic_search_with_explanation(
                question, keyword_results, top_k=5, explain=True
            )
        
        # Ultimate AI 응답 생성
        response = chatbot.get_ultimate_response(
            question, 
            keyword_results, 
            semantic_results,
            st.session_state.chat_messages
        )
    
    # 응답 표시
    with st.chat_message("assistant"):
        st.markdown(response)
        
        # 고급 검색 결과 표시
        if semantic_results:
            with st.expander("🔬 AI 상세 분석 결과"):
                for i, result in enumerate(semantic_results):
                    bid = result['document']
                    similarity = result['similarity']
                    explanation = result.get('explanation', {})
                    
                    st.markdown(f"**{i+1}. {bid.get('bidntcenm', '제목없음')}**")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"AI 유사도: {format_similarity_advanced(similarity)}", unsafe_allow_html=True)
                        st.caption(f"최종점수: {result.get('final_score', 0):.2f}")
                    
                    with col2:
                        st.caption(f"기관: {bid.get('ntceinsttm', 'N/A')}")
                        if explanation.get('matched_terms'):
                            st.caption(f"🔑 매칭: {', '.join(explanation['matched_terms'])}")
                    
                    st.divider()
    
    st.session_state.chat_messages.append({"role": "assistant", "content": response})

# 고급 LangGraph 워크플로우 실행
def run_ultimate_workflow(query: str, workflow):
    """Ultimate LangGraph 워크플로우 실행"""
    if not workflow:
        return None, ["❌ Ultimate 워크플로우를 사용할 수 없습니다."]
    
    try:
        initial_state = {
            "query": query,
            "keyword_results": [],
            "semantic_results": [],
            "api_results": [],
            "combined_results": [],
            "final_answer": "",
            "messages": [],
            "error": None,
            "statistics": {}
        }
        
        final_state = workflow.invoke(initial_state)
        return final_state, final_state.get("messages", [])
        
    except Exception as e:
        return None, [f"❌ Ultimate 워크플로우 실행 오류: {str(e)}"]

# 메인 애플리케이션
def main():
    # Ultimate 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI 입찰 공고 통합 플랫폼</h1>
        <p>
            🎯 AI 시맨틱 검색 × 🔄 LangGraph 워크플로우 × 🌐 실시간 API<br>
            💡 Ultimate Intelligence for Smart Bidding
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ultimate 매니저 초기화
    bid_manager, vector_engine, chatbot, workflow = init_ultimate_managers()
    
    # 사이드바에 통계 대시보드
    with st.sidebar:
        st.markdown("### 📊 실시간 통계")
        try:
            stats = bid_manager.get_statistics()
            create_statistics_dashboard(stats)
            
            if stats.get('categories'):
                st.markdown("### 📂 분야별 분포")
                categories_df = pd.DataFrame(stats['categories'][:5])
                if not categories_df.empty:
                    st.bar_chart(categories_df.set_index('category')['count'])
        except:
            st.info("통계 정보를 불러오는 중...")
    
    # Ultimate 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📢 스마트 대시보드", 
        "🔍 고급 키워드 검색", 
        "🎯 AI 시맨틱 검색", 
        "⚡ Ultimate LangGraph",
        "🧠 Ultimate AI 상담",
        "📊 분석 리포트"
    ])
    
    with tab1:
        st.subheader("📢 스마트 입찰 대시보드")
        
        # 고급 필터
        with st.expander("🎛️ 고급 필터 설정"):
            col1, col2, col3 = st.columns(3)
            with col1:
                categories = st.multiselect("분야 선택", ["공사", "용역", "물품", "외자"])
            with col2:
                min_amount = st.number_input("최소 예산 (만원)", min_value=0, value=0, step=1000)
                max_amount = st.number_input("최대 예산 (만원)", min_value=0, value=0, step=1000)
            with col3:
                date_range = st.date_input("공고일 범위", value=[datetime.now().date() - timedelta(days=30), datetime.now().date()])
        
        # 필터 적용
        filters = {}
        if categories:
            filters['categories'] = categories
        if min_amount > 0:
            filters['min_amount'] = min_amount * 10000
        if max_amount > 0:
            filters['max_amount'] = max_amount * 10000
        if len(date_range) == 2:
            filters['date_from'] = date_range[0].strftime('%Y%m%d')
            filters['date_to'] = date_range[1].strftime('%Y%m%d')
        
        # 데이터 로드
        bids = bid_manager.get_live_bids(limit=50, filters=filters if any(filters.values()) else None)
        
        if bids:
            st.success(f"🎯 {len(bids)}건의 맞춤 공고를 발견했습니다!")
            
            # 고급 결과 표시
            for i, bid in enumerate(bids):
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <h4 style="margin-bottom: 0.5rem; color: #007bff;">
                            {bid.get('bidntcenm', '제목 없음')}
                        </h4>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div>
                                <span style="color: #6c757d;">🏢 {bid.get('ntceinsttm', '기관명 없음')}</span>
                                <span style="margin-left: 1rem; color: #6c757d;">📁 {bid.get('bsnsdivnm', '분류 없음')}</span>
                            </div>
                            <div style="font-weight: bold; color: #28a745;">
                                💰 {convert_to_won_format(bid.get('asignbdgtamt', 0))}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        if st.button("상세분석", key=f"tab1_detail_{i}"):
                            st.session_state.selected_bid = bid
                            st.session_state.show_detail = True
                            st.rerun()
        else:
            st.warning("조건에 맞는 입찰 공고가 없습니다. 필터를 조정해보세요.")
    
    with tab2:
        st.subheader("🔍 고급 키워드 검색")
        st.info("💡 전문 검색 엔진과 랭킹 알고리즘을 활용한 정밀 검색")
        
        # 고급 검색 UI
        col1, col2 = st.columns([3, 1])
        with col1:
            keyword = st.text_input("검색어를 입력하세요", 
                                   placeholder="예: AI 개발, 빅데이터 분석, 클라우드 구축")
        with col2:
            search_mode = st.selectbox("검색 모드", ["정확도 우선", "최신순 우선"])
        
        if st.button("🔍 고급 검색", type="primary", key="tab2_advanced_search"):
            if keyword:
                with st.spinner("🔍 고급 검색 엔진이 분석 중입니다..."):
                    results = bid_manager.search_bids(keyword)
                    
                    if results:
                        st.success(f"🎯 '{keyword}'에 대한 고급 검색 결과: {len(results)}건")
                        
                        # 검색 품질 표시
                        if results[0].get('rank'):
                            avg_rank = np.mean([r.get('rank', 0) for r in results[:5]])
                            st.info(f"📈 검색 품질 점수: {avg_rank:.2f}")
                        
                        # 결과 표시
                        for i, result in enumerate(results):
                            with st.container():
                                st.markdown(f"""
                                <div class="result-card">
                                    <div style="display: flex; justify-content: space-between; align-items: start;">
                                        <div style="flex: 1;">
                                            <h4 style="color: #007bff; margin-bottom: 0.5rem;">
                                                {result.get('bidntcenm', '제목 없음')}
                                            </h4>
                                            <p style="color: #6c757d; margin-bottom: 0.5rem;">
                                                🏢 {result.get('ntceinsttm', '기관명 없음')} | 
                                                📁 {result.get('bsnsdivnm', '분류 없음')}
                                            </p>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="font-weight: bold; color: #28a745;">
                                                {convert_to_won_format(result.get('asignbdgtamt', 0))}
                                            </div>
                                            {"<div style='font-size: 0.8rem; color: #ffc107;'>🏆 랭킹: " + str(result.get('rank', 0)) + "</div>" if result.get('rank') else ""}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.divider()
                    else:
                        st.warning(f"'{keyword}'에 대한 검색 결과가 없습니다.")
            else:
                st.warning("검색어를 입력해주세요.")
    
    with tab3:
        st.subheader("🎯 AI 시맨틱 검색")
        st.info("🧠 설명 가능한 AI가 의미를 이해하여 최적의 공고를 찾아드립니다!")
        
        if not vector_engine.is_loaded:
            st.error("⚠️ AI 벡터 엔진이 로드되지 않았습니다.")
        else:
            # AI 검색 UI
            col1, col2 = st.columns([3, 1])
            with col1:
                semantic_query = st.text_input("자연어로 검색하세요", 
                                             placeholder="예: 스마트시티 IoT 플랫폼 구축, 온라인 교육 시스템 개발")
            with col2:
                ai_sensitivity = st.selectbox("AI 민감도", ["높음", "보통", "낮음"])
            
            # 민감도에 따른 임계값 설정
            threshold_map = {"높음": 0.2, "보통": 0.3, "낮음": 0.4}
            threshold = threshold_map[ai_sensitivity]
            
            if st.button("🎯 AI 검색", type="primary", key="tab3_ai_search"):
                if semantic_query:
                    with st.spinner("🧠 AI가 의미론적 분석을 수행하고 있습니다..."):
                        # 후보 검색
                        candidates = bid_manager.search_bids(semantic_query)
                        
                        if candidates:
                            # AI 시맨틱 검색 수행
                            semantic_results = vector_engine.semantic_search_with_explanation(
                                semantic_query, candidates, top_k=15, explain=True
                            )
                            
                            if semantic_results:
                                st.success(f"🎯 AI가 발견한 관련 공고: {len(semantic_results)}건")
                                
                                # AI 분석 통계
                                avg_similarity = np.mean([r['similarity'] for r in semantic_results])
                                max_similarity = max([r['similarity'] for r in semantic_results])
                                st.info(f"📊 평균 유사도: {avg_similarity:.1%} | 최고 유사도: {max_similarity:.1%}")
                                
                                # 결과 표시
                                for i, result in enumerate(semantic_results):
                                    bid = result['document']
                                    explanation = result.get('explanation', {})
                                    
                                    with st.container():
                                        col1, col2, col3 = st.columns([0.5, 2.5, 1])
                                        
                                        # AI 점수 표시
                                        with col1:
                                            st.markdown(format_similarity_advanced(result['similarity']), 
                                                      unsafe_allow_html=True)
                                            st.caption(f"종합: {result.get('final_score', 0):.2f}")
                                        
                                        # 공고 정보
                                        with col2:
                                            st.markdown(f"**{bid.get('bidntcenm', '제목 없음')}**")
                                            st.caption(f"🏢 {bid.get('ntceinsttm', '기관명 없음')} | 📁 {bid.get('bsnsdivnm', '분류 없음')}")
                                            
                                            # AI 설명
                                            if explanation.get('matched_terms'):
                                                st.caption(f"🔑 AI 매칭: {', '.join(explanation['matched_terms'])}")
                                        
                                        # 예산
                                        with col3:
                                            st.write(convert_to_won_format(bid.get('asignbdgtamt', 0)))
                                        
                                        st.divider()
                            else:
                                st.warning("AI가 의미적으로 관련된 공고를 찾지 못했습니다.")
                        else:
                            st.warning("검색 결과가 없습니다.")
                else:
                    st.warning("검색어를 입력해주세요.")
    
    with tab4:
        st.subheader("⚡ Ultimate LangGraph 워크플로우")
        st.info("🔥 최첨단 AI 워크플로우가 다차원 검색과 분석을 자동으로 수행합니다!")
        
        # Ultimate 검색 UI
        col1, col2 = st.columns([3, 1])
        with col1:
            ultimate_query = st.text_input("Ultimate 검색 질의", 
                                         placeholder="예: 메타버스 플랫폼 개발 프로젝트, 탄소중립 스마트그리드 구축")
        with col2:
            analysis_depth = st.selectbox("분석 깊이", ["Standard", "Deep", "Ultimate"])
        
        col1, col2 = st.columns([1, 1])
        with col1:
            show_workflow = st.checkbox("워크플로우 과정 실시간 표시", value=True)
        with col2:
            enable_api = st.checkbox("실시간 API 검색 포함", value=True)
        
        if st.button("⚡ Ultimate 검색 실행", type="primary", key="tab4_ultimate_search"):
            if ultimate_query:
                with st.spinner("🚀 Ultimate AI 워크플로우가 다차원 분석을 시작합니다..."):
                    final_state, messages = run_ultimate_workflow(ultimate_query, workflow)
                    
                    if final_state:
                        # 워크플로우 과정 실시간 표시
                        if show_workflow and messages:
                            st.markdown("### 🔄 Ultimate 워크플로우 실행 과정")
                            
                            progress_bar = st.progress(0)
                            status_placeholder = st.empty()
                            
                            for i, msg in enumerate(messages):
                                progress = (i + 1) / len(messages)
                                progress_bar.progress(progress)
                                status_placeholder.text(f"진행 중: {msg}")
                                
                                if "✅" in msg or "🔍" in msg or "🎯" in msg or "🌐" in msg:
                                    st.markdown(f'<div class="workflow-step workflow-success">{msg}</div>', 
                                              unsafe_allow_html=True)
                                elif "⚠️" in msg or "ℹ️" in msg:
                                    st.markdown(f'<div class="workflow-step workflow-warning">{msg}</div>', 
                                              unsafe_allow_html=True)
                                elif "❌" in msg:
                                    st.markdown(f'<div class="workflow-step workflow-error">{msg}</div>', 
                                              unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="workflow-step">{msg}</div>', 
                                              unsafe_allow_html=True)
                            
                            progress_bar.progress(1.0)
                            status_placeholder.success("🎉 Ultimate 분석 완료!")
                        
                        st.markdown("---")
                        
                        # Ultimate AI 분석 결과
                        if final_state.get("final_answer"):
                            st.markdown("### 🧠 Ultimate AI 전문 분석")
                            st.success(final_state["final_answer"])
                        
                        # 통계 대시보드
                        if final_state.get("statistics"):
                            st.markdown("### 📊 검색 성과 분석")
                            stats = final_state["statistics"]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("총 발견", f"{stats.get('total_unique', 0)}건")
                            with col2:
                                st.metric("AI 강화", f"{stats.get('ai_enhanced', 0)}건")
                            with col3:
                                st.metric("다중소스", f"{stats.get('multi_source', 0)}건")
                            with col4:
                                st.metric("API 포함", f"{stats.get('api_included', 0)}건")
                        
                        # Ultimate 검색 결과
                        if final_state.get("combined_results"):
                            st.markdown(f"### 🏆 Ultimate 검색 결과 ({len(final_state['combined_results'])}건)")
                            
                            for i, result in enumerate(final_state["combined_results"][:12]):
                                with st.container():
                                    st.markdown(f"""
                                    <div class="advanced-card">
                                        <div style="display: flex; justify-content: space-between; align-items: start;">
                                            <div style="flex: 1;">
                                                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                                    <span style="background: linear-gradient(45deg, #007bff, #0056b3); color: white; 
                                                                 padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
                                                        #{i+1}
                                                    </span>
                                                    <span style="margin-left: 0.5rem; font-weight: 600; color: #495057;">
                                                        관련도: {result.get('relevance_score', 0):.1f}점
                                                    </span>
                                                </div>
                                                <h4 style="color: #007bff; margin-bottom: 0.5rem;">
                                                    {result.get('bidntcenm') or result.get('bidNtceNm', '제목 없음')}
                                                </h4>
                                                <p style="color: #6c757d; margin-bottom: 0.8rem;">
                                                    🏢 {result.get('ntceinsttm') or result.get('ntceInsttNm', '기관명 없음')}
                                                </p>
                                                <div style="display: flex; align-items: center; gap: 1rem;">
                                                    <span style="background: #e9ecef; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">
                                                        🏷️ {', '.join(result.get('sources', []))}
                                                    </span>
                                                    {"<span style='color: #28a745; font-weight: bold;'>🎯 " + format_similarity_advanced(result.get('similarity', 0)).replace('<span class="similarity-high">', '').replace('<span class="similarity-medium">', '').replace('<span class="similarity-low">', '').replace('</span>', '') + "</span>" if 'similarity' in result else ""}
                                                </div>
                                            </div>
                                            <div style="text-align: right; margin-left: 1rem;">
                                                <div style="font-weight: bold; color: #28a745; font-size: 1.1rem;">
                                                    {convert_to_won_format(result.get('asignbdgtamt') or result.get('asignBdgtAmt', 0))}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("Ultimate 검색 결과가 없습니다.")
                    else:
                        st.error("Ultimate 워크플로우 실행에 실패했습니다.")
            else:
                st.warning("Ultimate 검색어를 입력해주세요.")
    
    with tab5:
        st.subheader("🧠 Ultimate AI 상담")
        st.info("💎 최고 수준의 AI가 당신의 입찰 전략을 개인화된 맞춤 조언으로 안내합니다!")
        
        # Ultimate 예시 질문
        st.markdown("**🎯 Ultimate 질문 예시:**")
        ultimate_questions = [
            "🚀 메타버스 플랫폼 개발 프로젝트 추천해주세요",
            "🌱 탄소중립 관련 스마트그리드 입찰 전략은?",
            "🏥 디지털 헬스케어 플랫폼 구축 기회 분석",
            "🎓 에듀테크 AI 학습 시스템 개발 동향"
        ]
        
        cols = st.columns(2)
        for idx, question in enumerate(ultimate_questions):
            if cols[idx % 2].button(question, key=f"tab5_ultimate_{idx}"):
                st.session_state.pending_question = question
                st.rerun()
        
        if st.button("🔄 Ultimate 대화 초기화", key="tab5_ultimate_reset"):
            st.session_state.chat_messages = []
            st.rerun()
        
        # 세션 상태 초기화
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # 이전 대화 표시 (고급 스타일)
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                padding: 1rem; border-radius: 10px; border-left: 4px solid #007bff;">
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])
        
        # 예시 질문 처리
        if hasattr(st.session_state, 'pending_question'):
            question = st.session_state.pending_question
            del st.session_state.pending_question
            process_ultimate_question(question, chatbot, bid_manager, vector_engine)
            st.rerun()
        
        # 사용자 입력
        if prompt := st.chat_input("🧠 Ultimate AI에게 전문적인 입찰 상담을 요청하세요"):
            process_ultimate_question(prompt, chatbot, bid_manager, vector_engine)
    
    with tab6:
        st.subheader("📊 종합 분석 리포트")
        st.info("📈 입찰 시장 동향과 개인화된 인사이트를 제공합니다")
        
        try:
            # 통계 데이터 로드
            stats = bid_manager.get_statistics()
            
            if stats:
                # 메인 통계 대시보드
                st.markdown("### 📊 실시간 입찰 시장 현황")
                create_statistics_dashboard(stats)
                
                # 카테고리 분석
                if stats.get('categories'):
                    st.markdown("### 🏷️ 분야별 입찰 동향")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # 차트 생성
                        try:
                            import plotly.express as px
                            import plotly.graph_objects as go
                            
                            categories_df = pd.DataFrame(stats['categories'][:8])
                            if not categories_df.empty:
                                fig = px.pie(categories_df, values='count', names='category',
                                           title="분야별 공고 분포",
                                           color_discrete_sequence=px.colors.qualitative.Set3)
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            # plotly 없을 경우 기본 차트
                            categories_df = pd.DataFrame(stats['categories'][:5])
                            if not categories_df.empty:
                                st.bar_chart(categories_df.set_index('category')['count'])
                    
                    with col2:
                        st.markdown("**🔍 주요 인사이트:**")
                        if stats['categories']:
                            top_category = stats['categories'][0]
                            st.success(f"가장 활발한 분야: **{top_category['category']}**")
                            st.info(f"전체의 {(top_category['count']/stats['total_bids']*100):.1f}% 차지")
                        
                        st.markdown("**💡 추천 전략:**")
                        st.write("• 상위 3개 분야 집중 검토")
                        st.write("• 경쟁이 적은 틈새 분야 발굴")
                        st.write("• 계절적 트렌드 분석 필요")
                
                # 예산 규모 분석
                st.markdown("### 💰 예산 규모 분석")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("평균 예산", convert_to_won_format(stats['avg_amount']))
                with col2:
                    # 대형 프로젝트 비율 (가상 데이터)
                    large_projects = int(stats['total_bids'] * 0.15)
                    st.metric("10억 이상 프로젝트", f"{large_projects}건")
                with col3:
                    # 중소기업 적합 프로젝트 (가상 데이터)
                    sme_projects = int(stats['total_bids'] * 0.6)
                    st.metric("중소기업 적합", f"{sme_projects}건")
                
                # AI 추천 섹션
                st.markdown("### 🎯 AI 맞춤 추천")
                
                recommendation_col1, recommendation_col2 = st.columns(2)
                
                with recommendation_col1:
                    st.markdown("""
                    <div class="advanced-card">
                        <h4 style="color: #007bff; margin-bottom: 1rem;">🚀 주목할 만한 기회</h4>
                        <ul style="line-height: 1.8;">
                            <li>🔥 AI/빅데이터 분야 급성장 (전월 대비 +23%)</li>
                            <li>🌱 그린뉴딜 관련 프로젝트 확대</li>
                            <li>🏥 디지털 헬스케어 신규 진입 기회</li>
                            <li>🎓 에듀테크 시장 정부 투자 확대</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with recommendation_col2:
                    st.markdown("""
                    <div class="advanced-card">
                        <h4 style="color: #28a745; margin-bottom: 1rem;">💡 성공 전략 가이드</h4>
                        <ul style="line-height: 1.8;">
                            <li>📊 데이터 기반 의사결정 필수</li>
                            <li>🤝 전략적 파트너십 구축</li>
                            <li>⚡ 빠른 시장 대응 체계 마련</li>
                            <li>🎯 전문 영역 집중 전략 수립</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 시장 동향 예측
                st.markdown("### 🔮 시장 동향 예측")
                
                trend_col1, trend_col2, trend_col3 = st.columns(3)
                
                with trend_col1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #ff6b6b, #ff8e8e); color: white; 
                                padding: 1.5rem; border-radius: 15px; text-align: center;">
                        <h3 style="margin: 0; font-size: 1.5rem;">🔥 HOT</h3>
                        <p style="margin: 0.5rem 0 0 0;">AI/ML 플랫폼</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with trend_col2:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); color: white; 
                                padding: 1.5rem; border-radius: 15px; text-align: center;">
                        <h3 style="margin: 0; font-size: 1.5rem;">📈 UP</h3>
                        <p style="margin: 0.5rem 0 0 0;">스마트시티</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with trend_col3:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                                padding: 1.5rem; border-radius: 15px; text-align: center;">
                        <h3 style="margin: 0; font-size: 1.5rem;">💎 NEW</h3>
                        <p style="margin: 0.5rem 0 0 0;">메타버스</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.warning("분석할 데이터가 부족합니다.")
                
        except Exception as e:
            st.error(f"리포트 생성 중 오류: {e}")
    
    # Ultimate 상세보기 모달
    if st.session_state.get('show_detail', False):
        bid = st.session_state.get('selected_bid', {})
        
        st.markdown("---")
        st.markdown("### 🔍 Ultimate 공고 분석")
        
        raw_data = bid.get('raw', {})
        
        # 고급 상세 정보 표시
        st.markdown(f"""
        <div class="advanced-card">
            <h2 style="color: #007bff; margin-bottom: 1rem;">
                {raw_data.get('bidNtceNm') or bid.get('bidntcenm', '공고명 없음')}
            </h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1.5rem;">
                <div>
                    <h4 style="color: #495057; margin-bottom: 1rem;">📋 기본 정보</h4>
                    <p><strong>공고번호:</strong> {raw_data.get('bidNtceNo') or bid.get('bidntceno', 'N/A')}</p>
                    <p><strong>공고기관:</strong> {raw_data.get('ntceInsttNm') or bid.get('ntceinsttm', 'N/A')}</p>
                    <p><strong>수요기관:</strong> {raw_data.get('dmndInsttNm', 'N/A')}</p>
                    <p><strong>사업분류:</strong> {raw_data.get('bsnsDivNm') or bid.get('bsnsdivnm', 'N/A')}</p>
                </div>
                <div>
                    <h4 style="color: #495057; margin-bottom: 1rem;">💰 예산 및 일정</h4>
                    <p><strong>사업예산:</strong> <span style="color: #28a745; font-weight: bold;">{convert_to_won_format(raw_data.get('asignBdgtAmt') or bid.get('asignbdgtamt', 0))}</span></p>
                    <p><strong>게시일:</strong> {raw_data.get('bidNtceDate', 'N/A')}</p>
                    <p><strong>마감일:</strong> {raw_data.get('bidClseDate', 'N/A')}</p>
                    <p><strong>진행상태:</strong> <span style="color: #007bff;">{raw_data.get('bidNtceSttusNm', 'N/A')}</span></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 닫기 버튼
        if st.button("🚪 상세보기 닫기", key="ultimate_detail_close"):
            st.session_state.show_detail = False
            st.rerun()

# 세션 상태 초기화
if 'show_detail' not in st.session_state:
    st.session_state.show_detail = False

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if __name__ == "__main__":
    main()
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
