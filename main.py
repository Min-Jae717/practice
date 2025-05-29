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
def init_manager():
    check_secrets()
    return SimpleBidManager()

# 메인 애플리케이션
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚀 입찰 공고 검색 서비스</h1>
        <p>실시간 입찰 정보를 쉽고 빠르게 확인하세요</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 매니저 초기화
    bid_manager = init_manager()
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📢 실시간 입찰 공고", "🔍 검색"])
    
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
