import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import json

from database import BidDataManager, convert_to_won_format
from langgraph_workflow import create_hybrid_search_workflow, HybridSearchState
from langchain_core.messages import HumanMessage

def show_live_bids_tab(bid_manager: BidDataManager):
    """실시간 입찰 공고 탭 UI 구성"""
    st.markdown("""
    <style>
        .main-header-2 {
            text-align: center;
            padding: 3.5rem 0;
            background: linear-gradient(135deg, #ff9a8b 0%, #ff6a88 100%);
            color: white;
            border-radius: 20px;
            margin-bottom: 3rem;
            box-shadow: 0 10px 20px rgba(0,0,0,0.25);
            overflow: hidden;
            position: relative;
            animation: fadeIn 1.5s ease-out;
        }
        .main-header-2::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: rotateBg 20s linear infinite;
            opacity: 0.7;
        }
        .main-header-2 h1 {
            font-size: 3.8rem;
            font-weight: 900;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
            margin-bottom: 0.8rem;
        }
        .main-header-2 p {
            font-size: 1.5rem;
            font-weight: 400;
            opacity: 0.95;
            line-height: 1.6;
            max-width: 900px;
            margin: 0.8rem auto 0 auto;
            position: relative;
            z-index: 1;
        }
        @keyframes rotateBg {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header-2">
        <h1>🚀 당신의 입찰 성공 파트너, AI 입찰 도우미!</h1>
        <p>
            매일 업데이트되는 실시간 공고 확인부터<br>
            인공지능 기반의 정확한 검색과 스마트한 질의응답까지,<br>
            복잡한 입찰 과정을 쉽고 빠르게 경험하세요.
        </p>
    </div>
    """, unsafe_allow_html=True)
         
    st.subheader("📢 현재 진행 중인 입찰 목록")
    
    # 데이터 로드
    try:
        live_bids = bid_manager.get_live_bids(limit=1000)
        if not live_bids:
            st.warning("현재 진행 중인 입찰 공고가 없습니다.")
            return
        
        # DataFrame 변환
        df_live = pd.DataFrame(live_bids)
        
        # 필요한 컬럼만 선택 (raw 데이터에서 추출된 값 우선 사용)
        display_data = []
        for row in live_bids:
            raw_data = row.get('raw', {})
            display_row = {
                "공고번호": raw_data.get('bidNtceNo', row.get('bidNtceNo', '')),
                "공고명": raw_data.get('bidNtceNm', row.get('bidNtceNm', '')),
                "공고기관": raw_data.get('ntceInsttNm', row.get('ntceInsttNm', '')),
                "분류": raw_data.get('bsnsDivNm', row.get('bsnsDivNm', '')),
                "금액": raw_data.get('asignBdgtAmt', row.get('asignBdgtAmt', 0)),
                "게시일": raw_data.get('bidNtceDate', row.get('bidNtceDate', '')),
                "마감일": raw_data.get('bidClseDate', row.get('bidClseDate', '')),
                "마감시간": raw_data.get('bidClseTm', row.get('bidClseTm', '')),
                "입찰공고상태명": raw_data.get('bidNtceSttusNm', row.get('bidNtceSttusNm', '')),
                "수요기관": raw_data.get('dmndInsttNm', row.get('dmndInsttNm', '')),
                "투찰가능업종명": raw_data.get('bidprcPsblIndstrytyNm', row.get('bidprcPsblIndstrytyNm', '')),
                "공동수급": raw_data.get('cmmnReciptMethdNm', row.get('cmmnReciptMethdNm', '')),
                "지역제한": raw_data.get('rgnLmtYn', row.get('rgnLmtYn', '')),
                "참가가능지역명": raw_data.get('prtcptPsblRgnNm', row.get('prtcptPsblRgnNm', '')),
                "추정가격": raw_data.get('presmptPrce', row.get('presmptPrce', 0)),
                "raw_data": row  # 상세보기용
            }
            display_data.append(display_row)
        
        df_live = pd.DataFrame(display_data)
        
        # 날짜 형식 변환
        def convert_date(date_str):
            try:
                if date_str and len(str(date_str)) == 8:
                    return pd.to_datetime(str(date_str), format='%Y%m%d')
                else:
                    return pd.NaT
            except:
                return pd.NaT
        
        df_live["마감일"] = df_live["마감일"].apply(convert_date)
        df_live["게시일"] = df_live["게시일"].apply(convert_date)
        df_live = df_live.sort_values(by=['게시일'], ascending=False, na_position='last')

        # 필터 UI
        search_keyword = st.text_input("🔎 공고명 또는 공고기관 검색")
        unique_categories = ["공사", "용역", "물품", "외자"]
        selected_cls = st.multiselect("📁 분류 선택", 
                                     options=unique_categories, 
                                     default=unique_categories)

        col2, col3, col4 = st.columns(3)        
        with col2:
            start_date = st.date_input("📅 게시일 기준 시작일", 
                                     value=(datetime.now() - timedelta(days=30)).date())
        with col3:
            end_date = st.date_input("📅 게시일 기준 종료일", 
                                   value=datetime.now().date())
        with col4:
            sort_col = st.selectbox("정렬기준", options=["실시간","게시일","마감일","금액"])
            if sort_col == "실시간":
                sort_order = "내림차순"
                st.empty()
            else:
                sort_order = st.radio("정렬 방향", options=["오름차순", "내림차순"], 
                                    horizontal=True, label_visibility="collapsed")
        
        # 필터링 적용
        filtered = df_live.copy()

        if selected_cls:
            filtered = filtered[filtered["분류"].isin(selected_cls)]

        if search_keyword:
            filtered = filtered[ 
                filtered["공고명"].str.contains(search_keyword, case=False, na=False, regex=False) |
                filtered["공고기관"].str.contains(search_keyword, case=False, na=False, regex=False) |
                filtered["공고번호"].str.contains(search_keyword, case=False, na=False, regex=False)
            ]

        # 날짜 필터링
        filtered = filtered[
            (filtered["게시일"].dt.date >= start_date) & 
            (filtered["게시일"].dt.date <= end_date)
        ]

        # 정렬 적용
        ascending = True if sort_order == "오름차순" else False

        if sort_col == "실시간":
            filtered = filtered.sort_values(by=["게시일"], ascending=False)
        elif sort_col == "게시일":
            filtered = filtered.sort_values(by=["게시일"], ascending=ascending)
        elif sort_col == "마감일":
            filtered = filtered.sort_values(by="마감일", ascending=ascending)
        elif sort_col == "금액":
            filtered = filtered.sort_values(by="금액", ascending=ascending)

        st.markdown(f"<div style='text-align: left; margin-bottom: 10px;'>검색 결과 {len(filtered)}건</div>", 
                   unsafe_allow_html=True)

        # 페이지네이션
        PAGE_SIZE = 10
        def paginate_dataframe(df, page_num, page_size):
            start_index = page_num * page_size
            end_index = (page_num + 1) * page_size
            return df.iloc[start_index:end_index]

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = 0

        total_pages = (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE
        paginated_df = paginate_dataframe(filtered, st.session_state["current_page"], PAGE_SIZE)
        
        st.write("")
        st.write("")
        
        # 테이블 헤더
        header_cols = st.columns([1,1.5, 3.5, 2.5, 1, 1.5,1.5, 1.5, 1])
        headers = ['공고번호',"구분",'공고명','공고기관','분류','금액','게시일','마감일','상세정보']
        for col, head in zip(header_cols, headers):
            col.markdown(f"**{head}**")

        st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

        # 행 렌더링
        for i, (idx, row) in enumerate(paginated_df.iterrows()):
            cols = st.columns([1,1.5, 3.5, 2.5, 1, 1.5,1.5, 1.5, 1])
            cols[0].write(row["공고번호"])
            cols[1].write(row["입찰공고상태명"])
            cols[2].markdown(row["공고명"])
            cols[3].write(row["공고기관"])
            cols[4].write(row["분류"])            
            금액 = row["추정가격"] if row["분류"] == "공사" else row["금액"]
            cols[5].write(convert_to_won_format(금액))
            
            게시일_표시 = row["게시일"].strftime("%Y-%m-%d") if pd.notna(row["게시일"]) else "정보없음"
            cols[6].write(게시일_표시)
            
            if pd.isna(row["마감일"]):
                cols[7].write("공고 참조")
            else:
                cols[7].write(row["마감일"].strftime("%Y-%m-%d"))
                
            if cols[8].button("보기", key=f"live_detail_{i}"):
                st.session_state["page"] = "detail"
                st.session_state["selected_live_bid"] = row["raw_data"]
                st.rerun()

            st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

        # 페이지 이동 버튼
        cols_pagination = st.columns([1, 3, 1])
        with cols_pagination[0]:
            if st.session_state["current_page"] > 0:
                if st.button("이전"):
                    st.session_state["current_page"] -= 1
                    st.rerun()

        with cols_pagination[2]:
            if st.session_state['current_page'] < total_pages - 1:
                if st.button("다음"):
                    st.session_state["current_page"] += 1
                    st.rerun()

        st.markdown(f"<div style='text-align: center;'> {st.session_state['current_page'] + 1} / {total_pages}</div>", 
                   unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")

def show_semantic_search_tab(semantic_engine, bid_manager: BidDataManager):
    """시맨틱 검색 탭 UI 구성"""
    st.markdown("""
    <style>
        .main-header-1 {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 20px;
            margin-bottom: 2.5rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            overflow: hidden;
            position: relative;
            animation: fadeIn 1.5s ease-out;
        }        
        .main-header-1 h1 {
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-header-1 p {
            font-size: 1.4rem;
            opacity: 0.9;
            line-height: 1.6;
            max-width: 800px;
            margin: 0.8rem auto 0 auto;
            position: relative;
            z-index: 1;
        }      
        .main-header-1::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, rgba(255,255,255,0.1) 0%, transparent 20%, transparent 80%, rgba(255,255,255,0.1) 100%);
            animation: scanLine 4s infinite linear;
            z-index: 0;
        }
        @keyframes scanLine {
            0% { transform: translateX(-100%); }
            50% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header-1">
        <h1>🔍 스마트 AI 검색 엔진</h1>
        <p>
            원하는 입찰 정보를 키워드 대신 자연어로 검색하고<br>
            AI가 요약해주는 핵심 내용을 확인하세요.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 검색 UI
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("검색어를 입력하세요", 
                                    placeholder="예: 서버 구축, 소프트웨어 개발, 건설 공사 등",
                                    key="semantic_search_input")
    with col2:
        search_button = st.button("🔍 검색", key="semantic_search_btn", type="primary")
    
    # 검색 옵션
    with st.expander("🔧 검색 옵션"):
        num_results = st.slider("검색 결과 수", min_value=5, max_value=30, value=10, step=5)
        similarity_threshold = st.slider("유사도 임계값", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    
    # 검색 실행
    if search_button and search_query:
        with st.spinner("검색 중..."):
            # 시맨틱 검색 수행
            search_results = semantic_engine.search(search_query, num_results, similarity_threshold)
            
            if search_results:
                # 유사도 임계값 필터링
                filtered_results = [(meta, score) for meta, score in search_results if score >= similarity_threshold]
                
                st.markdown(f"### 검색 결과: {len(filtered_results)}건")
                
                if filtered_results:
                    # RAG 응답 생성
                    rag_response = semantic_engine.generate_rag_response(search_query, filtered_results)
                    
                    # AI 응답 표시
                    st.markdown("#### 🤖 AI 검색 요약")
                    st.info(rag_response)
                    
                    st.markdown("---")
                    st.markdown("#### 📋 검색된 공고 목록")
                    
                    # 검색 결과를 표시
                    results_data = []
                    for metadata, score in filtered_results:
                        # Supabase에서 추가 정보 조회
                        bid_no = metadata.get('공고번호')
                        bid_doc = None
                        if bid_no:
                            bid_doc = bid_manager.get_bid_by_number(bid_no)
                        
                        if bid_doc:
                            raw_data = bid_doc.get('raw', {})
                            results_data.append({
                                "유사도": f"{score:.2%}",
                                "공고번호": metadata.get('공고번호', 'N/A'),
                                "공고명": metadata.get('공고명', 'N/A'),
                                "기관명": metadata.get('기관명', 'N/A'),
                                "수집일시": metadata.get('수집일시', 'N/A'),
                                "마감일": raw_data.get('bidClseDate', 'N/A'),
                                "예산": convert_to_won_format(raw_data.get('asignBdgtAmt', 0)),
                                "bid_doc": bid_doc
                            })
                    
                    if results_data:
                        # 결과 테이블 표시
                        for idx, result in enumerate(results_data):
                            with st.container():
                                col1, col2, col3, col4 = st.columns([0.8, 3, 2, 1])
                                
                                # 유사도 표시 (색상 코딩)
                                similarity_value = float(result["유사도"].strip('%')) / 100
                                if similarity_value >= 0.7:
                                    col1.markdown(f"<span style='color: green; font-weight: bold;'>{result['유사도']}</span>", 
                                                unsafe_allow_html=True)
                                elif similarity_value >= 0.5:
                                    col1.markdown(f"<span style='color: orange; font-weight: bold;'>{result['유사도']}</span>", 
                                                unsafe_allow_html=True)
                                else:
                                    col1.markdown(f"<span style='color: red;'>{result['유사도']}</span>", 
                                                unsafe_allow_html=True)
                                
                                col2.write(f"**{result['공고명']}**")
                                col3.write(result['기관명'])
                                
                                # 상세보기 버튼
                                if col4.button("상세", key=f"semantic_detail_{idx}"):
                                    if result['bid_doc']:
                                        st.session_state["page"] = "detail"
                                        st.session_state["selected_live_bid"] = result['bid_doc']
                                        st.rerun()
                                    
                                # 추가 정보 표시
                                with st.expander(f"더보기 - {result['공고번호']}"):
                                    st.write(f"**마감일:** {result['마감일']}")
                                    st.write(f"**예산:** {result['예산']}")
                                    st.write(f"**수집일시:** {result['수집일시']}")
                                    
                                    # GPT 요약 표시
                                    if result['bid_doc']:
                                        summary, created_at, summary_type = bid_manager.get_bid_summary(result['공고번호'])
                                        if summary and summary != "이 공고에 대한 요약이 아직 생성되지 않았습니다.":
                                            st.markdown("**📝 요약:**")
                                            st.write(summary)
                                    
                                    st.divider()
                else:
                    st.warning(f"유사도 {similarity_threshold:.1%} 이상인 검색 결과가 없습니다. 임계값을 낮춰보세요.")
            else:
                st.warning("검색 결과가 없습니다. 다른 검색어를 시도해보세요.")

def add_langgraph_search_tab(bid_manager: BidDataManager):
    """LangGraph AI 검색 탭 UI"""
    st.markdown("""
    <style>
        .langgraph-header-option2 {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 2.5rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
            overflow: hidden;
            position: relative;
            animation: fadeIn 1.5s ease-out;
        }
        .langgraph-header-option2 h1 {
            font-size: 3.5rem;
            font-weight: 900;
            text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
            position: relative;
            z-index: 1;
            margin-bottom: 0.8rem;
        }
        .langgraph-header-option2 p {
            font-size: 1.4rem;
            font-weight: 400;
            opacity: 0.95;
            line-height: 1.6;
            max-width: 900px;
            margin: 0.8rem auto 0 auto;
            position: relative;
            z-index: 1;
        }
        .langgraph-header-option2::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 300%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3) 50%, transparent);
            transform: translateX(-100%);
            animation: lightSweep 10s infinite ease-in-out;
            z-index: 0;
        }
        @keyframes lightSweep {
            0% { transform: translateX(-100%); }
            50% { transform: translateX(0%); }
            100% { transform: translateX(100%); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="langgraph-header-option2">
        <h1>✨ LangGraph 기반 고급 분석</h1>
        <p>
            AI 워크플로우로 입찰 데이터의 숨겨진 패턴을 파악하고<br>
            전략적인 의사결정을 위한 깊이 있는 통찰을 얻으세요.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 하이브리드 검색 워크플로우 생성
    hybrid_workflow = create_hybrid_search_workflow()
    
    # 검색 UI
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input("검색어를 입력하세요", 
                                   placeholder="예: AI 개발, 서버 구축, 소프트웨어 개발 등",
                                   key="langgraph_hybrid_search_input")
    with col2:
        search_button = st.button("🔍 검색", key="langgraph_hybrid_search_btn", type="primary")
    
    # 검색 옵션
    with st.expander("🔧 검색 옵션"):
        col1, col2 = st.columns(2)
        with col1:
            min_results_for_api = st.slider("API 검색 기준 (최소 결과 수)", 
                                           min_value=5, max_value=20, value=10,
                                           help="DB 검색 결과가 이 수치 미만일 때 API 검색 실행")
        with col2:
            show_process = st.checkbox("검색 프로세스 표시", value=True)
    
    # 검색 실행
    if search_button and search_query:
        # 프로세스 컨테이너
        if show_process:
            process_container = st.container()
        
        with st.spinner("하이브리드 검색 실행 중..."):
            # 초기 상태 설정
            initial_state: HybridSearchState = {
                "query": search_query,
                "search_method": [],
                "supabase_results": [],
                "vector_results": [],
                "api_results": {},
                "combined_results": [],
                "final_results": [],
                "summary": "",
                "messages": [],
                "error": None,
                "total_count": 0,
                "need_api_search": False
            }
            
            # 워크플로우 실행
            try:
                final_state = hybrid_workflow.invoke(initial_state)
                
                # 프로세스 로그 표시
                if show_process:
                    with process_container:
                        st.markdown("### 🔄 검색 프로세스")
                        for msg in final_state["messages"]:
                            if isinstance(msg, HumanMessage):
                                st.markdown(f'<div class="process-log">👤 {msg.content}</div>', 
                                          unsafe_allow_html=True)
                            else:
                                # 소스별 다른 스타일 적용
                                if "Supabase" in msg.content:
                                    style_class = "db-result"
                                elif "VectorDB" in msg.content:
                                    style_class = "vector-result"
                                elif "API" in msg.content:
                                    style_class = "api-result"
                                else:
                                    style_class = "process-log"
                                
                                st.markdown(f'<div class="{style_class}">🤖 {msg.content}</div>', 
                                          unsafe_allow_html=True)
                
                # 결과 표시
                if final_state["final_results"]:
                    st.markdown("---")
                    
                    # 요약 표시
                    st.success(final_state["summary"])
                    
                    # 검색 소스 태그
                    st.markdown("### 🏷️ 사용된 검색 방법")
                    cols = st.columns(len(final_state["search_method"]))
                    for idx, method in enumerate(final_state["search_method"]):
                        with cols[idx]:
                            if method == "Supabase":
                                st.info(f"📚 {method}")
                            elif method == "VectorDB":
                                st.warning(f"🔍 {method}")
                            elif method == "API":
                                st.success(f"🌐 {method}")
                    
                    # 상세 결과
                    st.markdown("### 📋 검색 결과")

                    for idx, item in enumerate(final_state["final_results"], 1):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([0.5, 3, 1.5, 1])

                            with col1:
                                st.markdown(f"**{idx}**")

                            with col2:
                                st.markdown(f"**{item.get('bidNtceNm', '제목 없음')}**")
                                st.caption(f"{item.get('ntceInsttNm', '기관명 없음')} | "
                                        f"공고번호: {item.get('bidNtceNo', 'N/A')}")

                            with col3:
                                sources = item.get('source', 'Unknown').split(', ')
                                source_tags = []
                                for source in sources:
                                    if 'Supabase' in source:
                                        source_tags.append("📚")
                                    elif 'VectorDB' in source:
                                        source_tags.append("🔍")
                                    elif 'API' in source:
                                        source_tags.append("🌐")
                                st.write(" ".join(source_tags) + f" (점수: {item.get('relevance_score', 0):.1f})")

                            with col4:
                                st.write(convert_to_won_format(item.get('asignBdgtAmt', 0)))
                                if st.button("상세", key=f"detail_hybrid_{idx}"):
                                    st.session_state["page"] = "detail"
                                    st.session_state["selected_live_bid"] = item
                                    st.rerun()

                            # 상세 정보 확장
                            bid_no = item.get('bidNtceNo')
                            if bid_no:
                                # Supabase에서 해당 공고의 상세 정보를 조회
                                bid_detail = bid_manager.get_bid_by_number(bid_no)

                                if bid_detail:
                                    raw_data = bid_detail.get('raw', {})
                                    # 상세 데이터 표시
                                    with st.expander("더보기"):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"**마감일:** {raw_data.get('bidClseDate', 'N/A')}")
                                            st.write(f"**분류:** {raw_data.get('bsnsDivNm', 'N/A')}")
                                            st.write(f"**검색 소스:** {item.get('source', 'N/A')}")
                                        with col2:
                                            st.write(f"**입찰방법:** {raw_data.get('bidMthdNm', 'N/A')}")
                                            st.write(f"**지역제한:** {raw_data.get('rgnLmtYn', 'N/A')}")
                                            st.write(f"**관련도 점수:** {item.get('relevance_score', 0):.1f}")

                                        # AI 요약이 있을 경우 추가
                                        summary, created_at, summary_type = bid_manager.get_bid_summary(bid_no)
                                        if summary and summary != "이 공고에 대한 요약이 아직 생성되지 않았습니다.":
                                            created_info = f"({summary_type} - 생성일: {created_at})" if created_at else ""
                                            st.markdown(f"**📝 AI 요약:** {summary} {created_info}")
                                
                                else:
                                    st.warning(f"공고번호 {bid_no}에 대한 상세 정보를 찾을 수 없습니다.")

                            st.markdown("---")
                else:
                    st.warning(f"'{search_query}'에 대한 검색 결과가 없습니다.")
                    
            except Exception as e:
                st.error(f"검색 중 오류 발생: {str(e)}")
    
    # LangGraph 정보
    with st.expander("🚀 LangGraph 하이브리드 검색 특징", expanded=False):
        st.markdown("""
        **지능형 하이브리드 검색 시스템:**
        
        **🔄 3단계 순차 검색:**
        1. **Supabase 검색** - 키워드 기반 정확한 매칭
        2. **VectorDB 검색** - AI 시맨틱 유사도 검색
        3. **나라장터 API** - 실시간 최신 데이터 (필요시만)
        
        **🎯 스마트 검색 전략:**
        - DB 검색 결과가 충분하면 API 호출 생략 (성능 최적화)
        - 결과가 부족할 때만 실시간 API 검색 실행
        - 중복 제거 및 관련도 기반 정렬
        
        **📊 통합 점수 시스템:**
        - 여러 소스에서 발견된 공고는 높은 점수
        - Supabase 기본 점수: 5점
        - VectorDB 유사도 점수: 0~10점
        - API 추가 점수: 3점
        
        **✨ 장점:**
        - 빠른 응답 속도 (캐시된 데이터 우선 사용)
        - 정확도와 완성도 높은 검색 결과
        - 투명한 검색 프로세스 추적
        - API 호출 최소화로 비용 절감
        """)

def add_chatbot_to_streamlit(chatbot, process_question_fn):
    """Streamlit 앱에 챗봇 기능 추가"""
    st.markdown("""
    <style>
        .chatbot-header-4 {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #ff7e5f 0%, #feb47b 100%);
            color: white;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            overflow: hidden;
            position: relative;
            animation: fadeIn 1.5s ease-out;
        }
        .chatbot-header-4 h1 {
            font-size: 3.5rem;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .chatbot-header-4 p {
            font-size: 1.4rem;
            opacity: 0.8;
            line-height: 1.6;
            max-width: 900px;
            margin: 0.8rem auto 0 auto;
            position: relative;    
            z-index: 1;
        }  
        .chatbot-header-4::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('data:image/svg+xml;utf8,<svg width="100%" height="100%" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="10" cy="10" r="7" fill="rgba(255,255,255,0.2)"/><circle cx="30" cy="50" r="10" fill="rgba(255,255,255,0.18)"/><circle cx="70" cy="20" r="8" fill="rgba(255,255,255,0.17)"/><circle cx="90" cy="80" r="9" fill="rgba(255,255,255,0.19)"/><circle cx="50" cy="90" r="6" fill="rgba(255,255,255,0.16)"/></svg>') repeat;
            background-size: 20% 20%;
            animation: bubbleFloat 20s infinite linear;
            z-index: 0;
        }
        @keyframes bubbleFloat {
            0% { transform: translateY(0) translateX(0); opacity: 0.8; }
            25% { transform: translateY(-5%) translateX(5%); opacity: 0.9; }
            50% { transform: translateY(-10%) translateX(0); opacity: 0.8; }
            75% { transform: translateY(-5%) translateX(-5%); opacity: 0.9; }
            100% { transform: translateY(0) translateX(0); opacity: 0.8; }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chatbot-header-4">
        <h1>🤖 AI 챗봇 도우미</h1>
        <p>
            궁금한 점이 있다면 언제든지 질문하세요.<br>
            AI가 친절하게 답변해 드릴게요!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 예시 질문 버튼들
    example_questions = [
        "AI 개발 관련 입찰 공고를 찾아주세요",
        "서버 구축 입찰 현황은 어떤가요?",
        "최근 공고된 소프트웨어 개발 입찰은?",
        "1억원 이상 IT 입찰 공고가 있나요?",
        "오늘 마감되는 입찰 공고는?",
        "특정 기관의 입찰 공고를 보여주세요"
    ]

    cols = st.columns(6)
    for idx, question in enumerate(example_questions):
        if cols[idx % 6].button(question, key=f"example_{question}"):
            st.session_state.pending_question = question
            st.rerun()

    if st.button("🔄 대화 초기화"):
        st.session_state.chat_messages = []
        if 'pending_question' in st.session_state:
            del st.session_state.pending_question
        st.rerun()

    # 세션 상태 초기화
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # 이전 대화 내용 표시
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 예시 질문 처리
    if 'pending_question' in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question
        process_question_fn(question, chatbot)
        st.rerun()
    
    # 사용자 입력 받기
    if prompt := st.chat_input("질문을 입력하세요 (예: AI 관련 입찰 공고를 찾아주세요)"):
        process_question_fn(prompt, chatbot)
