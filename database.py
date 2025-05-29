import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import streamlit as st
from typing import List, Dict, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import json

# Supabase 연결 설정
SUPABASE_URL = "your-supabase-url"  # 실제 Supabase URL로 변경
SUPABASE_KEY = "your-supabase-anon-key"  # 실제 Supabase anon key로 변경
DATABASE_URL = "postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres"  # 실제 연결 문자열로 변경

class SupabaseDB:
    """Supabase PostgreSQL 연결 클래스"""
    
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Supabase PostgreSQL 데이터베이스에 연결"""
        try:
            self.connection = psycopg2.connect(
                DATABASE_URL,
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
        except Exception as e:
            st.error(f"데이터베이스 연결 실패: {e}")
            self.connection = None
    
    def get_cursor(self):
        """커서 반환"""
        if not self.connection:
            self.connect()
        return self.connection.cursor() if self.connection else None
    
    def close(self):
        """연결 종료"""
        if self.connection:
            self.connection.close()

@st.cache_resource
def get_db_connection():
    """캐시된 데이터베이스 연결 반환"""
    return SupabaseDB()

class BidDataManager:
    """입찰 데이터 관리 클래스"""
    
    def __init__(self):
        self.db = get_db_connection()
    
    def get_live_bids(self, limit: int = 1000, offset: int = 0) -> List[Dict]:
        """실시간 입찰 공고 조회"""
        cursor = self.db.get_cursor()
        if not cursor:
            return []
        
        try:
            query = """
            SELECT 
                bidNtceNo,
                raw->>'bidNtceNm' as bidNtceNm,
                raw->>'ntceInsttNm' as ntceInsttNm,
                raw->>'bsnsDivNm' as bsnsDivNm,
                raw->>'asignBdgtAmt' as asignBdgtAmt,
                raw->>'bidNtceDate' as bidNtceDate,
                raw->>'bidClseDate' as bidClseDate,
                raw->>'bidClseTm' as bidClseTm,
                raw->>'bidNtceUrl' as bidNtceUrl,
                raw->>'bidNtceBgn' as bidNtceBgn,
                raw->>'bidNtceSttusNm' as bidNtceSttusNm,
                raw->>'dmndInsttNm' as dmndInsttNm,
                raw->>'bidprcPsblIndstrytyNm' as bidprcPsblIndstrytyNm,
                raw->>'cmmnReciptMethdNm' as cmmnReciptMethdNm,
                raw->>'rgnLmtYn' as rgnLmtYn,
                raw->>'prtcptPsblRgnNm' as prtcptPsblRgnNm,
                raw->>'presmptPrce' as presmptPrce,
                raw,
                created_at,
                updated_at
            FROM bids_live
            WHERE raw->>'bidNtceSttusNm' = '공고중'
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """
            cursor.execute(query, (limit, offset))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            st.error(f"입찰 공고 조회 오류: {e}")
            return []
    
    def search_bids_by_keyword(self, keyword: str, categories: List[str] = None, 
                              start_date: str = None, end_date: str = None) -> List[Dict]:
        """키워드로 입찰 공고 검색"""
        cursor = self.db.get_cursor()
        if not cursor:
            return []
        
        try:
            base_query = """
            SELECT 
                bidNtceNo,
                raw->>'bidNtceNm' as bidNtceNm,
                raw->>'ntceInsttNm' as ntceInsttNm,
                raw->>'bsnsDivNm' as bsnsDivNm,
                raw->>'asignBdgtAmt' as asignBdgtAmt,
                raw->>'bidNtceDate' as bidNtceDate,
                raw->>'bidClseDate' as bidClseDate,
                raw,
                created_at
            FROM bids_live
            WHERE 1=1
            """
            
            params = []
            conditions = []
            
            # 키워드 검색 (ILIKE 사용)
            if keyword:
                conditions.append("""
                (raw->>'bidNtceNm' ILIKE %s OR 
                 raw->>'ntceInsttNm' ILIKE %s OR 
                 raw->>'dmndInsttNm' ILIKE %s OR 
                 raw->>'bidprcPsblIndstrytyNm' ILIKE %s)
                """)
                keyword_param = f"%{keyword}%"
                params.extend([keyword_param, keyword_param, keyword_param, keyword_param])
            
            # 카테고리 필터
            if categories:
                conditions.append("raw->>'bsnsDivNm' = ANY(%s)")
                params.append(categories)
            
            # 날짜 필터
            if start_date:
                conditions.append("raw->>'bidNtceDate' >= %s")
                params.append(start_date)
            
            if end_date:
                conditions.append("raw->>'bidNtceDate' <= %s")
                params.append(end_date)
            
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            base_query += " ORDER BY created_at DESC LIMIT 100"
            
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            st.error(f"검색 오류: {e}")
            return []
    
    def get_bid_by_number(self, bid_no: str) -> Optional[Dict]:
        """공고번호로 특정 입찰 공고 조회"""
        cursor = self.db.get_cursor()
        if not cursor:
            return None
        
        try:
            query = """
            SELECT bidNtceNo, raw, created_at, updated_at
            FROM bids_live
            WHERE bidNtceNo = %s
            """
            cursor.execute(query, (bid_no,))
            result = cursor.fetchone()
            return dict(result) if result else None
        except Exception as e:
            st.error(f"공고 조회 오류: {e}")
            return None
    
    def get_bid_summary(self, bid_no: str) -> tuple:
        """GPT 요약 조회"""
        cursor = self.db.get_cursor()
        if not cursor:
            return "요약을 불러올 수 없습니다.", None, None
        
        try:
            # hwp_based 우선 조회
            query = """
            SELECT summary, summary_type, created_at
            FROM bid_summaries
            WHERE bidNtceNo = %s
            ORDER BY 
                CASE WHEN summary_type = 'hwp_based' THEN 1 
                     WHEN summary_type = 'basic_info' THEN 2 
                     ELSE 3 END,
                created_at DESC
            LIMIT 1
            """
            cursor.execute(query, (bid_no,))
            result = cursor.fetchone()
            
            if result:
                created_at = result['created_at'].strftime("%Y-%m-%d %H:%M") if result['created_at'] else None
                return result['summary'], created_at, result['summary_type']
            else:
                return "이 공고에 대한 요약이 아직 생성되지 않았습니다.", None, None
        except Exception as e:
            return f"요약 조회 중 오류 발생: {e}", None, None

class VectorSearchManager:
    """벡터 검색 관리 클래스"""
    
    def __init__(self):
        self.db = get_db_connection()
    
    def semantic_search(self, query_embedding: List[float], threshold: float = 0.3, limit: int = 10) -> List[Dict]:
        """시맨틱 검색 수행"""
        cursor = self.db.get_cursor()
        if not cursor:
            return []
        
        try:
            # pgvector의 cosine distance 사용
            query = """
            SELECT 
                sc.id,
                sc.content,
                sc.metadata,
                sc.bidNtceNo,
                1 - (sc.embedding <=> %s::vector) AS similarity,
                bl.raw as bid_info
            FROM semantic_chunks sc
            JOIN bids_live bl ON sc.bidNtceNo = bl.bidNtceNo
            WHERE 1 - (sc.embedding <=> %s::vector) > %s
            ORDER BY sc.embedding <=> %s::vector
            LIMIT %s
            """
            
            # embedding을 문자열로 변환
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            cursor.execute(query, (embedding_str, embedding_str, threshold, embedding_str, limit))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            st.error(f"벡터 검색 오류: {e}")
            return []
    
    def add_semantic_chunk(self, content: str, metadata: Dict, embedding: List[float], bid_no: str):
        """시맨틱 청크 추가"""
        cursor = self.db.get_cursor()
        if not cursor:
            return False
        
        try:
            query = """
            INSERT INTO semantic_chunks (content, metadata, embedding, bidNtceNo)
            VALUES (%s, %s, %s::vector, %s)
            """
            
            embedding_str = f"[{','.join(map(str, embedding))}]"
            cursor.execute(query, (content, Json(metadata), embedding_str, bid_no))
            return True
        except Exception as e:
            st.error(f"시맨틱 청크 추가 오류: {e}")
            return False

def convert_to_won_format(amount):
    """금액을 원 단위로 포맷팅"""
    try:
        if not amount or pd.isna(amount):
            return "공고 참조"
        
        # 문자열인 경우 쉼표 제거 후 숫자로 변환
        if isinstance(amount, str):
            amount = amount.replace(",", "")
        
        amount = float(amount)

        if amount >= 100000000:  # 1억 이상
            amount_in_100m = amount / 100000000
            return f"{amount_in_100m:.1f}억원"
        elif amount >= 10000:  # 1만 이상
            amount_in_10k = amount / 10000
            return f"{amount_in_10k:.1f}만원"
        else:
            return f"{int(amount):,}원"
        
    except (ValueError, TypeError):
        return "공고 참조"

def format_won(amount):
    """금액 포맷팅 (쉼표 추가)"""
    try:
        if isinstance(amount, str):
            amount = amount.replace(",", "")
        amount = int(float(amount))
        return f"{amount:,}원"
    except (ValueError, TypeError):
        return "공고 참조"

def format_joint_contract(value):
    """공동수급 포맷팅"""
    if value and str(value).strip():
        return f"허용 [{str(value).strip()}]"
    return "공고서참조"
