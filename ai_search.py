import streamlit as st
from openai import OpenAI
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from database import VectorSearchManager, BidDataManager
import json

# OpenAI API 키 설정
OPENAI_API_KEY = "your-openai-api-key"  # 실제 API 키로 변경

class BidSearchChatbot:
    """입찔 공고 검색 챗봇 클래스"""
    
    def __init__(self, openai_api_key: str):
        """챗봇 초기화"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.vector_manager = VectorSearchManager()
        self.bid_manager = BidDataManager()
        
        # 임베딩 모델 초기화
        try:
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
        except Exception as e:
            st.error(f"임베딩 모델 로드 실패: {e}")
            self.embedding_model = None
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        if not self.embedding_model:
            return []
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            st.error(f"임베딩 생성 오류: {e}")
            return []
    
    def search_vector_db(self, query: str, n_results: int = 10, threshold: float = 0.3) -> List[Dict]:
        """벡터 DB에서 관련 문서 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # 벡터 검색 수행
            results = self.vector_manager.semantic_search(
                query_embedding, threshold, n_results
            )
            
            # 결과 포맷 변환
            documents = []
            for result in results:
                try:
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    
                    document = {
                        "content": result.get('content', ''),
                        "metadata": metadata,
                        "score": result.get('similarity', 0),
                        "bidNtceNo": result.get('bidNtceNo', ''),
                        "bid_info": result.get('bid_info', {})
                    }
                    documents.append(document)
                except json.JSONDecodeError:
                    continue
            
            return documents
        except Exception as e:
            st.error(f"벡터 검색 오류: {e}")
            return []
    
    def get_simple_response(self, question: str, search_results: List[Dict]) -> str:
        """간단한 응답 생성"""
        if search_results:
            best_result = search_results[0]
            content = best_result.get("content", "관련 정보가 없습니다.")
            bid_info = best_result.get("bid_info", {})
            
            if isinstance(bid_info, dict):
                bid_name = bid_info.get('bidNtceNm', '공고명 없음')
                org_name = bid_info.get('ntceInsttNm', '기관명 없음')
                
                return f"""질문: {question}

가장 관련성 높은 공고:
- 공고명: {bid_name}
- 기관명: {org_name}
- 내용: {content[:300]}..."""
            else:
                return f"질문: {question}\n\n가장 관련성 높은 공고: {content}"
        else:
            return "관련된 공고를 찾을 수 없습니다."
    
    def get_gpt_response(self, question: str, search_results: List[Dict]) -> str:
        """GPT를 사용한 응답 생성"""
        if not search_results:
            return "관련된 공고를 찾을 수 없습니다."
        
        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(search_results[:5]):
            bid_info = doc.get('bid_info', {})
            metadata = doc.get('metadata', {})
            
            if isinstance(bid_info, dict):
                context_parts.append(f"""
[문서 {i+1}] (유사도: {doc['score']:.2f})
공고번호: {doc.get('bidNtceNo', 'N/A')}
공고명: {bid_info.get('bidNtceNm', metadata.get('공고명', 'N/A'))}
기관명: {bid_info.get('ntceInsttNm', metadata.get('기관명', 'N/A'))}
내용: {doc['content'][:300]}...
""")
        
        context = "\n".join(context_parts)
        
        system_prompt = """당신은 공공입찰 정보 검색 도우미입니다.
사용자의 질문에 대해 제공된 검색 결과를 바탕으로 답변해주세요.

답변 지침:
1. 검색된 공고들 중 가장 관련성 높은 것들을 우선적으로 언급
2. 각 공고의 핵심 정보(공고명, 기관, 마감일 등)를 간결하게 정리
3. 사용자가 추가로 확인해야 할 사항이 있다면 안내
4. 검색 결과가 충분하지 않다면 다른 검색어를 제안
"""
        
        user_prompt = f"""
사용자 질문: {question}

검색된 관련 공고:
{context}

위 검색 결과를 바탕으로 사용자 질문에 답변해주세요.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"응답 생성 중 오류 발생: {e}"

class SemanticSearchEngine:
    """시맨틱 검색 엔진"""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.vector_manager = VectorSearchManager()
        self.bid_manager = BidDataManager()
        
        # 임베딩 모델 초기화
        try:
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
        except Exception as e:
            st.error(f"임베딩 모델 로드 실패: {e}")
            self.embedding_model = None
    
    def search(self, query: str, num_results: int = 10, similarity_threshold: float = 0.3) -> List[Tuple[Dict, float]]:
        """시맨틱 검색 수행"""
        if not self.embedding_model:
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            
            # 벡터 검색 수행
            results = self.vector_manager.semantic_search(
                query_embedding.tolist(), similarity_threshold, num_results
            )
            
            # 결과 포맷 변환
            formatted_results = []
            for result in results:
                try:
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    
                    similarity_score = result.get('similarity', 0)
                    formatted_results.append((metadata, similarity_score))
                except json.JSONDecodeError:
                    continue
            
            return formatted_results
        except Exception as e:
            st.error(f"시맨틱 검색 중 오류 발생: {e}")
            return []
    
    def generate_rag_response(self, query: str, context_docs: List[Tuple[Dict, float]]) -> str:
        """검색된 문서를 기반으로 RAG 응답 생성"""
        if not context_docs:
            return "관련된 공고를 찾을 수 없습니다."
        
        # 컨텍스트 구성
        context_parts = []
        for i, (metadata, score) in enumerate(context_docs[:5]):
            context_parts.append(f"""
[문서 {i+1}] (유사도: {score:.2f})
공고번호: {metadata.get('공고번호', 'N/A')}
공고명: {metadata.get('공고명', 'N/A')}
기관명: {metadata.get('기관명', 'N/A')}
""")
        
        context = "\n".join(context_parts)
        
        system_prompt = """당신은 공공입찰 정보 검색 도우미입니다.
사용자의 질문에 대해 제공된 검색 결과를 바탕으로 답변해주세요.

답변 지침:
1. 검색된 공고들 중 가장 관련성 높은 것들을 우선적으로 언급
2. 각 공고의 핵심 정보(공고명, 기관, 마감일 등)를 간결하게 정리
3. 사용자가 추가로 확인해야 할 사항이 있다면 안내
4. 검색 결과가 충분하지 않다면 다른 검색어를 제안
"""
        
        user_prompt = f"""
사용자 질문: {query}

검색된 관련 공고:
{context}

위 검색 결과를 바탕으로 사용자 질문에 답변해주세요.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"응답 생성 중 오류 발생: {e}"

@st.cache_resource
def init_chatbot():
    """챗봇 인스턴스를 초기화하고 캐싱"""
    return BidSearchChatbot(openai_api_key=OPENAI_API_KEY)

@st.cache_resource
def init_semantic_search():
    """시맨틱 검색 엔진을 초기화하고 캐싱"""
    return SemanticSearchEngine(openai_api_key=OPENAI_API_KEY)
