# config.py - 환경 설정
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    url: str
    host: str
    port: int
    database: str
    user: str
    password: str
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        return cls(
            url=os.getenv('DATABASE_URL', ''),
            host=os.getenv('DB_HOST', 'db.your-project.supabase.co'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'postgres'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

@dataclass
class OpenAIConfig:
    """OpenAI API 설정"""
    api_key: str
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    @classmethod
    def from_env(cls) -> 'OpenAIConfig':
        return cls(
            api_key=os.getenv('OPENAI_API_KEY', ''),
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        )

@dataclass
class APIConfig:
    """나라장터 API 설정"""
    service_key: str
    base_url: str = "http://apis.data.go.kr/1230000/ad/BidPublicInfoService"
    
    @classmethod  
    def from_env(cls) -> 'APIConfig':
        return cls(
            service_key=os.getenv('NARAJANGTER_API_KEY', ''),
            base_url=os.getenv('API_BASE_URL', 'http://apis.data.go.kr/1230000/ad/BidPublicInfoService')
        )

@dataclass
class AppConfig:
    """애플리케이션 전체 설정"""
    debug: bool = False
    page_size: int = 10
    max_search_results: int = 100
    vector_similarity_threshold: float = 0.3
    
    database: DatabaseConfig
    openai: OpenAIConfig
    api: APIConfig
    
    @classmethod
    def load(cls) -> 'AppConfig':
        return cls(
            debug=os.getenv('DEBUG', 'False').lower() == 'true',
            page_size=int(os.getenv('PAGE_SIZE', '10')),
            max_search_results=int(os.getenv('MAX_SEARCH_RESULTS', '100')),
            vector_similarity_threshold=float(os.getenv('VECTOR_SIMILARITY_THRESHOLD', '0.3')),
            database=DatabaseConfig.from_env(),
            openai=OpenAIConfig.from_env(),
            api=APIConfig.from_env()
        )
