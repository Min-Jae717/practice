import streamlit as st
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
    def from_secrets(cls) -> 'DatabaseConfig':
        """Streamlit secrets에서 데이터베이스 설정 로드"""
        try:
            db_secrets = st.secrets["database"]
            return cls(
                url=db_secrets.get("url", ""),
                host=db_secrets.get("host", ""),
                port=int(db_secrets.get("port", "5432")),
                database=db_secrets.get("database", "postgres"),
                user=db_secrets.get("user", "postgres"),
                password=db_secrets.get("password", "")
            )
        except KeyError as e:
            st.error(f"데이터베이스 설정을 찾을 수 없습니다: {e}")
            st.stop()

@dataclass
class OpenAIConfig:
    """OpenAI API 설정"""
    api_key: str
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    @classmethod
    def from_secrets(cls) -> 'OpenAIConfig':
        """Streamlit secrets에서 OpenAI 설정 로드"""
        try:
            openai_secrets = st.secrets["openai"]
            return cls(
                api_key=openai_secrets["api_key"],
                model=openai_secrets.get("model", "gpt-4o-mini"),
                embedding_model=openai_secrets.get("embedding_model", "text-embedding-3-small")
            )
        except KeyError as e:
            st.error(f"OpenAI API 설정을 찾을 수 없습니다: {e}")
            st.stop()

@dataclass
class APIConfig:
    """나라장터 API 설정"""
    service_key: str
    base_url: str = "http://apis.data.go.kr/1230000/ad/BidPublicInfoService"
    
    @classmethod  
    def from_secrets(cls) -> 'APIConfig':
        """Streamlit secrets에서 API 설정 로드"""
        try:
            api_secrets = st.secrets["narajangter"]
            return cls(
                service_key=api_secrets["service_key"],
                base_url=api_secrets.get("base_url", "http://apis.data.go.kr/1230000/ad/BidPublicInfoService")
            )
        except KeyError as e:
            st.error(f"나라장터 API 설정을 찾을 수 없습니다: {e}")
            st.stop()

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
        """Streamlit secrets에서 전체 설정 로드"""
        try:
            app_secrets = st.secrets.get("app", {})
            return cls(
                debug=app_secrets.get("debug", False),
                page_size=int(app_secrets.get("page_size", "10")),
                max_search_results=int(app_secrets.get("max_search_results", "100")),
                vector_similarity_threshold=float(app_secrets.get("vector_similarity_threshold", "0.3")),
                database=DatabaseConfig.from_secrets(),
                openai=OpenAIConfig.from_secrets(),
                api=APIConfig.from_secrets()
            )
        except Exception as e:
            st.error(f"애플리케이션 설정 로드 실패: {e}")
            st.stop()

# 전역 설정 인스턴스 (캐시된)
@st.cache_resource
def get_app_config() -> AppConfig:
    """앱 설정을 로드하고 캐시"""
    return AppConfig.load()

# secrets.toml 템플릿 내용
SECRETS_TEMPLATE = '''# .streamlit/secrets.toml
# Streamlit Community Cloud에서 사용할 설정

[database]
url = "postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres"
host = "db.your-project.supabase.co"
port = "5432"
database = "postgres"
user = "postgres"
password = "your-supabase-password"

[openai]
api_key = "sk-proj-your-openai-api-key"
model = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"

[narajangter]
service_key = "your-narajangter-api-key"
base_url = "http://apis.data.go.kr/1230000/ad/BidPublicInfoService"

[app]
debug = false
page_size = "10"
max_search_results = "100"
vector_similarity_threshold = "0.3"
'''

def check_secrets():
    """필수 secrets 확인"""
    required_secrets = {
        "database": ["url", "password"],
        "openai": ["api_key"],
        "narajangter": ["service_key"]
    }
    
    missing_secrets = []
    
    for section, keys in required_secrets.items():
        if section not in st.secrets:
            missing_secrets.append(f"섹션 '{section}' 전체")
            continue
            
        for key in keys:
            if key not in st.secrets[section]:
                missing_secrets.append(f"{section}.{key}")
    
    if missing_secrets:
        st.error("다음 설정이 누락되었습니다:")
        for secret in missing_secrets:
            st.error(f"- {secret}")
        st.info("Streamlit Community Cloud의 'Secrets' 섹션에서 설정을 추가해주세요.")
        st.stop()
    
    return True
