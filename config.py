import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # LLM Configuration
    LLM_TYPE: str = "google"  # openai, mistral, google, local
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RETRIEVAL_DOCS: int = 5
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Paths
    UPLOAD_DIR: str = "uploaded_docs"
    INDEX_DIR: str = "data/faiss_index"
    CACHE_DIR: str = "data/cache"
    
    # Web Search
    MAX_WEB_RESULTS: int = 3
    ENABLE_WEB_SEARCH: bool = True
    
    # UI Settings
    MAX_CHAT_HISTORY: int = 50

config = Config()

CACHE_DIR = "cache"