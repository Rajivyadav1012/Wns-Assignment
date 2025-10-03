"""
Configuration Settings for Multimodal RAG Chatbot
Step 1: Project Setup
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application configuration settings"""
    
    # ==================== API KEYS ====================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    
    # ==================== OBSERVABILITY ====================
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "multimodal-rag-chatbot")
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    ENABLE_TRACING: bool = True
    
    # ==================== APPLICATION ====================
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    
    # ==================== PATHS ====================
    BASE_DIR: Path = Path(__file__).parent.parent
    VECTOR_DB_PATH: Path = Path(os.getenv("VECTOR_DB_PATH", "./data/vector_db"))
    MEMORY_DB_PATH: Path = Path(os.getenv("MEMORY_DB_PATH", "./data/memory.db"))
    DOCUMENTS_PATH: Path = Path(os.getenv("DOCUMENTS_PATH", "./data/documents"))
    LOGS_PATH: Path = BASE_DIR / "logs"
    
    # ==================== RAG SETTINGS ====================
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # ==================== MEMORY SETTINGS ====================
    MAX_MEMORY_MESSAGES: int = 10
    ENABLE_MEMORY: bool = True
    
    # ==================== GUARDRAILS ====================
    ENABLE_GUARDRAILS: bool = True
    TOXICITY_THRESHOLD: float = 0.7
    NSFW_THRESHOLD: float = 0.7
    
    # ==================== WEB SEARCH ====================
    ENABLE_WEB_SEARCH: bool = True
    MAX_SEARCH_RESULTS: int = 5
    
    # ==================== SUPPORTED MODELS ====================
    SUPPORTED_LLM_PROVIDERS: Dict[str, List[str]] = {
        "OpenAI": [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo"
        ],
        "Anthropic": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        "Google": [
            "gemini-pro",
            "gemini-pro-vision"
        ],
        "Groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "llama3-70b-8192",
            "llama3-8b-8192"
            "gemma-7b-it"
        ]
    }
    
    # Default selections
    DEFAULT_LLM_PROVIDER: str = "OpenAI"
    DEFAULT_LLM_MODEL: str = "gpt-3.5-turbo"
    
    # ==================== RAG VARIANTS ====================
    RAG_VARIANTS: List[str] = [
        "Standard RAG",
        "Knowledge Graph RAG",
        "Hybrid RAG"
    ]
    DEFAULT_RAG_VARIANT: str = "Standard RAG"
    
    # ==================== FILE SUPPORT ====================
    SUPPORTED_TEXT_FILES: List[str] = [".txt", ".pdf", ".docx"]
    SUPPORTED_IMAGE_FILES: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]
    
    # ==================== KNOWLEDGE GRAPH ====================
    KG_MAX_ENTITIES: int = 50
    KG_MIN_ENTITY_FREQUENCY: int = 2
    
    # ==================== PROMPTS ====================
    SYSTEM_PROMPT: str = """You are a helpful AI assistant with access to documents and web search. 
Always cite your sources when answering questions. 
If you don't know the answer, say so clearly."""
    
    RAG_PROMPT_TEMPLATE: str = """Answer the following question based on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer with citations:"""
    
    @classmethod
    def initialize_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.VECTOR_DB_PATH,
            cls.DOCUMENTS_PATH,
            cls.MEMORY_DB_PATH.parent,
            cls.LOGS_PATH
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Check which API keys are configured"""
        def is_valid_key(key: str) -> bool:
            """Check if API key is valid (not empty and not placeholder)"""
            if not key or len(key) < 10:
                return False
            if "your-" in key.lower() or "here" in key.lower():
                return False
            return True
        
        return {
            "OpenAI": is_valid_key(cls.OPENAI_API_KEY),
            "Anthropic": is_valid_key(cls.ANTHROPIC_API_KEY),
            "Google": is_valid_key(cls.GOOGLE_API_KEY),
            "Groq": is_valid_key(cls.GROQ_API_KEY),
            "Serper": is_valid_key(cls.SERPER_API_KEY),
            "Langsmith": is_valid_key(cls.LANGSMITH_API_KEY)
        }
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of LLM providers with valid API keys"""
        api_keys = cls.validate_api_keys()
        available = []
        
        if api_keys["OpenAI"]:
            available.append("OpenAI")
        if api_keys["Anthropic"]:
            available.append("Anthropic")
        if api_keys["Google"]:
            available.append("Google")
        if api_keys["Groq"]:
            available.append("Groq")
        
        return available if available else ["OpenAI"]

# Initialize directories on import
Settings.initialize_directories()

# Global settings instance
settings = Settings()