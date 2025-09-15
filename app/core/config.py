"""
Application configuration using Pydantic Settings
"""

from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    API_VERSION: str = Field(default="v1", env="API_VERSION")
    API_TITLE: str = Field(default="PrepSmart Scoring API", env="API_TITLE")
    API_DESCRIPTION: str = Field(default="AI-powered PTE practice scoring system", env="API_DESCRIPTION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_ORG_ID: Optional[str] = Field(default=None, env="OPENAI_ORG_ID")
    OPENAI_MODEL_GPT: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL_GPT")
    OPENAI_MODEL_WHISPER: str = Field(default="whisper-1", env="OPENAI_MODEL_WHISPER")
    OPENAI_MAX_TOKENS: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    OPENAI_TEMPERATURE: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    
    # Database Configuration (Optional for now)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # WordPress Integration
    WORDPRESS_BASE_URL: Optional[str] = Field(default=None, env="WORDPRESS_BASE_URL")
    WORDPRESS_API_KEY: Optional[str] = Field(default=None, env="WORDPRESS_API_KEY")
    WORDPRESS_DB_HOST: Optional[str] = Field(default="localhost", env="WORDPRESS_DB_HOST")
    WORDPRESS_DB_NAME: Optional[str] = Field(default="wordpress", env="WORDPRESS_DB_NAME")
    WORDPRESS_DB_USER: Optional[str] = Field(default="root", env="WORDPRESS_DB_USER")
    WORDPRESS_DB_PASSWORD: Optional[str] = Field(default="", env="WORDPRESS_DB_PASSWORD")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    CORS_CREDENTIALS: bool = Field(default=True, env="CORS_CREDENTIALS")
    CORS_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        env="CORS_METHODS"
    )
    CORS_HEADERS: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    # File Storage
    STORAGE_PROVIDER: str = Field(default="local", env="STORAGE_PROVIDER")  # local, s3, r2
    LOCAL_STORAGE_PATH: str = Field(default="./uploads", env="LOCAL_STORAGE_PATH")
    STATIC_FILES_URL: str = Field(default="/static", env="STATIC_FILES_URL")
    MAX_FILE_SIZE: int = Field(default=26214400, env="MAX_FILE_SIZE")  # 25MB
    
    # Audio Processing
    WHISPER_MODEL: str = Field(default="base", env="WHISPER_MODEL")
    WHISPER_DEVICE: str = Field(default="cpu", env="WHISPER_DEVICE")
    ENABLE_WORD_TIMESTAMPS: bool = Field(default=True, env="ENABLE_WORD_TIMESTAMPS")
    
    # Performance
    ENABLE_CACHING: bool = Field(default=True, env="ENABLE_CACHING")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")
    WARM_UP_MODELS: bool = Field(default=False, env="WARM_UP_MODELS")
    SKIP_MODEL_LOADING: bool = Field(default=False, env="SKIP_MODEL_LOADING")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="./logs/api.log", env="LOG_FILE")
    ENABLE_ACCESS_LOG: bool = Field(default=True, env="ENABLE_ACCESS_LOG")
    
    # Documentation
    DOCS_URL: Optional[str] = Field(default="/docs", env="DOCS_URL")
    REDOC_URL: Optional[str] = Field(default="/redoc", env="REDOC_URL")
    OPENAPI_URL: Optional[str] = Field(default="/openapi.json", env="OPENAPI_URL")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Allow extra fields for forward compatibility
        extra = "allow"


# Create settings instance
settings = Settings()

# Create necessary directories
Path(settings.LOCAL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)