"""
PrepSmart API - Main Application Entry Point
Configured for cross-server communication with WordPress
"""

import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import settings
from app.api.v1 import api_router

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting PrepSmart API...")
    logger.info(f"OpenAI API Key configured: {'Yes' if settings.OPENAI_API_KEY else 'No'}")
    logger.info(f"CORS Origins: {settings.CORS_ORIGINS}")
    
    # Test OpenAI connection
    try:
        from app.services.scoring.gpt_service import GPTService
        gpt_service = GPTService()
        logger.info("GPT Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GPT Service: {e}")
    
    logger.info("PrepSmart API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down PrepSmart API...")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    openapi_url=settings.OPENAPI_URL,
    lifespan=lifespan,
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CRITICAL: CORS configuration for cross-server communication
# This allows your WordPress site to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual WordPress domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} from {client_ip}"
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} in {process_time:.4f}s"
    )
    
    # Add process time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True
    )
    
    if settings.DEBUG:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred"
            }
        )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check(request: Request) -> Dict[str, Any]:
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": "development" if settings.DEBUG else "production",
        "openai_configured": bool(settings.OPENAI_API_KEY)
    }


# Test endpoint for frontend connection
@app.get("/test", tags=["Test"])
async def test_connection() -> Dict[str, str]:
    """Test endpoint to verify connection from WordPress."""
    return {
        "status": "success",
        "message": "PrepSmart API is running",
        "version": settings.API_VERSION,
        "cors_enabled": True,
        "timestamp": datetime.utcnow().isoformat()
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "PrepSmart API",
        "version": settings.API_VERSION,
        "docs": f"{settings.DOCS_URL}" if settings.DEBUG else "API Documentation",
        "health": "/health",
        "test": "/test"
    }


# Include API router
app.include_router(api_router, prefix=f"/api/{settings.API_VERSION}")


# Mount static files if using local storage
if settings.STORAGE_PROVIDER == "local":
    if not os.path.exists(settings.LOCAL_STORAGE_PATH):
        os.makedirs(settings.LOCAL_STORAGE_PATH)
    
    app.mount(
        settings.STATIC_FILES_URL,
        StaticFiles(directory=settings.LOCAL_STORAGE_PATH),
        name="static"
    )


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )