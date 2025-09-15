"""
API v1 Router
Combines all endpoint routers
"""

from fastapi import APIRouter

# Import task-specific routers
from app.api.v1.writing.summarize_text import router as swt_router

# Create main API router
api_router = APIRouter()

# Include all task routers
api_router.include_router(
    swt_router,
    prefix="/writing",
    tags=["Writing Tasks"]
)

# Future routers will be added here:
# api_router.include_router(essay_router, prefix="/writing", tags=["Writing Tasks"])
# api_router.include_router(read_aloud_router, prefix="/speaking", tags=["Speaking Tasks"])
# api_router.include_router(listening_router, prefix="/listening", tags=["Listening Tasks"])