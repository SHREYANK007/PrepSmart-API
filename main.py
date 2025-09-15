from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="PrepSmart Scoring API",
    description="AI-powered PTE practice scoring system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pte.prepsmart.au", "http://82.29.167.191", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "PrepSmart API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PrepSmart API"}

@app.get("/test")
async def test():
    return {"message": "API test successful", "timestamp": "2025-01-15"}

from app.api.v1.writing import summarize_text
app.include_router(summarize_text.router, prefix="/api/v1/writing", tags=["Writing"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)