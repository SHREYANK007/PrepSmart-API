from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="PrepSmart Scoring API",
    description="AI-powered PTE practice scoring system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pte.prepsmart.au", "https://dashboard.prepsmart.au", "https://preppte.com", "http://pte.prepsmart.au", "http://dashboard.prepsmart.au", "http://82.29.167.191", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class SummarizeTextResponse(BaseModel):
    success: bool
    scores: Dict[str, int]
    feedback: Dict[str, str]
    overall_feedback: str
    total_score: int
    percentage: int
    band: str
    improvements: list
    strengths: list
    ai_recommendations: list
    ai_suggestions: Dict[str, str]

@app.get("/")
async def root():
    return {"message": "PrepSmart API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PrepSmart API"}

@app.get("/test")
async def test():
    return {"message": "API test successful", "timestamp": "2025-01-15"}

@app.post("/api/v1/writing/summarize-written-text", response_model=SummarizeTextResponse)
async def score_summarize_written_text(
    question_title: str = Form(...),
    reading_passage: str = Form(...),
    key_points: str = Form(...),
    user_summary: str = Form(...)
):
    # Simple scoring for now - we'll add GPT-4 later
    word_count = len(user_summary.split())
    
    # PTE Scoring logic - 4 criteria only (Content, Form, Grammar, Vocabulary)
    # Content: 0-2 points
    content_score = 2 if word_count > 30 else 1 if word_count > 15 else 0
    
    # Form: 0-1 point (word count must be 5-75)
    form_score = 1 if 5 <= word_count <= 75 else 0
    
    # Grammar: 0-2 points (combined with spelling)
    grammar_score = 2 if word_count > 20 else 1
    
    # Vocabulary: 0-2 points
    vocabulary_score = 2 if word_count > 25 else 1 if word_count > 10 else 0
    
    scores = {
        "content": content_score,
        "form": form_score, 
        "grammar": grammar_score,
        "vocabulary": vocabulary_score
    }
    
    # Total score out of 7 (2+1+2+2)
    total_score = sum(scores.values())
    percentage = round((total_score / 7) * 100)
    
    if percentage >= 86:
        band = "Excellent"
    elif percentage >= 71:
        band = "Very Good"
    elif percentage >= 57:
        band = "Good"
    else:
        band = "Needs Improvement"
    
    # Generate detailed dummy data for testing
    ai_recommendations = []
    ai_suggestions = {}
    
    # Content-based recommendations
    if content_score < 2:
        ai_recommendations.extend([
            "Include more main ideas from the passage",
            "Focus on the central theme and supporting arguments",
            "Ensure you capture the cause-effect relationships mentioned"
        ])
        ai_suggestions["content"] = "Try to identify 3-4 key points from the passage and include them in your summary."
    else:
        ai_recommendations.append("Excellent job capturing the main ideas")
        ai_suggestions["content"] = "Your content coverage is strong. Maintain this level of detail."
    
    # Form-based recommendations
    if form_score == 0:
        ai_recommendations.append(f"Adjust word count (currently {word_count} words, needs 5-75)")
        ai_suggestions["form"] = "Your summary should be between 5-75 words. Practice being more concise."
    else:
        ai_suggestions["form"] = "Word count is within the required range. Good job!"
    
    # Grammar recommendations (includes spelling in PTE)
    if grammar_score < 2:
        ai_recommendations.append("Improve grammar and check spelling")
        ai_suggestions["grammar"] = "Check for grammatical errors and spelling mistakes. Use proper punctuation and sentence structures."
    else:
        ai_suggestions["grammar"] = "Grammar and spelling are accurate. Good use of sentence structures."
    
    # Vocabulary recommendations
    if vocabulary_score < 2:
        ai_recommendations.append("Use more academic vocabulary")
        ai_suggestions["vocabulary"] = "Replace basic words with academic alternatives (e.g., 'show' → 'demonstrate', 'important' → 'significant')."
    else:
        ai_suggestions["vocabulary"] = "Good use of academic vocabulary. Continue using precise terminology."
    
    return SummarizeTextResponse(
        success=True,
        scores=scores,
        feedback={
            "content": "Good coverage of key points" if content_score == 2 else "Need more key points" if content_score == 1 else "Missing key points",
            "form": "Proper word count" if form_score == 1 else "Check word count (5-75 words)",
            "grammar": "Grammar and spelling look good" if grammar_score == 2 else "Some grammar/spelling issues",
            "vocabulary": "Good vocabulary usage" if vocabulary_score == 2 else "Try varied vocabulary" if vocabulary_score == 1 else "Limited vocabulary"
        },
        overall_feedback=f"Your summary scored {total_score}/7. " + (
            "Great job!" if total_score >= 6 else
            "Good effort, room for improvement." if total_score >= 4 else
            "Keep practicing to improve your score."
        ),
        total_score=total_score,
        percentage=percentage,
        band=band,
        improvements=["Focus on key points", "Check word count"] if total_score < 6 else [],
        strengths=["Clear writing"] if total_score >= 4 else ["Attempted response"],
        ai_recommendations=ai_recommendations,
        ai_suggestions=ai_suggestions
    )

if __name__ == "__main__":
    import uvicorn
    import ssl
    
    # Use Let's Encrypt certificates with ptewizard.com
    domain = "ptewizard.com"  # Using ptewizard.com domain
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8001,
            ssl_keyfile=f"/etc/letsencrypt/live/{domain}-0001/privkey.pem",
            ssl_certfile=f"/etc/letsencrypt/live/{domain}-0001/fullchain.pem"
        )
    except FileNotFoundError:
        print("SSL certificates not found, running HTTP")
        uvicorn.run(app, host="0.0.0.0", port=8001)