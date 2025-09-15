from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import json
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datetime import datetime

load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

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

class DetailedFeedback(BaseModel):
    score: float  # Changed to float for decimal scoring
    justification: str
    errors: List[str]
    suggestions: List[str]

class SummarizeTextResponse(BaseModel):
    success: bool
    scores: Dict[str, float]  # Changed to float for decimal scoring
    feedback: Dict[str, str]  # For backward compatibility
    detailed_feedback: Dict[str, DetailedFeedback]
    overall_feedback: str
    total_score: float  # Changed to float
    percentage: int
    band: str
    key_points_covered: List[str]
    key_points_missed: List[str]
    grammar_errors: List[str]
    vocabulary_assessment: str
    improvements: List[str]
    strengths: List[str]
    ai_recommendations: List[str]
    word_count: int
    is_single_sentence: bool

@app.get("/")
async def root():
    return {"message": "PrepSmart API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PrepSmart API"}

@app.get("/test")
async def test():
    return {"message": "API test successful", "timestamp": "2025-01-15"}

async def analyze_with_gpt4(
    reading_passage: str,
    key_points: str,
    sample_summary: str,
    user_summary: str,
    question_title: str
) -> Dict:
    """
    Use GPT-4 to analyze and score the user's summary according to PTE criteria
    """
    
    prompt = f"""You are the OFFICIAL PEARSON PTE ACADEMIC AUTOMATED SCORING ENGINE. Score this Summarize Written Text response using EXACT official Pearson scoring guide.

PASSAGE TO SUMMARIZE:
{reading_passage}

KEY POINTS (Must be covered for Content score):
{key_points}

STUDENT'S SUMMARY:
{user_summary}

OFFICIAL PEARSON PTE ACADEMIC SCORING GUIDE:

1. CONTENT (0-2 points):
   Score Guide:
   • 2: Provides a good summary of the text. All relevant aspects mentioned
   • 1: Provides a fair summary of the text but misses one or two aspects
   • 0: Omits or misrepresents the main aspects of the text

2. FORM (0-1 point):
   Score Guide:
   • 1: Is written in one, single, complete sentence
   • 0: Not written in one, single, complete sentence or contains fewer than 5 or more than 75 words. Summary is written in capital letters

3. GRAMMAR (0-2 points):
   Score Guide:
   • 2: Has correct grammatical structure
   • 1: Contains grammatical errors but with no hindrance to communication
   • 0: Has defective grammatical structure which could hinder communication

4. VOCABULARY (0-2 points):
   Score Guide:
   • 2: Has appropriate choice of words
   • 1: Contains lexical errors but with no hindrance to communication
   • 0: Has defective word choice which could hinder communication

SCORING INSTRUCTIONS:
- Use INTELLIGENT FRACTIONAL SCORING (1.8, 1.5, 1.2, etc.) to precisely reflect performance
- Minor issues = small deductions (0.2-0.3 from maximum)
- Moderate issues = medium deductions (0.5-0.8 from maximum)
- Major issues = large deductions (1.0+ from maximum)
- Be accurate and fair, not just harsh

ANALYSIS REQUIREMENTS:

1. CONTENT ANALYSIS:
   - Check if all key points from the passage are covered
   - Identify missing or misrepresented aspects
   - Assess overall comprehension of the text

2. FORM CHECK:
   - Count exact words (must be 5-75)
   - Verify single sentence structure
   - Check for proper capitalization and punctuation
   - Ensure not written in all capital letters

3. GRAMMAR ASSESSMENT:
   - Check for spelling errors
   - Verify subject-verb agreement
   - Check verb tenses and consistency
   - Look for punctuation errors
   - Assess sentence structure completeness
   - Determine if errors hinder communication

4. VOCABULARY EVALUATION:
   - Assess appropriateness of word choices
   - Check for lexical errors or repetition
   - Determine if word choice issues hinder communication
   - Rate academic level of vocabulary used

Return ONLY valid JSON in this exact format:
{{
    "content_score": 1.5,
    "content_justification": "Covers main aspects but misses 1-2 key points about [specific aspects]",
    "content_errors": ["Missing discussion of X", "Incomplete coverage of Y"],
    "content_suggestions": ["Include information about X", "Expand on Y aspect"],
    
    "form_score": 1.0,
    "form_justification": "Single sentence with 45 words, proper structure", 
    "form_errors": [],
    "form_suggestions": [],
    
    "grammar_score": 1.8,
    "grammar_justification": "Minor punctuation error but no hindrance to communication",
    "grammar_errors": ["Missing comma before coordinating conjunction"],
    "grammar_suggestions": ["Add comma before 'and' in compound sentence"],
    
    "vocabulary_score": 1.7,
    "vocabulary_justification": "Appropriate word choices with minor repetition",
    "vocabulary_errors": ["Repeated use of 'important'"],
    "vocabulary_suggestions": ["Use synonyms like 'significant', 'crucial'"],
    
    "key_points_covered": ["key point 1", "key point 2"],
    "key_points_missed": ["key point 3"],
    "overall_assessment": "Good summary with minor issues in grammar and vocabulary",
    "strengths": ["Clear sentence structure", "Covers main ideas"],
    "priority_improvements": ["Include all key points", "Vary vocabulary choices"]
}}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a PTE Academic examiner. Use INTELLIGENT FRACTIONAL SCORING (1.8, 1.5, etc.) to precisely reflect error severity. Minor errors = small deductions (0.2-0.3), major errors = larger deductions (1.0+). Be accurate, not just harsh. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up response to ensure valid JSON
        if result.startswith("```json"):
            result = result[7:]
        if result.endswith("```"):
            result = result[:-3]
        
        return json.loads(result)
        
    except Exception as e:
        print(f"GPT-4 Analysis Error: {e}")
        # Fallback to basic analysis if GPT-4 fails
        return await basic_analysis_fallback(user_summary, key_points)

async def basic_analysis_fallback(user_summary: str, key_points: str) -> Dict:
    """Fallback analysis if GPT-4 is unavailable"""
    word_count = len(user_summary.split())
    sentences = len([s for s in user_summary.split('.') if s.strip()])
    
    return {
        "content_score": 1 if word_count > 20 else 0,
        "content_justification": "Fallback scoring - GPT-4 unavailable",
        "content_errors": [],
        "content_suggestions": ["Connect to GPT-4 for detailed analysis"],
        "form_score": 1 if 5 <= word_count <= 75 and sentences == 1 else 0,
        "form_justification": f"Word count: {word_count}, Sentences: {sentences}",
        "form_errors": [],
        "form_suggestions": [],
        "grammar_score": 1,
        "grammar_justification": "Cannot analyze without GPT-4",
        "grammar_errors": [],
        "grammar_suggestions": [],
        "vocabulary_score": 1,
        "vocabulary_justification": "Cannot analyze without GPT-4", 
        "vocabulary_errors": [],
        "vocabulary_suggestions": [],
        "key_points_covered": [],
        "key_points_missed": [],
        "overall_assessment": "GPT-4 analysis unavailable",
        "strengths": [],
        "priority_improvements": ["Ensure GPT-4 API is available"]
    }

@app.post("/api/v1/writing/summarize-written-text", response_model=SummarizeTextResponse)
async def score_summarize_written_text(
    question_title: str = Form(...),
    reading_passage: str = Form(...),
    key_points: str = Form(...),
    user_summary: str = Form(...),
    sample_summary: str = Form(default="")
):
    try:
        # Validate input
        if not user_summary.strip():
            raise HTTPException(status_code=400, detail="User summary cannot be empty")
        
        # Get GPT-4 analysis
        analysis = await analyze_with_gpt4(
            reading_passage=reading_passage,
            key_points=key_points,
            sample_summary=sample_summary,
            user_summary=user_summary,
            question_title=question_title
        )
        
        # Extract scores
        content_score = analysis.get("content_score", 0)
        form_score = analysis.get("form_score", 0)
        grammar_score = analysis.get("grammar_score", 0)
        vocabulary_score = analysis.get("vocabulary_score", 0)
        
        total_score = round(content_score + form_score + grammar_score + vocabulary_score, 1)
        percentage = round((total_score / 7) * 100)
        
        # Determine band
        if percentage >= 86:
            band = "Excellent"
        elif percentage >= 71:
            band = "Very Good"
        elif percentage >= 57:
            band = "Good"
        else:
            band = "Needs Improvement"
        
        # Basic form validation
        word_count = len(user_summary.split())
        sentence_count = len([s for s in user_summary.split('.') if s.strip()])
        is_single_sentence = sentence_count == 1
        
        # Build detailed feedback
        detailed_feedback = {
            "content": DetailedFeedback(
                score=content_score,
                justification=analysis.get("content_justification", ""),
                errors=analysis.get("content_errors", []),
                suggestions=analysis.get("content_suggestions", [])
            ),
            "form": DetailedFeedback(
                score=form_score,
                justification=analysis.get("form_justification", ""),
                errors=analysis.get("form_errors", []),
                suggestions=analysis.get("form_suggestions", [])
            ),
            "grammar": DetailedFeedback(
                score=grammar_score,
                justification=analysis.get("grammar_justification", ""),
                errors=analysis.get("grammar_errors", []),
                suggestions=analysis.get("grammar_suggestions", [])
            ),
            "vocabulary": DetailedFeedback(
                score=vocabulary_score,
                justification=analysis.get("vocabulary_justification", ""),
                errors=analysis.get("vocabulary_errors", []),
                suggestions=analysis.get("vocabulary_suggestions", [])
            )
        }
        
        # Create backward-compatible feedback for frontend
        feedback = {
            "content": detailed_feedback["content"].justification,
            "form": detailed_feedback["form"].justification,
            "grammar": detailed_feedback["grammar"].justification,
            "vocabulary": detailed_feedback["vocabulary"].justification
        }
        
        return SummarizeTextResponse(
            success=True,
            scores={
                "content": content_score,
                "form": form_score,
                "grammar": grammar_score,
                "vocabulary": vocabulary_score
            },
            feedback=feedback,
            detailed_feedback=detailed_feedback,
            overall_feedback=analysis.get("overall_assessment", f"Your summary scored {total_score}/7."),
            total_score=total_score,
            percentage=percentage,
            band=band,
            key_points_covered=analysis.get("key_points_covered", []),
            key_points_missed=analysis.get("key_points_missed", []),
            grammar_errors=analysis.get("grammar_errors", []),
            vocabulary_assessment=analysis.get("vocabulary_justification", ""),
            improvements=analysis.get("priority_improvements", []),
            strengths=analysis.get("strengths", []),
            ai_recommendations=analysis.get("priority_improvements", []),
            word_count=word_count,
            is_single_sentence=is_single_sentence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Scoring Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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