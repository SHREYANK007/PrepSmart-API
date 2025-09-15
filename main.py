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
    score: int
    justification: str
    errors: List[str]
    suggestions: List[str]

class SummarizeTextResponse(BaseModel):
    success: bool
    scores: Dict[str, int]
    feedback: Dict[str, str]  # For backward compatibility
    detailed_feedback: Dict[str, DetailedFeedback]
    overall_feedback: str
    total_score: int
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
    
    prompt = f"""You are an official PTE Academic examiner with 10+ years of experience. Score this Summarize Written Text response STRICTLY according to Pearson's official PTE marking criteria.

QUESTION: {question_title}

READING PASSAGE:
{reading_passage}

KEY POINTS TO COVER:
{key_points}

SAMPLE CORRECT ANSWER:
{sample_summary}

USER'S SUMMARY TO EVALUATE:
{user_summary}

OFFICIAL PTE MARKING CRITERIA:

1. CONTENT (0-2 points):
   - 2 points: All key points covered, main idea clearly present
   - 1 point: Most key points covered, main idea somewhat present  
   - 0 points: Key points missed, main idea unclear/missing

2. FORM (0-1 point):
   - 1 point: Single sentence, 5-75 words, starts with capital, ends with period
   - 0 points: Multiple sentences OR wrong word count OR incorrect punctuation

3. GRAMMAR (0-2 points) - BE EXTREMELY STRICT:
   - 2 points: PERFECT grammar - no errors whatsoever (missing comma = 1 point)
   - 1 point: 1-2 minor errors (missing comma, wrong tense, article error)
   - 0 points: 3+ errors OR any major error affecting meaning

   GRAMMAR ERRORS TO CHECK STRICTLY:
   • Missing commas (especially before coordinating conjunctions)
   • Wrong verb tenses or subject-verb disagreement
   • Missing or incorrect articles (a, an, the)
   • Spelling mistakes of any kind
   • Wrong prepositions
   • Incomplete sentences or run-on sentences
   • Wrong word forms (adjective vs adverb)
   • Capitalization errors
   • Missing periods or wrong punctuation

4. VOCABULARY (0-2 points):
   - 2 points: Appropriate academic vocabulary, excellent word choice
   - 1 point: Adequate vocabulary with some basic words
   - 0 points: Very limited, repetitive, or inappropriate vocabulary

CRITICAL ANALYSIS REQUIRED - BE RUTHLESS:

1. FORM CHECK:
   - Count exact words (must be 5-75)
   - Verify single sentence (no period in middle, no conjunctions creating new clauses)
   - Check capitalization and ending punctuation

2. GRAMMAR CHECK (FIND EVERY ERROR):
   - Scan every word for spelling mistakes
   - Check every comma placement (especially before "and", "but", "so", "or")
   - Verify subject-verb agreement
   - Check article usage (a/an/the)
   - Verify correct verb tenses
   - Check preposition usage
   - Look for wrong word forms (e.g., "quick" vs "quickly")
   - Check for incomplete thoughts or run-ons

3. CONTENT ANALYSIS:
   - Match user summary against each key point
   - Identify what's covered vs completely missing
   - Check if main idea is captured

4. VOCABULARY ASSESSMENT:
   - Rate academic level of word choices
   - Check for repetition or basic vocabulary
   - Note any inappropriate word usage

BE EXTREMELY HARSH - PTE DEDUCTS FOR MINOR ERRORS!

Return ONLY valid JSON in this exact format:
{{
    "content_score": 0-2,
    "content_justification": "detailed explanation",
    "content_errors": ["specific issues found"],
    "content_suggestions": ["specific improvements"],
    
    "form_score": 0-1,
    "form_justification": "detailed explanation", 
    "form_errors": ["specific issues found"],
    "form_suggestions": ["specific improvements"],
    
    "grammar_score": 0-2,
    "grammar_justification": "detailed explanation of EVERY error found",
    "grammar_errors": ["EXAMPLE: 'Word 5: missing comma before and', 'Word 12: wrong tense should be past', 'Word 8: spelling error - enviroment should be environment'"],
    "grammar_suggestions": ["specific fixes with exact positions"],
    
    "vocabulary_score": 0-2,
    "vocabulary_justification": "detailed explanation",
    "vocabulary_errors": ["vocabulary issues found"],
    "vocabulary_suggestions": ["specific vocabulary improvements"],
    
    "key_points_covered": ["list of covered key points"],
    "key_points_missed": ["list of missed key points"],
    "overall_assessment": "comprehensive summary of performance",
    "strengths": ["specific strengths identified"],
    "priority_improvements": ["most important areas to focus on"]
}}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an EXTREMELY STRICT PTE Academic examiner. Deduct points for ANY grammar error - even missing commas. Be ruthless and thorough. Respond with valid JSON only."},
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
        
        total_score = content_score + form_score + grammar_score + vocabulary_score
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