#!/usr/bin/env python3
"""
Write Essay API Endpoint
Handles PTE Write Essay scoring (200-300 words)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging

# Import the essay scorer
from app.services.scoring.write_essay_scorer import get_essay_scorer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/writing",
    tags=["Writing - Essay"]
)

class WriteEssayRequest(BaseModel):
    """Request model for essay scoring"""
    essay_prompt: str = Field(..., description="Essay question/topic")
    user_essay: str = Field(..., description="Student's essay (200-300 words)")
    
class ComponentScore(BaseModel):
    """Individual component score"""
    score: float
    max_score: float
    feedback: str
    errors: List[str] = []
    suggestions: List[str] = []

class WriteEssayResponse(BaseModel):
    """Response model for essay scoring"""
    success: bool
    total_score: float = Field(..., description="Total score out of 26")
    percentage: int = Field(..., description="Percentage score")
    band: str = Field(..., description="Performance band")
    
    # Component scores
    content: ComponentScore
    formal_requirements: ComponentScore
    development_coherence: ComponentScore
    grammar: ComponentScore
    linguistic_range: ComponentScore
    vocabulary_range: ComponentScore
    spelling: ComponentScore
    
    # Overall feedback
    strengths: List[str] = []
    improvements: List[str] = []
    harsh_assessment: str = ""
    
    # Detailed errors
    all_errors: Dict[str, List[str]] = {}
    
    # API metrics
    api_cost: Optional[float] = None

@router.post("/write-essay", response_model=WriteEssayResponse)
async def score_write_essay(request: WriteEssayRequest):
    """
    Score a PTE Write Essay task
    
    - **essay_prompt**: The essay question or topic
    - **user_essay**: Student's written essay (should be 200-300 words)
    
    Returns comprehensive scoring across 7 components (26 points total)
    """
    try:
        logger.info("="*50)
        logger.info("📝 Write Essay scoring request received")
        logger.info(f"Prompt: {request.essay_prompt[:100]}...")
        logger.info(f"Essay length: {len(request.user_essay.split())} words")
        
        # Get scorer instance
        scorer = get_essay_scorer()
        
        # Score the essay
        result = scorer.score_essay(
            user_essay=request.user_essay,
            essay_prompt=request.essay_prompt
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Scoring failed"))
        
        # Extract scores and feedback
        scores = result["scores"]
        component_feedback = result["component_feedback"]
        errors = result["errors"]
        suggestions = result.get("suggestions", {})
        
        # Build response
        response = WriteEssayResponse(
            success=True,
            total_score=result["total_score"],
            percentage=result["percentage"],
            band=result["band"],
            
            # Component scores with detailed feedback
            content=ComponentScore(
                score=scores["content"],
                max_score=6,
                feedback=component_feedback["content"],
                errors=suggestions.get("content", []),
                suggestions=[s.get("suggestion", "") for s in suggestions.get("content", [])]
            ),
            formal_requirements=ComponentScore(
                score=scores["form"],
                max_score=2,
                feedback=component_feedback["form"],
                errors=errors.get("form", []),
                suggestions=["Maintain 200-300 words", "Use proper paragraph structure"]
            ),
            development_coherence=ComponentScore(
                score=scores["development"],
                max_score=6,
                feedback=component_feedback["development"],
                errors=errors.get("coherence", []),
                suggestions=[s.get("suggestion", "") for s in suggestions.get("coherence", [])]
            ),
            grammar=ComponentScore(
                score=scores["grammar"],
                max_score=2,
                feedback=component_feedback["grammar"],
                errors=errors.get("grammar", [])[:5],  # Limit to 5 errors
                suggestions=[s.get("correction", "") for s in suggestions.get("grammar", [])][:5]
            ),
            linguistic_range=ComponentScore(
                score=scores["linguistic"],
                max_score=6,
                feedback=component_feedback["linguistic"],
                errors=errors.get("linguistic", []),
                suggestions=["Vary sentence structures", "Use complex grammatical forms"]
            ),
            vocabulary_range=ComponentScore(
                score=scores["vocabulary"],
                max_score=2,
                feedback=component_feedback["vocabulary"],
                errors=errors.get("vocabulary", [])[:5],
                suggestions=[s.get("correction", "") for s in suggestions.get("vocabulary", [])][:5]
            ),
            spelling=ComponentScore(
                score=scores["spelling"],
                max_score=2,
                feedback=component_feedback["spelling"],
                errors=errors.get("spelling", [])[:5],
                suggestions=["Check spelling carefully", "Maintain consistency"]
            ),
            
            # Overall feedback
            strengths=result.get("strengths", []),
            improvements=result.get("improvements", []),
            harsh_assessment=result.get("harsh_assessment", ""),
            
            # All errors for reference
            all_errors=errors,
            
            # API cost
            api_cost=result.get("api_cost", 0.0)
        )
        
        logger.info(f"✅ Essay scored: {result['total_score']}/26 ({result['percentage']}%)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Essay scoring error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/write-essay/status")
async def get_essay_scorer_status():
    """Check if essay scorer is initialized and ready"""
    try:
        scorer = get_essay_scorer()
        return {
            "status": "ready",
            "gpt_available": scorer.use_gpt,
            "components": {
                "gector": scorer.gector_model is not None,
                "language_tool": scorer.language_tool is not None,
                "spacy": scorer.nlp is not None,
                "sentence_transformer": scorer.sentence_model is not None,
                "keybert": scorer.keybert is not None
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }