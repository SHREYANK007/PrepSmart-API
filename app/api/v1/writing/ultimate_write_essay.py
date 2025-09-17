#!/usr/bin/env python3
"""
ULTIMATE Write Essay API Endpoint
SWT-style precision scoring with 100+ point English validation
"""

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import time

# Import the ultimate scorer
try:
    from app.services.scoring.ultimate_write_essay_scorer import score_ultimate_write_essay
    ULTIMATE_SCORER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Ultimate scorer unavailable: {e}")
    ULTIMATE_SCORER_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter()

class UltimateEssayResponse(BaseModel):
    """Ultimate response model with decimal precision scores"""
    success: bool
    scores: Dict[str, float]
    total_score: float
    percentage: int
    band: str
    word_count: int
    paragraph_count: int
    
    # Detailed component scores (APEUni style)
    component_scores: Dict[str, str]
    
    # All detected errors
    errors: Dict[str, List[str]]
    
    # GPT enhancements
    additional_errors_found: List[Dict[str, str]]
    ml_error_reclassifications: List[Dict[str, str]]
    
    # Feedback
    detailed_feedback: Dict[str, Any]
    verification_notes: str
    gpt_confidence: float
    
    # Metadata
    processing_time: float
    api_cost: float
    scorer_version: str
    errors_detected_total: int

@router.post("/ultimate", response_model=UltimateEssayResponse)
async def score_ultimate_essay(
    question_title: str = Form(...),
    essay_prompt: str = Form(...),
    essay_type: str = Form(...),
    key_arguments: str = Form(...),
    sample_essay: str = Form(default=""),
    user_essay: str = Form(...)
):
    """
    ULTIMATE Write Essay scoring endpoint with SWT-style precision
    
    Features:
    - Ultra-comprehensive spelling detection (catches "strickly" etc.)
    - Decimal precision scoring (1.8/2, 3.4/6 like APEUni) 
    - GPT as ultimate 100+ point English validator
    - ML error cross-validation and reclassification
    - SWT-style comprehensive final verification
    """
    try:
        if not ULTIMATE_SCORER_AVAILABLE:
            raise HTTPException(status_code=503, detail="Ultimate scorer not available")
        
        start_time = time.time()
        
        # Validate input
        if not user_essay or len(user_essay.strip()) < 50:
            raise HTTPException(status_code=400, detail="Essay too short (minimum 50 characters)")
        
        word_count = len(user_essay.split())
        if word_count < 150:
            raise HTTPException(status_code=400, detail=f"Essay too short: {word_count} words (minimum 150)")
        
        logger.info("🎯 Starting ULTIMATE Write Essay Scoring")
        
        # Call ultimate scorer
        result = score_ultimate_write_essay(user_essay, essay_prompt)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Ultimate scoring failed"))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        logger.info(f"✅ Ultimate scoring completed in {processing_time:.2f}s")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ultimate essay scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal scoring error: {str(e)}")

@router.post("/test-spelling")
async def test_spelling_detection(test_text: str = Form(...)):
    """
    Test endpoint for ultra-comprehensive spelling detection
    """
    try:
        if not ULTIMATE_SCORER_AVAILABLE:
            raise HTTPException(status_code=503, detail="Ultimate scorer not available")
        
        from app.services.scoring.ultimate_write_essay_scorer import get_ultimate_essay_scorer
        
        scorer = get_ultimate_essay_scorer()
        spelling_score, spelling_errors = scorer.ultra_spelling_check(test_text)
        
        return {
            "success": True,
            "text_analyzed": test_text,
            "spelling_score": f"{spelling_score.raw_score}/{spelling_score.max_score}",
            "errors_found": len(spelling_errors),
            "errors": [
                {
                    "error": err.error_text,
                    "correction": err.correction,
                    "rule": err.rule_violated,
                    "confidence": err.confidence
                } for err in spelling_errors
            ],
            "database_size": len(scorer.spelling_errors_database)
        }
        
    except Exception as e:
        logger.error(f"Spelling test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_ultimate_scoring_status():
    """Get the status of the ultimate scoring system"""
    
    status = {
        "ultimate_scorer_available": ULTIMATE_SCORER_AVAILABLE,
        "timestamp": time.time()
    }
    
    if ULTIMATE_SCORER_AVAILABLE:
        try:
            from app.services.scoring.ultimate_write_essay_scorer import get_ultimate_essay_scorer
            scorer = get_ultimate_essay_scorer()
            
            status.update({
                "gector_available": scorer.gector_model is not None,
                "language_tool_available": scorer.language_tool is not None,
                "sentence_transformer_available": scorer.sentence_model is not None,
                "gpt_available": scorer.use_gpt,
                "spelling_database_size": len(scorer.spelling_errors_database),
                "total_api_cost": scorer.total_api_cost,
                "features": [
                    "Ultra-comprehensive spelling detection",
                    "Decimal precision scoring (APEUni style)",
                    "GPT as 100+ point English validator",
                    "ML error cross-validation",
                    "SWT-style final verification"
                ]
            })
        except Exception as e:
            status["error"] = str(e)
    
    return status

@router.post("/validate-essay")
async def validate_with_gpt_only(
    essay_prompt: str = Form(...),
    user_essay: str = Form(...)
):
    """
    GPT-only validation endpoint for testing ultimate English inspection
    """
    try:
        if not ULTIMATE_SCORER_AVAILABLE:
            raise HTTPException(status_code=503, detail="Ultimate scorer not available")
        
        from app.services.scoring.ultimate_write_essay_scorer import get_ultimate_essay_scorer
        
        scorer = get_ultimate_essay_scorer()
        
        if not scorer.use_gpt:
            raise HTTPException(status_code=503, detail="GPT validation unavailable - check OPENAI_API_KEY")
        
        # Run basic ML analysis first
        ml_results = {
            'spelling_score': 2.0,
            'grammar_score': 2.0,
            'vocabulary_score': 2.0,
            'content_score': 6.0,
            'development_score': 6.0,
            'linguistic_score': 6.0,
            'form_score': 2.0,
            'all_errors': []
        }
        
        # Run ultimate GPT verification
        gpt_result = scorer.ultimate_gpt_final_verification(essay_prompt, user_essay, ml_results)
        
        return {
            "success": True,
            "gpt_verification": gpt_result,
            "api_cost": scorer.total_api_cost
        }
        
    except Exception as e:
        logger.error(f"GPT validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))