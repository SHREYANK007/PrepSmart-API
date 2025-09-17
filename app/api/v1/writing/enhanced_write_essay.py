#!/usr/bin/env python3
"""
Enhanced Write Essay API Endpoint
Integrates with the enhanced scoring system while maintaining compatibility
"""

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import time

# Import the enhanced scorer
try:
    from app.services.scoring.enhanced_write_essay_scorer import score_enhanced_write_essay
    ENHANCED_SCORER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced scorer unavailable: {e}")
    ENHANCED_SCORER_AVAILABLE = False
    # Fallback to original scorer
    from app.services.scoring.write_essay_scorer import score_write_essay

logger = logging.getLogger(__name__)

router = APIRouter()

class EnhancedEssayResponse(BaseModel):
    """Enhanced response model with all analysis results"""
    success: bool
    scores: Dict[str, float]
    total_score: float
    mapped_score: Optional[float] = None
    percentage: int
    band: str
    word_count: int
    paragraph_count: int
    
    # Enhanced analysis
    syntactic_complexity: Optional[Dict[str, float]] = None
    vocabulary_analysis: Optional[Dict[str, Any]] = None
    spelling_analysis: Optional[Dict[str, Any]] = None
    structure_analysis: Optional[Dict[str, Any]] = None
    
    # Error details
    errors: Dict[str, List[str]]
    
    # Feedback
    feedback: Dict[str, str]
    detailed_feedback: Optional[Dict[str, Any]] = None
    overall_feedback: str
    strengths: List[str] = []
    improvements: List[str] = []
    ai_recommendations: List[str] = []
    
    # Metadata
    verification_notes: Optional[str] = None
    processing_time: Optional[float] = None
    api_cost: Optional[float] = None
    model_versions: Optional[Dict[str, str]] = None

@router.post("/enhanced", response_model=EnhancedEssayResponse)
async def score_enhanced_essay(
    question_title: str = Form(...),
    essay_prompt: str = Form(...),
    essay_type: str = Form(...),
    key_arguments: str = Form(...),
    sample_essay: str = Form(default=""),
    user_essay: str = Form(...)
):
    """
    Enhanced Write Essay scoring endpoint with comprehensive analysis
    
    Returns detailed scoring with:
    - L2 Syntactic Complexity Analysis
    - Advanced vocabulary and CEFR analysis
    - Paragraph structure analysis with embeddings
    - Enhanced spelling with Hunspell
    - GPT verification with schema validation
    - Non-linear score mapping
    """
    try:
        start_time = time.time()
        
        # Validate input
        if not user_essay or len(user_essay.strip()) < 50:
            raise HTTPException(status_code=400, detail="Essay too short (minimum 50 characters)")
        
        word_count = len(user_essay.split())
        if word_count < 150:
            raise HTTPException(status_code=400, detail=f"Essay too short: {word_count} words (minimum 150)")
        
        # Use enhanced scorer if available
        if ENHANCED_SCORER_AVAILABLE:
            logger.info("🚀 Using Enhanced Write Essay Scorer")
            result = score_enhanced_write_essay(user_essay, essay_prompt)
        else:
            logger.info("⚠️ Using Fallback Scorer")
            result = score_write_essay(user_essay, essay_prompt)
            # Transform to enhanced format
            result = _transform_to_enhanced_format(result)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Scoring failed"))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        # Generate feedback and recommendations
        enhanced_result = _generate_enhanced_feedback(result, essay_prompt, key_arguments)
        
        logger.info(f"✅ Enhanced scoring completed in {processing_time:.2f}s")
        
        return JSONResponse(content=enhanced_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced essay scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal scoring error: {str(e)}")

def _transform_to_enhanced_format(result: Dict) -> Dict:
    """Transform original scorer result to enhanced format"""
    if not result.get("success"):
        return result
    
    # Basic transformation for compatibility
    enhanced = {
        "success": True,
        "scores": result.get("scores", {}),
        "total_score": result.get("total_score", 0),
        "mapped_score": result.get("total_score", 0) * 3.46,  # Basic mapping
        "percentage": result.get("percentage", 0),
        "band": result.get("band", "Limited"),
        "word_count": result.get("word_count", 0),
        "paragraph_count": result.get("paragraph_count", 0),
        
        # Basic analysis (fallback values)
        "syntactic_complexity": {
            "mean_sentence_length": 15.0,
            "subordination_ratio": 0.2,
            "complex_structures": 0.3
        },
        "vocabulary_analysis": {
            "cefr_distribution": {"A1": 0.3, "A2": 0.3, "B1": 0.2, "B2": 0.1, "C1": 0.1, "C2": 0.0},
            "lexical_diversity": 0.5,
            "academic_ratio": 0.05
        },
        "spelling_analysis": {
            "total_errors": len(result.get("errors", {}).get("spelling", [])),
            "error_types": {"common": 0, "academic": 0, "other": 0},
            "severity": 0.1
        },
        "structure_analysis": {
            "paragraph_similarities": [],
            "coherence_score": 4.0
        },
        
        "errors": result.get("errors", {}),
        "verification_notes": "Using fallback scorer",
        "processing_time": 0.0,
        "api_cost": 0.0,
        "model_versions": {
            "scorer": "fallback",
            "embeddings": "unavailable",
            "gpt": "unavailable"
        }
    }
    
    return enhanced

def _generate_enhanced_feedback(result: Dict, essay_prompt: str, key_arguments: str) -> Dict:
    """Generate comprehensive feedback and recommendations"""
    
    scores = result.get("scores", {})
    total_score = result.get("total_score", 0)
    
    # Generate component feedback
    feedback = {}
    detailed_feedback = {}
    strengths = []
    improvements = []
    ai_recommendations = []
    
    # Content feedback
    content_score = scores.get("content", 0)
    if content_score >= 5:
        feedback["content"] = "Excellent content coverage and argument development"
        strengths.append("Strong content addressing the prompt comprehensively")
    elif content_score >= 3:
        feedback["content"] = "Good content but could be more detailed"
        improvements.append("Develop arguments with more specific examples and evidence")
    else:
        feedback["content"] = "Content needs significant improvement"
        improvements.append("Focus on directly addressing the essay prompt")
        ai_recommendations.append("Re-read the prompt carefully and ensure each paragraph addresses a key aspect")
    
    # Grammar feedback
    grammar_score = scores.get("grammar", 0)
    if grammar_score >= 1.5:
        feedback["grammar"] = "Good grammatical control with minor errors"
        strengths.append("Generally accurate grammar usage")
    elif grammar_score >= 1.0:
        feedback["grammar"] = "Some grammatical errors that need attention"
        improvements.append("Review complex sentence structures and verb tenses")
    else:
        feedback["grammar"] = "Significant grammatical errors affecting communication"
        improvements.append("Focus on basic sentence structure and common grammar rules")
        ai_recommendations.append("Practice writing shorter, clearer sentences before attempting complex structures")
    
    # Vocabulary feedback
    vocab_score = scores.get("vocabulary", 0)
    vocab_analysis = result.get("vocabulary_analysis", {})
    academic_ratio = vocab_analysis.get("academic_ratio", 0)
    
    if vocab_score >= 1.5:
        feedback["vocabulary"] = "Good vocabulary range and appropriate word choice"
        strengths.append("Effective use of academic and sophisticated vocabulary")
    elif vocab_score >= 1.0:
        feedback["vocabulary"] = "Adequate vocabulary but could be more varied"
        improvements.append("Incorporate more academic vocabulary and avoid repetition")
        ai_recommendations.append("Learn 5-10 new academic words per week and practice using them in context")
    else:
        feedback["vocabulary"] = "Limited vocabulary range needs expansion"
        improvements.append("Focus on building academic vocabulary")
        ai_recommendations.append("Use a variety of synonyms and avoid overusing basic words like 'good', 'bad', 'big'")
    
    # Spelling feedback
    spelling_score = scores.get("spelling", 0)
    spelling_analysis = result.get("spelling_analysis", {})
    total_errors = spelling_analysis.get("total_errors", 0)
    
    if spelling_score >= 1.5:
        feedback["spelling"] = "Excellent spelling accuracy"
        strengths.append("Very good spelling control")
    elif spelling_score >= 1.0:
        feedback["spelling"] = f"Good spelling with {total_errors} minor error(s)"
        if total_errors > 0:
            improvements.append("Review spelling of academic and complex words")
    else:
        feedback["spelling"] = f"Spelling needs improvement - {total_errors} error(s) detected"
        improvements.append("Use spell-check and practice spelling of common academic words")
        ai_recommendations.append("Keep a personal spelling list of words you frequently misspell")
    
    # Linguistic Range feedback
    linguistic_score = scores.get("linguistic", 0)
    complexity = result.get("syntactic_complexity", {})
    subordination_ratio = complexity.get("subordination_ratio", 0)
    
    if linguistic_score >= 4:
        feedback["linguistic"] = "Excellent variety in sentence structures and complexity"
        strengths.append("Sophisticated use of complex sentence structures")
    elif linguistic_score >= 3:
        feedback["linguistic"] = "Good sentence variety with some complex structures"
        improvements.append("Try to include more subordinate clauses and complex sentences")
    else:
        feedback["linguistic"] = "Limited sentence variety - needs more complex structures"
        improvements.append("Practice using a variety of sentence types (simple, compound, complex)")
        ai_recommendations.append("Learn to combine ideas using conjunctions like 'although', 'whereas', 'because'")
    
    # Development/Structure feedback
    development_score = scores.get("development", 0)
    structure_analysis = result.get("structure_analysis", {})
    coherence_score = structure_analysis.get("coherence_score", 0)
    
    if development_score >= 4:
        feedback["development"] = "Well-developed ideas with clear organization"
        strengths.append("Clear paragraph structure and logical flow")
    elif development_score >= 3:
        feedback["development"] = "Good development but could improve transitions"
        improvements.append("Use more linking words to connect ideas between paragraphs")
    else:
        feedback["development"] = "Ideas need better development and organization"
        improvements.append("Ensure each paragraph has a clear main idea with supporting details")
        ai_recommendations.append("Follow a clear structure: Introduction → Body Paragraphs → Conclusion")
    
    # Form feedback
    form_score = scores.get("form", 0)
    word_count = result.get("word_count", 0)
    
    if form_score >= 1.5:
        feedback["form"] = f"Good essay format and appropriate length ({word_count} words)"
        strengths.append("Meets formal requirements for essay writing")
    elif form_score >= 1.0:
        feedback["form"] = f"Meets basic requirements ({word_count} words)"
    else:
        if word_count < 200:
            feedback["form"] = f"Essay too short ({word_count} words) - aim for 200-300 words"
            improvements.append("Develop ideas more fully to reach the required word count")
        elif word_count > 300:
            feedback["form"] = f"Essay too long ({word_count} words) - stay within 200-300 words"
            improvements.append("Edit to be more concise while maintaining key ideas")
        else:
            feedback["form"] = "Review essay structure and formatting"
    
    # Generate overall feedback
    if total_score >= 20:
        overall_feedback = f"Excellent essay! Your score of {total_score}/26 ({result.get('percentage', 0)}%) shows strong writing skills across all areas."
    elif total_score >= 15:
        overall_feedback = f"Good essay with a score of {total_score}/26 ({result.get('percentage', 0)}%). Focus on the improvement areas to achieve excellence."
    elif total_score >= 10:
        overall_feedback = f"Your essay scored {total_score}/26 ({result.get('percentage', 0)}%). There's good potential - work on the key areas identified."
    else:
        overall_feedback = f"Your essay scored {total_score}/26 ({result.get('percentage', 0)}%). Focus on fundamental writing skills and practice regularly."
    
    # Add AI recommendations based on score analysis
    if not ai_recommendations:
        ai_recommendations.append("Practice writing essays regularly with timed conditions")
        ai_recommendations.append("Read academic texts to improve vocabulary and sentence structures")
        ai_recommendations.append("Plan your essay structure before writing: Introduction → Arguments → Conclusion")
    
    # Ensure we have at least 3-5 recommendations
    while len(ai_recommendations) < 5:
        additional_recommendations = [
            "Use a variety of linking words (however, furthermore, consequently, etc.)",
            "Support your arguments with specific examples or evidence",
            "Review your essay for grammar and spelling before submitting",
            "Practice paraphrasing ideas instead of repeating the same words",
            "Time yourself when practicing - aim to complete essays in 20 minutes"
        ]
        for rec in additional_recommendations:
            if rec not in ai_recommendations and len(ai_recommendations) < 5:
                ai_recommendations.append(rec)
    
    # Build enhanced result
    enhanced_result = result.copy()
    enhanced_result.update({
        "feedback": feedback,
        "detailed_feedback": detailed_feedback,
        "overall_feedback": overall_feedback,
        "strengths": strengths[:3],  # Limit to top 3
        "improvements": improvements[:3],  # Limit to top 3
        "ai_recommendations": ai_recommendations[:5]  # Limit to top 5
    })
    
    return enhanced_result

# Legacy endpoint for backward compatibility
@router.post("/legacy", response_model=EnhancedEssayResponse)
async def score_essay_legacy(
    question_title: str = Form(...),
    essay_prompt: str = Form(...),
    essay_type: str = Form(...),
    key_arguments: str = Form(...),
    sample_essay: str = Form(default=""),
    user_essay: str = Form(...)
):
    """Legacy endpoint that uses original scorer but returns enhanced format"""
    
    try:
        # Use original scorer
        if ENHANCED_SCORER_AVAILABLE:
            from app.services.scoring.write_essay_scorer import score_write_essay as original_scorer
        else:
            original_scorer = score_write_essay
        
        result = original_scorer(user_essay, essay_prompt)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Scoring failed"))
        
        # Transform to enhanced format
        enhanced_result = _transform_to_enhanced_format(result)
        enhanced_result = _generate_enhanced_feedback(enhanced_result, essay_prompt, key_arguments)
        
        return JSONResponse(content=enhanced_result)
        
    except Exception as e:
        logger.error(f"Legacy essay scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

# Calibration endpoint for tuning
@router.post("/calibrate")
async def calibrate_scoring(calibration_data: List[Dict]):
    """
    Calibration endpoint for tuning score mapping against real data
    
    Expected format:
    [
        {
            "raw_scores": {"content": 4, "grammar": 1.5, ...},
            "expected_score": 65
        },
        ...
    ]
    """
    try:
        if not ENHANCED_SCORER_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced scorer not available for calibration")
        
        from app.services.scoring.enhanced_write_essay_scorer import get_enhanced_essay_scorer
        
        scorer = get_enhanced_essay_scorer()
        updated_params = scorer.calibrate_score_mapping(calibration_data)
        
        return {
            "success": True,
            "message": "Score mapping calibrated successfully",
            "updated_parameters": updated_params,
            "calibration_samples": len(calibration_data)
        }
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")

@router.get("/status")
async def get_scoring_status():
    """Get the status of the enhanced scoring system"""
    
    status = {
        "enhanced_scorer_available": ENHANCED_SCORER_AVAILABLE,
        "timestamp": time.time()
    }
    
    if ENHANCED_SCORER_AVAILABLE:
        try:
            from app.services.scoring.enhanced_write_essay_scorer import get_enhanced_essay_scorer
            scorer = get_enhanced_essay_scorer()
            
            status.update({
                "gector_available": scorer.gector_model is not None,
                "language_tool_available": scorer.language_tool is not None,
                "sentence_transformer_available": scorer.sentence_model is not None,
                "hunspell_available": scorer.hunspell_checker is not None,
                "gpt_available": scorer.use_gpt,
                "mapping_parameters": scorer.mapping_params
            })
        except Exception as e:
            status["error"] = str(e)
    
    return status