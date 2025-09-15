"""
Summarize Written Text API Endpoint
Scoring: Content (2), Form (1), Grammar (2), Vocabulary (2) = Total 7 points
"""

from fastapi import APIRouter, HTTPException, Form, Depends
from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime

from app.models.writing_models import SummarizeTextRequest, SummarizeTextResponse
from app.services.scoring.gpt_service import GPTService
from app.services.wordpress.wp_client import WordPressClient
from app.core.dependencies import get_current_user

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/writing",
    tags=["Writing - Summarize Written Text"]
)

# Initialize services
gpt_service = GPTService()
wp_client = WordPressClient()


@router.post("/summarize-written-text", response_model=SummarizeTextResponse)
async def score_summarize_written_text(
    # Question context from WordPress
    question_id: int = Form(..., description="Question ID from WordPress"),
    question_title: str = Form(..., description="Title of the question"),
    reading_passage: str = Form(..., description="The passage to be summarized"),
    key_points: str = Form(..., description="Key points that should be covered"),
    sample_summary: Optional[str] = Form(None, description="Sample summary for reference"),
    
    # User input
    user_summary: str = Form(..., description="User's written summary"),
    user_id: Optional[int] = Form(None, description="WordPress user ID"),
    
    # Optional metadata
    time_taken: Optional[int] = Form(None, description="Time taken in seconds"),
    word_count: Optional[int] = Form(None, description="Word count of user summary")
):
    """
    Score Summarize Written Text task using GPT-4.
    
    Scoring Criteria:
    - Content (2 points): Key points coverage, accuracy
    - Form (1 point): Single sentence, 5-75 words
    - Grammar (2 points): Grammar and sentence structure
    - Vocabulary (2 points): Word choice and variety
    
    Total: 7 points
    """
    
    try:
        logger.info(f"Processing SWT for question_id: {question_id}, user_id: {user_id}")
        
        # Step 1: Validate input
        if not user_summary.strip():
            raise HTTPException(status_code=400, detail="User summary cannot be empty")
        
        # Calculate word count if not provided
        if word_count is None:
            word_count = len(user_summary.split())
        
        # Step 2: Create GPT prompt
        gpt_prompt = f"""
Act as a proficient Pearson PTE Academic AI scoring bot. You are tasked with scoring a "Summarize Written Text" response.

SCORING CRITERIA (Total: 7 points):
- Content (2 points): How well the key points from the passage are covered
- Form (1 point): Single sentence between 5-75 words
- Grammar (2 points): Grammatical correctness and sentence structure  
- Vocabulary (2 points): Appropriate word choice and range

QUESTION DETAILS:
Title: {question_title}

READING PASSAGE:
{reading_passage}

KEY POINTS TO COVER:
{key_points}

{f"SAMPLE SUMMARY (for reference): {sample_summary}" if sample_summary else ""}

USER'S SUMMARY:
{user_summary}

Word Count: {word_count} words

EVALUATION TASK:
Please evaluate the user's summary and provide:
1. Individual scores for each criterion
2. Specific feedback for each criterion
3. Overall feedback and suggestions for improvement

Return your response in the following JSON format:
{{
    "scores": {{
        "content": 0-2,
        "form": 0-1,
        "grammar": 0-2,
        "vocabulary": 0-2
    }},
    "total_score": 0-7,
    "feedback": {{
        "content": "Specific feedback on content coverage",
        "form": "Feedback on form requirements",
        "grammar": "Grammar feedback with examples if needed",
        "vocabulary": "Vocabulary usage feedback"
    }},
    "overall_feedback": "General assessment and improvement suggestions",
    "strengths": ["List of strengths"],
    "improvements": ["List of areas for improvement"]
}}
"""

        # Step 3: Call GPT-4 API
        logger.info("Calling GPT-4 for scoring...")
        gpt_response = await gpt_service.get_scoring(
            prompt=gpt_prompt,
            max_tokens=1000,
            temperature=0.3  # Lower temperature for consistent scoring
        )
        
        # Step 4: Parse GPT response
        try:
            scoring_result = json.loads(gpt_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse GPT response: {gpt_response}")
            # Fallback parsing or error handling
            scoring_result = parse_gpt_response_fallback(gpt_response)
        
        # Step 5: Calculate percentage and grade
        total_score = scoring_result.get("total_score", 
                        sum(scoring_result["scores"].values()))
        percentage = round((total_score / 7) * 100)
        
        # Determine grade
        if percentage >= 90:
            grade = "Excellent"
        elif percentage >= 80:
            grade = "Very Good"
        elif percentage >= 70:
            grade = "Good"
        elif percentage >= 60:
            grade = "Fair"
        else:
            grade = "Needs Improvement"
        
        # Step 6: Save attempt to WordPress database (optional)
        if user_id:
            try:
                await wp_client.save_attempt(
                    user_id=user_id,
                    question_id=question_id,
                    task_type="summarize_written_text",
                    user_response=user_summary,
                    scores=scoring_result["scores"],
                    total_score=total_score,
                    feedback=scoring_result["feedback"]
                )
            except Exception as e:
                logger.error(f"Failed to save to WordPress: {e}")
                # Don't fail the request if WordPress save fails
        
        # Step 7: Prepare response
        response = SummarizeTextResponse(
            success=True,
            scores=scoring_result["scores"],
            total_score=total_score,
            percentage=percentage,
            grade=grade,
            feedback=scoring_result["feedback"],
            overall_feedback=scoring_result.get("overall_feedback", ""),
            strengths=scoring_result.get("strengths", []),
            improvements=scoring_result.get("improvements", []),
            word_count=word_count,
            processing_time=None,  # Will be calculated by middleware
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Successfully scored SWT: total_score={total_score}, grade={grade}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing SWT: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing summary: {str(e)}"
        )


def parse_gpt_response_fallback(response_text: str) -> Dict[str, Any]:
    """
    Fallback parser if JSON parsing fails.
    Attempts to extract scores from text response.
    """
    try:
        # Default structure
        result = {
            "scores": {
                "content": 1,
                "form": 0,
                "grammar": 1,
                "vocabulary": 1
            },
            "total_score": 3,
            "feedback": {
                "content": "Unable to parse detailed feedback",
                "form": "Unable to parse detailed feedback",
                "grammar": "Unable to parse detailed feedback",
                "vocabulary": "Unable to parse detailed feedback"
            },
            "overall_feedback": "Response was evaluated but detailed parsing failed",
            "strengths": [],
            "improvements": []
        }
        
        # Try to extract scores from text
        import re
        
        # Look for content score
        content_match = re.search(r'content[:\s]+(\d)', response_text, re.IGNORECASE)
        if content_match:
            result["scores"]["content"] = min(int(content_match.group(1)), 2)
        
        # Look for form score
        form_match = re.search(r'form[:\s]+(\d)', response_text, re.IGNORECASE)
        if form_match:
            result["scores"]["form"] = min(int(form_match.group(1)), 1)
        
        # Look for grammar score
        grammar_match = re.search(r'grammar[:\s]+(\d)', response_text, re.IGNORECASE)
        if grammar_match:
            result["scores"]["grammar"] = min(int(grammar_match.group(1)), 2)
        
        # Look for vocabulary score
        vocab_match = re.search(r'vocabulary[:\s]+(\d)', response_text, re.IGNORECASE)
        if vocab_match:
            result["scores"]["vocabulary"] = min(int(vocab_match.group(1)), 2)
        
        # Calculate total
        result["total_score"] = sum(result["scores"].values())
        
        return result
        
    except Exception as e:
        logger.error(f"Fallback parser also failed: {e}")
        # Return minimum scores as last resort
        return {
            "scores": {"content": 1, "form": 0, "grammar": 1, "vocabulary": 1},
            "total_score": 3,
            "feedback": {
                "content": "Scoring service temporarily unavailable",
                "form": "Scoring service temporarily unavailable",
                "grammar": "Scoring service temporarily unavailable",
                "vocabulary": "Scoring service temporarily unavailable"
            },
            "overall_feedback": "Automated scoring failed. Please try again.",
            "strengths": [],
            "improvements": []
        }


@router.post("/summarize-written-text/validate")
async def validate_summary(
    user_summary: str = Form(...),
    check_word_count: bool = Form(True),
    check_sentence_count: bool = Form(True)
) -> Dict[str, Any]:
    """
    Quick validation endpoint for user summary before submission.
    Checks form requirements without GPT scoring.
    """
    
    words = user_summary.split()
    word_count = len(words)
    
    # Count sentences (basic check)
    sentence_count = len([s for s in user_summary.split('.') if s.strip()])
    
    validation = {
        "valid": True,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "errors": [],
        "warnings": []
    }
    
    # Check word count (5-75 words required)
    if check_word_count:
        if word_count < 5:
            validation["valid"] = False
            validation["errors"].append(f"Too short: {word_count} words (minimum 5)")
        elif word_count > 75:
            validation["valid"] = False
            validation["errors"].append(f"Too long: {word_count} words (maximum 75)")
    
    # Check sentence count (must be single sentence)
    if check_sentence_count:
        if sentence_count > 1:
            validation["warnings"].append(f"Multiple sentences detected ({sentence_count}). Should be single sentence.")
    
    # Check for empty or whitespace only
    if not user_summary.strip():
        validation["valid"] = False
        validation["errors"].append("Summary cannot be empty")
    
    return validation