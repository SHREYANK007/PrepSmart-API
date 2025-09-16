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
- Content (2 points): How well the user understands and summarizes the MAIN MESSAGE of the passage (ignore distracting intro/filler content)
- Form (1 point): Single sentence between 5-75 words
- Grammar (2 points): Grammatical correctness and sentence structure - BE ULTRA-HARSH ON EVERY MICROSCOPIC ERROR
- Vocabulary (2 points): Word choice, spelling, and language precision - BE ULTRA-HARSH ON EVERY TINY MISTAKE

🚨 ULTRA-STRICT GRAMMAR RULES - DEDUCT FOR EVERY TINY ERROR:

⚠️ MICROSCOPIC GRAMMAR RULES (DEDUCT 0.2 POINTS EACH TINY ERROR):
1. COMMA ERRORS - CATCH EVERY SINGLE ONE:
   - Missing comma before subordinating conjunctions: as, when, while, although, since, because, if, unless, until, before, after, where, that, which
   - Missing comma in compound sentences before: and, but, or, so, yet, for, nor
   - Missing comma after introductory phrases
   - Missing comma around non-essential clauses

2. APOSTROPHE ERRORS - ZERO TOLERANCE:
   - Missing apostrophes in contractions: wont→won't, cant→can't, dont→don't, weve→we've
   - Missing possessive apostrophes: childs→child's, students→student's
   - Wrong its/it's usage

3. ARTICLE ERRORS - CATCH EVERY MISSING/WRONG ARTICLE:
   - Missing "a", "an", "the" 
   - Wrong a/an usage before vowel sounds
   - Incorrect definite/indefinite article choice

4. SUBJECT-VERB AGREEMENT - EVERY DISAGREEMENT:
   - Singular/plural mismatches
   - "data shows"→"data show", "research indicate"→"research indicates"
   - Third person singular -s endings

5. PREPOSITION ERRORS - ALL WRONG PREPOSITIONS:
   - "different than"→"different from"
   - "impact on"/"impact of" confusion
   - Wrong preposition with specific verbs/nouns

6. CAPITALIZATION ERRORS:
   - Missing capital letters at sentence start
   - Wrong capitalization of proper nouns

7. SPELLING ERRORS - EVERY MISSPELLING:
   - Even 1-letter mistakes count
   - Typos, homophones, common misspellings

8. WORD ORDER/SYNTAX ERRORS:
   - Misplaced modifiers
   - Awkward word arrangements

🚨 ULTRA-STRICT VOCABULARY RULES - DEDUCT FOR EVERY ERROR:

1. SPELLING ERRORS (DEDUCT 0.4 POINTS EACH):
   - "recieve" → ERROR! Should be "receive"
   - "seperate" → ERROR! Should be "separate"  
   - "occured" → ERROR! Should be "occurred"
   - "beleive" → ERROR! Should be "believe"
   - "definately" → ERROR! Should be "definitely"
   - EVERY misspelling must be caught and reported

2. WORD CHOICE ERRORS (DEDUCT 0.3 POINTS EACH):
   - "effect" vs "affect" misuse
   - "then" vs "than" confusion
   - "there", "their", "they're" errors
   - "its" vs "it's" confusion
   - "your" vs "you're" errors

3. REDUNDANCY/REPETITION (DEDUCT 0.2 POINTS EACH):
   - Repeating same words unnecessarily
   - "various different" → ERROR! Just use "various" or "different"
   - "each and every" → ERROR! Just use "each" or "every"

4. INAPPROPRIATE REGISTER (DEDUCT 0.3 POINTS EACH):
   - Informal words in academic writing
   - "kids" → Should be "children"
   - "stuff" → Should be "material" or "content"
   - Contractions in formal writing

5. COLLOCATION ERRORS (DEDUCT 0.3 POINTS EACH):
   - "make a research" → ERROR! Should be "conduct research"
   - "do a mistake" → ERROR! Should be "make a mistake"
   - Wrong verb-noun combinations

🔍 SPECIFIC TEST CASE:
"The marshmallow test shows that a child's ability to delay gratification, once thought to be innate, is strongly influenced by environment as children who experienced reliable promises resisted longer than those previously deceived."

ERRORS TO CATCH:
- "environment as children" → MISSING COMMA → "environment, as children" (DEDUCT 0.3)
- Check for ANY spelling mistakes
- Check for word choice errors
- Check for article usage

SCORE LIKE APEUNI - BE ULTRA-HARSH ON EVERY TINY MISTAKE!

QUESTION DETAILS:
Title: {question_title}

READING PASSAGE:
{reading_passage}

🧠 CONTENT ANALYSIS INSTRUCTIONS:
1. Read the ENTIRE passage and identify the CORE MESSAGE/MAIN IDEA
2. Ignore introductory fluff, background info, or distracting details
3. Focus on what the passage is REALLY trying to communicate
4. Judge if the user captured this main essence, not just random details

CRITICAL: Do NOT rely on predefined key points - use YOUR intelligence to determine what's important!

USER'S SUMMARY:
{user_summary}

Word Count: {word_count} words

EVALUATION TASK - BE ULTRA-STRICT LIKE APEUNI:
🔍 STEP-BY-STEP ANALYSIS REQUIRED:

1. GRAMMAR ANALYSIS (Scan word by word):
   a) Check EVERY comma placement - look for subordinating conjunctions (as, when, while, that, where, etc.)
   b) Check EVERY apostrophe - contractions and possessives
   c) Check EVERY article usage (a, an, the)
   d) Check subject-verb agreement in EVERY sentence
   e) Check preposition usage
   f) Check tense consistency

2. VOCABULARY ANALYSIS (Check every word):
   a) Scan for ANY spelling mistakes
   b) Check word choice accuracy (effect/affect, then/than, etc.)
   c) Look for redundancy and repetition
   d) Check formality level
   e) Verify collocation correctness

3. PROVIDE ULTRA-SPECIFIC FEEDBACK:
   - Report EXACT word position where error occurs
   - Show EXACT correction needed
   - Explain WHY it's wrong
   - Give specific rule that was violated

🚨 MANDATORY CHECKS - DO THESE EXACT STEPS:

STEP 1: CHECK EVERY SINGLE COMMA POSITION (like APEUni):

A. AFTER INTRODUCTORY PHRASES:
- "on a plate" → needs comma after if followed by new clause
- "also" → needs comma after if it starts a new thought
- "thus" → needs comma after
- "however" → needs comma after
- "therefore" → needs comma after

B. BEFORE SUBORDINATING CONJUNCTIONS:
- "environment as children" → MISSING COMMA → "environment, as children"
- "time when people" → MISSING COMMA → "time, when people"
- "situation since researchers" → MISSING COMMA → "situation, since researchers"

C. BEFORE COORDINATING CONJUNCTIONS (and, but, or, so):
- "plate also now" → if "also" joins two clauses, needs comma before
- "innate and it's also" → if joining two sentences, needs comma before "and"

D. AROUND NON-ESSENTIAL CLAUSES:
- "which shows that..." → needs comma before "which"
- "who experienced..." → check if restrictive or non-restrictive

STEP 2: BE AS STRICT AS APEUNI:
- Find EXACT missing comma positions  
- Report EXACT words where comma is missing
- Example: "Missing comma after 'also'" (just like APEUni does)
- Give specific grammar rule explanations

FOR GRAMMAR SCORING (Ultra-strict - deduct 0.2 per error):
- 2.0: PERFECT - Zero errors
- 1.8: 1 tiny error (missing comma, apostrophe)
- 1.6: 2 errors 
- 1.4: 3 errors
- 1.2: 4 errors
- 1.0: 5 errors
- 0.8: 6 errors
- 0.6: 7 errors
- 0.4: 8 errors
- 0.2: 9 errors
- 0.0: 10+ errors

FOR VOCABULARY SCORING (Ultra-strict - deduct 0.2 per error):
- 2.0: PERFECT - Zero spelling/word choice errors
- 1.8: 1 tiny error (spelling, word choice)
- 1.6: 2 errors
- 1.4: 3 errors
- 1.2: 4 errors
- 1.0: 5 errors
- 0.8: 6 errors
- 0.6: 7 errors
- 0.4: 8 errors
- 0.2: 9 errors
- 0.0: 10+ errors

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
        "grammar": "ULTRA-DETAILED grammar feedback with EXACT word positions, specific corrections needed, and rules violated",
        "vocabulary": "ULTRA-DETAILED vocabulary feedback with exact spelling errors, word choice mistakes, and specific corrections"
    }},
    "grammar_errors": [
        "Word position X: Missing comma before 'as children' - should be 'environment, as children' (Rule: comma before subordinating conjunction)",
        "Word position Y: Missing apostrophe in 'childs' - should be 'child's' (Rule: possessive apostrophe)"
    ],
    "vocabulary_errors": [
        "Word position X: Spelling error 'recieve' - should be 'receive' (Rule: i before e except after c)",
        "Word position Y: Wrong word choice 'effect' - should be 'affect' (Rule: affect is verb, effect is noun)"
    ],
    "detailed_analysis": {{
        "total_grammar_errors": 2,
        "total_vocabulary_errors": 1,
        "error_breakdown": "Comma errors: 1, Spelling errors: 1, Word choice: 0"
    }},
    "overall_feedback": "COMPREHENSIVE assessment with improvement suggestions",
    "strengths": ["List specific strengths found"],
    "improvements": ["List EXACT areas for improvement with specific examples"]
}}

🔍 FINAL QUALITY CHECK - BEFORE RESPONDING:
1. Did I search for "environment as children"? 
2. If found, did I deduct points and report it as error?
3. Did I check EVERY "as", "when", "while", "since", "because" for missing commas?
4. Am I being strict enough (like APEUni would be)?

EXAMPLE 1 - MISSING COMMA AFTER "ALSO":
{{
    "scores": {{"grammar": 1.8, "vocabulary": 2.0}},
    "feedback": {{"grammar": "The summary has a minor punctuation error with a missing comma, but it does not hinder communication. ERRORS: Missing comma after 'also'"}},
    "grammar_errors": ["Missing comma after 'also'"],
    "vocabulary_errors": [],
    "detailed_analysis": {{"total_grammar_errors": 1, "total_vocabulary_errors": 0, "error_breakdown": "Comma errors: 1, Spelling errors: 0"}}
}}

EXAMPLE 2 - MISSING COMMA BEFORE "AS":
{{
    "scores": {{"grammar": 1.8, "vocabulary": 2.0}},
    "feedback": {{"grammar": "Missing comma before subordinating conjunction. ERRORS: Missing comma before 'as children' - should be 'environment, as children'"}},
    "grammar_errors": ["Missing comma before 'as children'"],
    "vocabulary_errors": [],
    "detailed_analysis": {{"total_grammar_errors": 1, "total_vocabulary_errors": 0, "error_breakdown": "Comma errors: 1, Spelling errors: 0"}}
}}

⚠️ CRITICAL: Report comma errors EXACTLY like APEUni:
- "Missing comma after 'word'" 
- "Missing comma before 'word'"
- Be specific about which word needs the comma
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
            grammar_errors=scoring_result.get("grammar_errors", []),  # Specific grammar errors with positions
            vocabulary_errors=scoring_result.get("vocabulary_errors", []),  # Specific vocabulary errors
            detailed_analysis=scoring_result.get("detailed_analysis", {}),  # Error breakdown
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