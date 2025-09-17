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
# Essay models and functions (inline for now)
class EssayFeedback(BaseModel):
    score: float
    justification: str
    errors: List[str]
    suggestions: List[str]

class WriteEssayResponse(BaseModel):
    success: bool
    scores: Dict[str, float]
    feedback: Dict[str, str]
    detailed_feedback: Dict[str, EssayFeedback]
    overall_feedback: str
    total_score: float
    percentage: int
    band: str
    word_count: int
    paragraph_count: int
    key_arguments_covered: List[str]
    key_arguments_missed: List[str]
    strengths: List[str]
    improvements: List[str]
    ai_recommendations: List[str]

async def analyze_essay_with_gpt4(
    essay_prompt: str,
    essay_type: str,
    key_arguments: str,
    sample_essay: str,
    user_essay: str,
    question_title: str
) -> Dict:
    """Use GPT-4 to analyze and score the essay according to exact PTE criteria"""
    
    prompt = f"""You are the OFFICIAL PEARSON PTE ACADEMIC AUTOMATED SCORING ENGINE. Score this Write Essay response using EXACT official Pearson scoring guide.

ESSAY PROMPT:
{essay_prompt}

ESSAY TYPE:
{essay_type}

KEY ARGUMENTS/POINTS TO COVER:
{key_arguments}

SAMPLE ESSAY REFERENCE:
{sample_essay}

STUDENT'S ESSAY:
{user_essay}

OFFICIAL PEARSON PTE ACADEMIC ESSAY SCORING GUIDE (Total: 26 points):

1. CONTENT (0-6 points) - PEARSON EXACT CRITERIA:
   • 6: The essay fully addresses the prompt in depth, demonstrating full command of the argument by reformulating the issue seamlessly in own words and expanding on important points with specificity. The argument is supported convincingly with subsidiary points and relevant examples throughout the response.
   • 5: The essay adequately addresses the prompt, presenting a persuasive argument with relevant ideas. Main points are highlighted, and relevant supporting detail is given to support the response effectively, with minor exceptions.
   • 4: The essay adequately addresses the main point of the prompt. The argument is generally convincing, though lacks depth or nuance. Supporting detail is inconsistent throughout the response. It is present for some points but weaker or missing for others.
   • 3: The essay is relevant to the prompt but does not address the main points adequately. Supporting detail is often missing or inappropriate.
   • 2: The essay attempts to address the prompt, but does so superficially, with little relevant information and largely generic statements or over reliance on repeating language from the prompt. Few supporting details are included. Ideas that are present lack relevance, with only tangential links to the topic.
   • 1: The essay attempts to address the prompt, but demonstrates an incomplete understanding of the prompt with communication limited to generic or repetitive phrasing, or repeating language from the prompt. Supporting details, if present, are used in a disjointed or haphazard manner, with no clear links to the topic.
   • 0: The essay does not properly deal with the prompt.

2. FORM (0-2 points) - PEARSON EXACT CRITERIA:
   • 2: Length is between 200 and 300 words
   • 1: Length is between 120 and 199 or between 301 and 380 words
   • 0: Length is less than 120 or more than 380 words. Essay is written in capital letters, contains no punctuation or only consists of bullet points or very short sentences

3. GRAMMAR (0-2 points) - PEARSON EXACT CRITERIA:
   • 2: Shows consistent grammatical control of complex language. Errors are rare and difficult to spot
   • 1: Shows a relatively high degree of grammatical control. No mistakes which would lead to misunderstandings
   • 0: Contains mainly simple structures and/or several basic mistakes

4. SPELLING (0-2 points) - PEARSON EXACT CRITERIA:
   • 2: Correct spelling
   • 1: One spelling error
   • 0: More than one spelling error

5. VOCABULARY RANGE (0-2 points) - PEARSON EXACT CRITERIA:
   • 2: Good command of a broad lexical repertoire, idiomatic expressions and colloquialisms
   • 1: Shows a good range of vocabulary for matters connected to general academic topics. Lexical shortcomings lead to circumlocution or some imprecision
   • 0: Contains mainly basic vocabulary insufficient to deal with the topic at the required level

6. GENERAL LINGUISTIC RANGE (0-6 points) - PEARSON EXACT CRITERIA:
   • 6: A variety of expressions and vocabulary are used appropriately to formulate ideas with ease and precision throughout the response. No signs of limitations restricting what can be communicated. Errors in language use, if present, are rare and minor, and meaning incompletely clear.
   • 5: A variety of expressions and vocabulary are used appropriately throughout the response. Ideas are expressed clearly without much sign of restriction. Occasional errors in language use are present, but the meaning is clear.
   • 4: The range of expression and vocabulary is sufficient to articulate basic ideas. Most ideas are clear, but limitations are evident when conveying complex / abstract ideas, causing repetition, circumlocution, and difficulty with formulation at times. Errors in language use cause occasional lapses in clarity, but the main idea can still be followed.
   • 3: The range of expression and vocabulary is narrow and simple expressions are used repeatedly. Communication is restricted to simple ideas that can be articulated through basic language. Errors in language use cause some disruptions for the reader.
   • 2: Limited vocabulary and simple expressions dominate the response. Communication is compromised and some ideas are unclear. Basic errors in language use are common, causing frequent breakdowns and misunderstanding.
   • 1: Vocabulary and linguistic expression are highly restricted. There are significant limitations in communication and ideas are generally unclear. Errors in language use are pervasive and impede meaning.
   • 0: Meaning is not accessible.

7. DEVELOPMENT, STRUCTURE AND COHERENCE (0-6 points) - PEARSON EXACT CRITERIA:
   • 6: The essay has an effective logical structure, flows smoothly, and can be followed with ease. An argument is clear and cohesive, developed systematically at length. A well-developed introduction and conclusion are present. Ideas are organised cohesively into paragraphs, and paragraphs are clear and logically sequenced. The essay uses a variety of connective devices effectively and consistently to convey relationships between ideas.
   • 5: The essay has a conventional and appropriate structure that follows logically, if not always smoothly. An argument is clear, with some points developed at length. Introduction, conclusion and logical paragraphs are present. The essay uses connective devices to link utterances into clear, coherent discourse, though there may be some gaps or abrupt transitions between one idea to the next.
   • 4: Conventional structure is mostly present, but some elements may be missing, requiring some effort to follow. An argument is present but lacks development of some elements or may be difficult to follow. Simple paragraph breaks are present, but they are not always effective, and some elements or paragraphs are poorly linked. The ideas in the response are not well connected. The lack of connection might come from an ordering of the ideas which is difficult to grasp, or a lack of language establishing coherence among ideas.
   • 3: Traces of the conventional structure are present, but the essay is composed of simple points or disconnected ideas. A position/opinion is present, although it is not sufficiently developed into a logical argument and often lacks clarity. Essay does not make effective use of paragraphs or lacks paragraphs but presents ideas with some coherence and logical sequencing. The response consists mainly of unconnected ideas, with little organizational structure evident, and requires significant effort to follow. The most frequently occurring connective devices link simple sentences and larger elements linearly, but more complex relationships are not expressed clearly or appropriately.
   • 2: There is little recognisable structure. Ideas are presented in a disorganised manner and are difficult to follow. A position/opinion may be present but lacks development or clarity. The essay lacks coherence, and mainly consists of disconnected elements. Can link groups of words with simple connective devices (e.g., "and", "but" and "because").
   • 1: Response consists of disconnected ideas. There is no hierarchy of ideas or coherence among points. No clear position/opinion can be identified. Words and short statements are linked with very basic linear connective devices(e.g., "and" or "then").
   • 0: There is no recognisable structure.

SCORING INSTRUCTIONS:
- Use INTELLIGENT FRACTIONAL SCORING (e.g., 4.8, 5.2, 1.7) to precisely reflect performance
- Minor issues = small deductions (0.1-0.3 from category maximum)
- Moderate issues = medium deductions (0.5-0.8 from category maximum)  
- Major issues = large deductions (1.0+ from category maximum)
- DO NOT deduct whole points for minor mistakes - use GPT intelligence for fair decimal scoring
- Compare against the key arguments provided to assess content coverage
- Reference the sample essay for quality benchmarking

CRITICAL ANALYSIS REQUIRED:
1. CONTENT: Compare user essay against key arguments/points - what's covered vs missing
2. FORM: Count exact words and check structure
3. GRAMMAR: Identify errors but assess communication impact
4. SPELLING: Count exact spelling errors
5. VOCABULARY: Assess range and appropriateness
6. LINGUISTIC RANGE: Evaluate expression variety and precision
7. COHERENCE: Check logical flow and organization

Return ONLY valid JSON in this exact format (USE DECIMAL SCORES):
{{
    "content_score": 4.8,
    "content_justification": "Addresses prompt well but misses 1-2 key arguments about [specific points]",
    "content_errors": ["Missing discussion of environmental impact", "Weak coverage of economic benefits"],
    "content_suggestions": ["Include analysis of environmental consequences", "Expand on economic advantages"],
    
    "form_score": 2.0,
    "form_justification": "245 words, proper essay structure with clear paragraphs",
    "form_errors": [],
    "form_suggestions": [],
    
    "grammar_score": 1.7,
    "grammar_justification": "Generally good control with minor errors that don't affect understanding",
    "grammar_errors": ["Subject-verb disagreement in paragraph 2", "Tense consistency issue"],
    "grammar_suggestions": ["Review subject-verb agreement", "Maintain consistent tense throughout"],
    
    "spelling_score": 1.0,
    "spelling_justification": "One spelling error found",
    "spelling_errors": ["'recieve' should be 'receive'"],
    "spelling_suggestions": ["Double-check spelling before submission"],
    
    "vocabulary_score": 1.8,
    "vocabulary_justification": "Good range with minor imprecision in word choice",
    "vocabulary_errors": ["Imprecise use of 'utilize' instead of 'use'"],
    "vocabulary_suggestions": ["Use simpler, more precise vocabulary"],
    
    "linguistic_score": 4.3,
    "linguistic_justification": "Good variety but some repetitive patterns limit expression",
    "linguistic_errors": ["Overuse of 'Furthermore' as connector", "Repetitive sentence openings"],
    "linguistic_suggestions": ["Vary transitional phrases", "Use different sentence starters"],
    
    "coherence_score": 5.1,
    "coherence_justification": "Well-structured with clear progression and effective paragraphing",
    "coherence_errors": ["Weak transition between body paragraphs 2-3"],
    "coherence_suggestions": ["Strengthen connections between main ideas"],
    
    "word_count": 245,
    "paragraph_count": 4,
    "key_arguments_covered": ["Economic benefits", "Social impact"],
    "key_arguments_missed": ["Environmental considerations", "Long-term sustainability"],
    "overall_assessment": "Strong essay with good structure and argumentation, minor improvements needed",
    "strengths": ["Clear thesis", "Logical organization", "Relevant examples"],
    "priority_improvements": ["Include missing key arguments", "Improve vocabulary precision", "Fix spelling errors"]
}}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a PTE Academic essay examiner. Evaluate essays fairly using the 26-point scoring system. Be constructive and helpful. Respond with valid JSON only."},
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
        print(f"GPT-4 Essay Analysis Error: {e}")
        # Return basic fallback analysis
        word_count = len(user_essay.split())
        paragraph_count = len([p for p in user_essay.split('\n\n') if p.strip()])
        
        return {
            "content_score": 3.0,
            "content_justification": "Fallback scoring - GPT-4 unavailable",
            "content_errors": [],
            "content_suggestions": ["Connect to GPT-4 for detailed analysis"],
            "linguistic_score": 3.0,
            "linguistic_justification": "Cannot analyze without GPT-4",
            "linguistic_errors": [],
            "linguistic_suggestions": [],
            "coherence_score": 3.0,
            "coherence_justification": "Cannot analyze without GPT-4",
            "coherence_errors": [],
            "coherence_suggestions": [],
            "form_score": 2.0 if 200 <= word_count <= 300 else 0,
            "form_justification": f"Word count: {word_count}",
            "form_errors": [],
            "form_suggestions": [],
            "grammar_score": 1.0,
            "grammar_justification": "Cannot analyze without GPT-4",
            "grammar_errors": [],
            "grammar_suggestions": [],
            "spelling_score": 1.0,
            "spelling_justification": "Cannot analyze without GPT-4",
            "spelling_errors": [],
            "spelling_suggestions": [],
            "vocabulary_score": 1.0,
            "vocabulary_justification": "Cannot analyze without GPT-4",
            "vocabulary_errors": [],
            "vocabulary_suggestions": [],
            "word_count": word_count,
            "paragraph_count": paragraph_count,
            "overall_assessment": "GPT-4 analysis unavailable",
            "strengths": [],
            "priority_improvements": ["Ensure GPT-4 API is available"]
        }

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

# Global scorer instance - initialize once to handle multiple requests
global_scorer = None

def get_scorer():
    """Get global scorer instance, initialize if needed"""
    global global_scorer
    if global_scorer is None:
        try:
            from app.services.scoring.hybrid_scorer_enhanced import get_enhanced_scorer
            global_scorer = get_enhanced_scorer()
            print("✅ Global enhanced scorer initialized with all layers")
        except ImportError:
            from app.services.scoring.hybrid_scorer import hybrid_scorer
            global_scorer = hybrid_scorer
            print("✅ Global fallback scorer initialized")
    return global_scorer

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
    vocabulary_errors: List[str]  # Add specific vocabulary errors
    detailed_analysis: Dict = {}  # Add error breakdown analysis
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
        
        # Use enhanced scorer directly
        from app.services.scoring.hybrid_scorer_enhanced import get_enhanced_scorer
        hybrid_scorer = get_enhanced_scorer()
        
        print(f"DEBUG: Using scorer type: {type(hybrid_scorer).__name__}")
        print(f"DEBUG: About to call hybrid scorer with: '{user_summary}'")
        analysis = hybrid_scorer.comprehensive_score(
            user_summary=user_summary,
            passage=reading_passage,
            key_points=key_points
        )
        print(f"DEBUG: Hybrid scorer result: success={analysis.get('success')}")
        print(f"DEBUG: Grammar errors found: {len(analysis.get('grammar_errors', []))}")
        
        if not analysis.get("success"):
            print("DEBUG: Falling back to GPT scoring")
            # Fallback to GPT if hybrid fails
            analysis = await analyze_with_gpt4(
                reading_passage=reading_passage,
                key_points=key_points,
                sample_summary=sample_summary,
                user_summary=user_summary,
                question_title=question_title
            )
        else:
            print("DEBUG: Using hybrid scorer results")
        
        # Extract scores (hybrid format)
        if analysis.get("success"):
            # Hybrid scorer format
            scores = analysis.get("scores", {})
            content_score = scores.get("content", 0)
            form_score = scores.get("form", 0)
            grammar_score = scores.get("grammar", 0)
            vocabulary_score = scores.get("vocabulary", 0)
        else:
            # Fallback GPT format
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
            overall_feedback=analysis.get("harsh_assessment", analysis.get("overall_assessment", f"Your summary scored {total_score}/7.")),
            total_score=total_score,
            percentage=percentage,
            band=band,
            key_points_covered=analysis.get("key_points_covered", []),
            key_points_missed=analysis.get("key_points_missed", []),
            grammar_errors=analysis.get("grammar_errors", []),
            vocabulary_errors=analysis.get("vocabulary_errors", []),
            detailed_analysis=analysis.get("detailed_analysis", {}),
            vocabulary_assessment=analysis.get("vocabulary_justification", ""),
            improvements=analysis.get("improvements", []),
            strengths=analysis.get("strengths", []),
            ai_recommendations=analysis.get("improvements", []),
            word_count=word_count,
            is_single_sentence=is_single_sentence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Scoring Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Essay endpoint using organized module

@app.post("/api/v1/writing/essay", response_model=WriteEssayResponse)
async def score_write_essay(
    question_title: str = Form(...),
    essay_prompt: str = Form(...),
    essay_type: str = Form(...),
    key_arguments: str = Form(...),
    sample_essay: str = Form(default=""),
    user_essay: str = Form(...)
):
    """
    Score Write Essay using PTE 26-point system with 3-layer hybrid scorer
    """
    try:
        # Validate input
        if not user_essay.strip():
            raise HTTPException(status_code=400, detail="Essay cannot be empty")
        
        # Use Ultimate scorer with ML + GPT independent analysis (NO FALLBACK)
        from app.services.scoring.ultimate_write_essay_scorer import score_ultimate_write_essay
        
        print("ULTIMATE: Starting ML + GPT independent analysis workflow")
        raw_analysis = score_ultimate_write_essay(user_essay, essay_prompt)
        
        # CRITICAL: Clean all Unicode characters from Ultimate scorer response
        def deep_clean_unicode(obj):
            if isinstance(obj, str):
                # Handle ALL Unicode characters that cause Windows encoding issues
                import re
                # Replace common problematic Unicode first
                cleaned = obj.replace('→', ' to ').replace('✗', '').replace('✓', '').replace('↑', '').replace('↓', '').replace('⚠', 'WARNING').replace('️', '')
                # Remove any remaining non-ASCII characters
                cleaned = re.sub(r'[^\x00-\x7F]+', '', cleaned)
                return cleaned.strip()
            elif isinstance(obj, list):
                return [deep_clean_unicode(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: deep_clean_unicode(v) for k, v in obj.items()}
            return obj
            
        analysis = deep_clean_unicode(raw_analysis)
        
        if not analysis.get("success"):
            raise HTTPException(status_code=500, detail=f"Ultimate scorer failed: {analysis.get('error', 'ML+GPT analysis failed')}")
        
        print(f"ULTIMATE: ML+GPT analysis completed with total: {analysis.get('total_score', 0)}/26")
        
        # Extract scores from Ultimate scorer ML+GPT analysis
        scores = analysis["scores"]
        content_score = scores.get("content", 0)
        linguistic_score = scores.get("linguistic", 0)
        coherence_score = scores.get("development", 0)  # Ultimate uses "development"
        form_score = scores.get("form", 0)
        grammar_score = scores.get("grammar", 0)
        spelling_score = scores.get("spelling", 0)
        vocabulary_score = scores.get("vocabulary", 0)
        
        total_score = round(content_score + linguistic_score + coherence_score + 
                          form_score + grammar_score + spelling_score + vocabulary_score, 1)
        percentage = round((total_score / 26) * 100)
        
        # Determine band
        if percentage >= 85:
            band = "Excellent"
        elif percentage >= 70:
            band = "Very Good"
        elif percentage >= 55:
            band = "Good"
        elif percentage >= 40:
            band = "Fair"
        else:
            band = "Needs Improvement"
        
        # Build detailed feedback from Ultimate scorer ML+GPT analysis
        comp_scores = analysis["component_scores"]
        errors_dict = analysis.get("errors", {})
        
        try:
            print(f"DEBUG: errors_dict keys = {list(errors_dict.keys())}")
            print(f"DEBUG: spelling errors count = {len(errors_dict.get('spelling', []))}")
            print(f"DEBUG: grammar errors count = {len(errors_dict.get('grammar', []))}")
        except Exception as debug_error:
            print(f"DEBUG: Could not print errors_dict due to encoding: {debug_error}")
        
        # Extract SWT-style suggestions from Ultimate scorer
        specific_suggestions = analysis.get("specific_suggestions", [])
        ai_recommendations = analysis.get("ai_recommendations", [])
        strategic_improvements = analysis.get("strategic_improvements", [])
        improvement_areas = analysis.get("improvement_areas", [])
        
        # Function to get SPECIFIC ERROR-BASED suggestions
        def get_error_specific_suggestions(component, component_errors, component_score):
            suggestions = []
            try:
                print(f"DEBUG: get_error_specific_suggestions called for {component}")
                print(f"DEBUG: component_errors count = {len(component_errors) if component_errors else 0}")
                print(f"DEBUG: component_score = {component_score}")
            except Exception as debug_error:
                print(f"DEBUG: Unicode error in suggestion debug: {debug_error}")
            
            # PRIORITY 1: Create suggestions based on ACTUAL ERRORS FOUND
            if component == "spelling" and component_errors:
                print(f"DEBUG: Adding spelling suggestions for {len(component_errors)} errors")
                # Show actual spelling mistakes - handle Unicode safely
                for error in component_errors[:3]:  # Show first 3 errors
                    try:
                        # Aggressive Unicode cleaning for spelling errors
                        import re
                        clean_error = str(error).replace('→', ' to ').replace('✗', '').replace('✓', '').strip()
                        # Remove any remaining non-ASCII characters
                        clean_error = re.sub(r'[^\x00-\x7F]+', '', clean_error).strip()
                        if clean_error:
                            suggestions.append(f"Fix spelling: {clean_error}")
                        else:
                            suggestions.append("Fix spelling error detected")
                    except Exception as e:
                        print(f"DEBUG: Error processing spelling suggestion: {e}")
                        suggestions.append("Fix spelling error found by system")
                if len(component_errors) > 3:
                    suggestions.append(f"Plus {len(component_errors) - 3} more spelling errors to fix")
                    
            elif component == "grammar" and component_errors:
                # Show actual grammar errors
                for error in component_errors[:2]:
                    suggestions.append(f"Grammar issue: {error}")
                suggestions.append("Review all grammar errors detected by the system")
                
            elif component == "vocabulary" and component_errors:
                # Show vocabulary issues
                for error in component_errors[:2]:
                    suggestions.append(f"Vocabulary: {error}")
                    
            elif component == "content" and component_errors:
                # Content gaps
                for gap in component_errors[:2]:
                    suggestions.append(f"Missing: {gap}")
                    
            elif component == "development" and component_errors:
                # Structure issues
                for issue in component_errors[:2]:
                    suggestions.append(f"Structure: {issue}")
                    
            elif component == "linguistic" and component_errors:
                # Linguistic issues
                for issue in component_errors[:2]:
                    suggestions.append(f"Linguistic: {issue}")
                    
            elif component == "form" and component_errors:
                # Form problems
                for issue in component_errors[:2]:
                    suggestions.append(f"Form issue: {issue}")
            
            # PRIORITY 2: Add specific AI recommendations if available
            for rec in specific_suggestions:
                if component.lower() in rec.lower() and len(suggestions) < 3:
                    suggestions.append(rec)
            
            # PRIORITY 3: Add improvement areas from GPT analysis
            for improvement in improvement_areas:
                if component.lower() in improvement.lower() and len(suggestions) < 3:
                    suggestions.append(improvement)
                    
            # PRIORITY 4: Always provide at least basic suggestions
            if len(suggestions) == 0:
                score = component_score
                max_score = 6 if component in ["content", "linguistic", "development", "coherence"] else 2
                
                # Provide specific suggestions for each component
                if component == "spelling":
                    if score < 2.0:
                        suggestions.append("Review your spelling - errors detected")
                        suggestions.append("Use spell-check before submitting")
                        suggestions.append("Practice commonly misspelled words")
                elif component == "grammar":
                    if score < 2.0:
                        suggestions.append("Fix grammatical errors in your essay")
                        suggestions.append("Review punctuation and sentence structure")
                        suggestions.append("Check subject-verb agreement")
                elif component == "vocabulary":
                    suggestions.append("Expand your vocabulary range")
                    suggestions.append("Avoid repeating common words")
                    suggestions.append("Use more academic terminology")
                elif component == "content":
                    suggestions.append("Address all parts of the essay prompt")
                    suggestions.append("Provide specific examples and evidence")
                    suggestions.append("Develop your arguments more thoroughly")
                elif component == "development" or component == "coherence":
                    suggestions.append("Improve paragraph organization")
                    suggestions.append("Use transition words between ideas")
                    suggestions.append("Ensure logical flow of arguments")
                elif component == "linguistic":
                    suggestions.append("Use more complex sentence structures")
                    suggestions.append("Include subordinate clauses")
                    suggestions.append("Vary sentence lengths and types")
                elif component == "form":
                    if score < 2.0:
                        suggestions.append("Check essay length (200-300 words required)")
                        suggestions.append("Use proper paragraph structure")
                        suggestions.append("Include introduction and conclusion")
                    
            print(f"DEBUG: Final suggestions for {component}: {suggestions}")
            return suggestions[:3]  # Maximum 3 suggestions per component
        
        detailed_feedback = {
            "content": EssayFeedback(
                score=content_score,
                justification=comp_scores.get("content", f"Content: {content_score}/6"),
                errors=errors_dict.get("content", []),
                suggestions=get_error_specific_suggestions("content", errors_dict.get("content", []), content_score)
            ),
            "linguistic": EssayFeedback(
                score=linguistic_score,
                justification=comp_scores.get("linguistic", f"Linguistic Range: {linguistic_score}/6"),
                errors=errors_dict.get("linguistic", []),
                suggestions=get_error_specific_suggestions("linguistic", errors_dict.get("linguistic", []), linguistic_score)
            ),
            "coherence": EssayFeedback(
                score=coherence_score,
                justification=comp_scores.get("development", f"Development: {coherence_score}/6"),
                errors=errors_dict.get("development", []),
                suggestions=get_error_specific_suggestions("development", errors_dict.get("development", []), coherence_score)
            ),
            "form": EssayFeedback(
                score=form_score,
                justification=comp_scores.get("form", f"Form: {form_score}/2"),
                errors=errors_dict.get("form", []),
                suggestions=get_error_specific_suggestions("form", errors_dict.get("form", []), form_score)
            ),
            "grammar": EssayFeedback(
                score=grammar_score,
                justification=comp_scores.get("grammar", f"Grammar: {grammar_score}/2"),
                errors=errors_dict.get("grammar", []),
                suggestions=get_error_specific_suggestions("grammar", errors_dict.get("grammar", []), grammar_score)
            ),
            "spelling": EssayFeedback(
                score=spelling_score,
                justification=comp_scores.get("spelling", f"Spelling: {spelling_score}/2"),
                errors=errors_dict.get("spelling", []),
                suggestions=get_error_specific_suggestions("spelling", errors_dict.get("spelling", []), spelling_score)
            ),
            "vocabulary": EssayFeedback(
                score=vocabulary_score,
                justification=comp_scores.get("vocabulary", f"Vocabulary: {vocabulary_score}/2"),
                errors=errors_dict.get("vocabulary", []),
                suggestions=get_error_specific_suggestions("vocabulary", errors_dict.get("vocabulary", []), vocabulary_score)
            )
        }
        
        # Create backward-compatible feedback
        feedback = {
            "content": detailed_feedback["content"].justification,
            "linguistic": detailed_feedback["linguistic"].justification,
            "coherence": detailed_feedback["coherence"].justification,
            "form": detailed_feedback["form"].justification,
            "grammar": detailed_feedback["grammar"].justification,
            "spelling": detailed_feedback["spelling"].justification,
            "vocabulary": detailed_feedback["vocabulary"].justification
        }
        
        # CRITICAL: Clean all Unicode characters that cause encoding issues
        def clean_unicode_text(text):
            if isinstance(text, str):
                return text.replace('→', 'to').replace('✗', '').replace('✓', '').replace('↑', '').replace('↓', '').strip()
            elif isinstance(text, list):
                return [clean_unicode_text(item) for item in text]
            elif isinstance(text, dict):
                return {k: clean_unicode_text(v) for k, v in text.items()}
            return text
            
        # Clean the detailed_feedback to prevent Unicode errors
        for component_name, feedback in detailed_feedback.items():
            feedback.errors = clean_unicode_text(feedback.errors)
            feedback.suggestions = clean_unicode_text(feedback.suggestions)
            feedback.justification = clean_unicode_text(feedback.justification)
        
        # Convert detailed_feedback to dictionary format for the frontend
        feedback_dict = {}
        for component_name, feedback_obj in detailed_feedback.items():
            feedback_dict[component_name] = {
                "score": feedback_obj.score,
                "justification": feedback_obj.justification,
                "errors": feedback_obj.errors,
                "suggestions": feedback_obj.suggestions
            }
        
        return WriteEssayResponse(
            success=True,
            scores={
                "content": content_score,
                "linguistic": linguistic_score,
                "coherence": coherence_score,
                "form": form_score,
                "grammar": grammar_score,
                "spelling": spelling_score,
                "vocabulary": vocabulary_score
            },
            feedback=feedback_dict,
            detailed_feedback=detailed_feedback,
            overall_feedback=analysis.get("verification_notes", f"Your essay scored {total_score}/26."),
            total_score=total_score,
            percentage=percentage,
            band=band,
            word_count=analysis.get("word_count", 0),
            paragraph_count=analysis.get("paragraph_count", 0),
            key_arguments_covered=analysis.get("key_arguments_covered", []),
            key_arguments_missed=analysis.get("key_arguments_missed", []),
            strengths=analysis.get("strengths", analysis.get("swt_style_insights", {}).get("strengths", ["Good effort on essay writing"])),
            improvements=analysis.get("improvement_areas", analysis.get("swt_style_insights", {}).get("improvement_areas", [])),
            ai_recommendations=analysis.get("ai_recommendations", analysis.get("specific_suggestions", analysis.get("strategic_improvements", ["Practice writing regularly", "Focus on grammar and spelling accuracy", "Expand vocabulary range"])))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Essay Scoring Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ULTIMATE Write Essay endpoint - SWT-style precision
@app.post("/api/v1/writing/essay/ultimate")
async def score_ultimate_write_essay_endpoint(
    question_title: str = Form(...),
    essay_prompt: str = Form(...),
    essay_type: str = Form(...),
    key_arguments: str = Form(...),
    sample_essay: str = Form(default=""),
    user_essay: str = Form(...)
):
    """
    ULTIMATE Write Essay scoring with SWT-style precision
    
    Features:
    - Ultra-comprehensive spelling detection (catches "strickly" etc.)
    - Decimal precision scoring (1.8/2, 3.4/6 like APEUni) 
    - GPT as ultimate 100+ point English validator
    - ML error cross-validation and reclassification
    - SWT-style comprehensive final verification
    """
    try:
        # Validate input
        if not user_essay or len(user_essay.strip()) < 50:
            raise HTTPException(status_code=400, detail="Essay too short (minimum 50 characters)")
        
        word_count = len(user_essay.split())
        if word_count < 150:
            raise HTTPException(status_code=400, detail=f"Essay too short: {word_count} words (minimum 150)")
        
        # Try ultimate scorer
        try:
            from app.services.scoring.ultimate_write_essay_scorer import score_ultimate_write_essay
            
            print("🎯 Starting ULTIMATE Write Essay Scoring")
            result = score_ultimate_write_essay(user_essay, essay_prompt)
            
            if not result.get("success"):
                raise Exception(f"Ultimate scoring failed: {result.get('error', 'Unknown error')}")
            
            print(f"✅ Ultimate scoring completed with total score: {result.get('total_score', 0)}/26")
            
            # Transform for frontend compatibility (add SWT-style fields expected by frontend)
            if result.get("success"):
                # Add frontend-compatible fields
                result["strengths"] = result.get("strengths", [])
                result["improvements"] = result.get("improvement_areas", [])
                result["ai_recommendations"] = result.get("ai_recommendations", [])
                result["feedback"] = {
                    "content": f"Content Score: {result.get('scores', {}).get('content', 0)}/6",
                    "form": f"Form Score: {result.get('scores', {}).get('form', 0)}/2",
                    "development": f"Development Score: {result.get('scores', {}).get('development', 0)}/6",
                    "grammar": f"Grammar Score: {result.get('scores', {}).get('grammar', 0)}/2",
                    "linguistic": f"Linguistic Score: {result.get('scores', {}).get('linguistic', 0)}/6",
                    "vocabulary": f"Vocabulary Score: {result.get('scores', {}).get('vocabulary', 0)}/2",
                    "spelling": f"Spelling Score: {result.get('scores', {}).get('spelling', 0)}/2"
                }
                result["overall_feedback"] = f"Your essay scored {result.get('total_score', 0)}/26 ({result.get('percentage', 0)}%) - {result.get('band', 'Unknown')} level. {result.get('verification_notes', '')}"
            
            return result
            
        except ImportError:
            raise HTTPException(status_code=503, detail="Ultimate scorer not available")
        except Exception as e:
            print(f"Ultimate scorer error: {e}")
            raise HTTPException(status_code=500, detail=f"Ultimate scoring error: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Ultimate essay scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Test spelling detection endpoint
@app.post("/api/v1/writing/test-spelling")
async def test_spelling_detection_endpoint(test_text: str = Form(...)):
    """
    Test endpoint for ultra-comprehensive spelling detection
    """
    try:
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
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Ultimate scorer not available")
    except Exception as e:
        print(f"Spelling test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Test GPT spelling detection endpoint
@app.post("/api/v1/writing/test-gpt-spelling")
async def test_gpt_spelling_detection(test_text: str = Form(...)):
    """
    Test endpoint to check if GPT finds spelling errors directly
    """
    try:
        from app.services.scoring.ultimate_write_essay_scorer import get_ultimate_essay_scorer
        
        scorer = get_ultimate_essay_scorer()
        
        if not scorer.use_gpt:
            raise HTTPException(status_code=503, detail="GPT unavailable - check OPENAI_API_KEY")
        
        # Simple ML results for testing
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
        
        # Test GPT spelling detection
        gpt_result = scorer.ultimate_gpt_final_verification("Test spelling", test_text, ml_results)
        
        return {
            "success": True,
            "text_analyzed": test_text,
            "gpt_found_errors": gpt_result.get("additional_errors_found", []),
            "gpt_spelling_score": gpt_result.get("final_scores", {}).get("spelling", 0),
            "api_cost": scorer.total_api_cost,
            "gpt_confidence": gpt_result.get("confidence", 0.0)
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Ultimate scorer not available")
    except Exception as e:
        print(f"GPT spelling test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
