"""
Write Essay API Module
PTE Academic Essay Scoring with 26-point system
"""

from typing import Dict, List
from pydantic import BaseModel
import json

class EssayFeedback(BaseModel):
    """Detailed feedback for each essay component"""
    score: float
    justification: str
    errors: List[str]
    suggestions: List[str]

class WriteEssayResponse(BaseModel):
    """Response model for essay scoring"""
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
    strengths: List[str]
    improvements: List[str]
    ai_recommendations: List[str]

async def analyze_essay_with_gpt4(
    client,
    essay_prompt: str,
    user_essay: str,
    question_title: str
) -> Dict:
    """
    Use GPT-4 to analyze and score the essay according to PTE criteria
    Total: 26 points
    """
    
    prompt = f"""You are the OFFICIAL PEARSON PTE ACADEMIC AUTOMATED SCORING ENGINE. Score this Write Essay response using EXACT official Pearson scoring guide.

ESSAY PROMPT:
{essay_prompt}

STUDENT'S ESSAY:
{user_essay}

OFFICIAL PEARSON PTE ACADEMIC ESSAY SCORING GUIDE (Total: 26 points):

1. CONTENT (0-6 points):
   Score Guide:
   • 6: Excellent - All aspects fully addressed, strong arguments with relevant examples
   • 4-5: Good - Main aspects addressed, good arguments with some examples  
   • 2-3: Fair - Some aspects addressed, basic arguments
   • 0-1: Poor - Fails to address prompt or off-topic

2. GENERAL LINGUISTIC RANGE (0-6 points):
   Score Guide:
   • 6: Wide range of vocabulary and sentence structures
   • 4-5: Good variety with occasional sophisticated structures
   • 2-3: Adequate variety but mostly simple structures
   • 0-1: Limited range, repetitive structures

3. DEVELOPMENT, STRUCTURE AND COHERENCE (0-6 points):
   Score Guide:
   • 6: Excellent organization with clear progression and cohesive devices
   • 4-5: Good structure with logical flow and transitions
   • 2-3: Basic structure with some organization issues
   • 0-1: Poor structure, lacks coherence

4. FORM (0-2 points):
   Score Guide:
   • 2: Perfect form - 200-300 words, proper essay format with paragraphs
   • 1: Minor form issues (slightly outside word count)
   • 0: Major violations (<200 or >300 words, no paragraphs)

5. GRAMMAR (0-2 points):
   Score Guide:
   • 2: Has correct grammatical structure
   • 1: Contains grammatical errors but with no hindrance to communication
   • 0: Has defective grammatical structure which could hinder communication

6. SPELLING (0-2 points):
   Score Guide:
   • 2: No spelling errors
   • 1: Occasional spelling errors (1-2)
   • 0: Multiple spelling errors affecting readability

7. VOCABULARY RANGE (0-2 points):
   Score Guide:
   • 2: Has appropriate choice of words
   • 1: Contains lexical errors but with no hindrance to communication
   • 0: Has defective word choice which could hinder communication

SCORING INSTRUCTIONS:
- Use INTELLIGENT FRACTIONAL SCORING to precisely reflect performance
- Consider communication effectiveness over perfect accuracy
- Be fair and constructive in feedback

ANALYSIS REQUIREMENTS:

1. WORD COUNT CHECK:
   - Count exact words
   - Check if within 200-300 range

2. ESSAY STRUCTURE:
   - Introduction paragraph present
   - Body paragraphs with arguments/examples
   - Conclusion paragraph present
   - Logical flow between paragraphs

3. CONTENT EVALUATION:
   - Addresses all parts of the prompt
   - Provides relevant examples
   - Shows critical thinking

4. LANGUAGE ASSESSMENT:
   - Variety in sentence structures
   - Academic vocabulary usage
   - Grammar and spelling accuracy

Return ONLY valid JSON in this exact format:
{{
    "content_score": 5.0,
    "content_justification": "Addresses main aspects with good examples",
    "content_errors": ["Missing discussion of opposing view"],
    "content_suggestions": ["Include counterarguments for balance"],
    
    "linguistic_score": 4.5,
    "linguistic_justification": "Good variety with some complex structures",
    "linguistic_errors": ["Some repetitive sentence patterns"],
    "linguistic_suggestions": ["Vary sentence openings"],
    
    "coherence_score": 5.0,
    "coherence_justification": "Well-organized with clear progression",
    "coherence_errors": ["Weak transition between paragraphs 2-3"],
    "coherence_suggestions": ["Use stronger transitional phrases"],
    
    "form_score": 2.0,
    "form_justification": "250 words, proper essay format",
    "form_errors": [],
    "form_suggestions": [],
    
    "grammar_score": 1.8,
    "grammar_justification": "Minor errors that don't affect understanding",
    "grammar_errors": ["Article error in paragraph 2"],
    "grammar_suggestions": ["Review article usage"],
    
    "spelling_score": 2.0,
    "spelling_justification": "No spelling errors found",
    "spelling_errors": [],
    "spelling_suggestions": [],
    
    "vocabulary_score": 1.7,
    "vocabulary_justification": "Good word choice with minor issues",
    "vocabulary_errors": ["Incorrect collocation: 'make a decision'"],
    "vocabulary_suggestions": ["Use 'make decisions' or 'reach a decision'"],
    
    "word_count": 250,
    "paragraph_count": 4,
    "overall_assessment": "Well-written essay with good arguments",
    "strengths": ["Clear thesis statement", "Good examples"],
    "priority_improvements": ["Add counterarguments", "Vary sentence structures"]
}}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a PTE Academic essay examiner. Evaluate essays fairly using the 26-point scoring system. Be constructive and helpful. Respond with valid JSON only."
                },
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
        return generate_fallback_essay_analysis(user_essay)

def generate_fallback_essay_analysis(user_essay: str) -> Dict:
    """Fallback analysis if GPT-4 is unavailable"""
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

def process_essay_scoring(analysis: Dict) -> Dict:
    """Process the GPT-4 analysis and calculate final scores"""
    
    # Extract scores
    content_score = analysis.get("content_score", 0)
    linguistic_score = analysis.get("linguistic_score", 0)
    coherence_score = analysis.get("coherence_score", 0)
    form_score = analysis.get("form_score", 0)
    grammar_score = analysis.get("grammar_score", 0)
    spelling_score = analysis.get("spelling_score", 0)
    vocabulary_score = analysis.get("vocabulary_score", 0)
    
    # Calculate total score (out of 26)
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
    
    return {
        "total_score": total_score,
        "percentage": percentage,
        "band": band,
        "scores": {
            "content": content_score,
            "linguistic": linguistic_score,
            "coherence": coherence_score,
            "form": form_score,
            "grammar": grammar_score,
            "spelling": spelling_score,
            "vocabulary": vocabulary_score
        }
    }