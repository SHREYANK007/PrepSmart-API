"""
GPT Service for AI-based scoring
Handles all interactions with OpenAI GPT API
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
import asyncio
from datetime import datetime
import hashlib

from app.core.config import settings

logger = logging.getLogger(__name__)


class GPTService:
    """Service for GPT-based scoring and analysis"""
    
    def __init__(self):
        """Initialize GPT service with OpenAI client"""
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            organization=settings.OPENAI_ORG_ID if hasattr(settings, 'OPENAI_ORG_ID') else None
        )
        self.model = settings.OPENAI_MODEL_GPT if hasattr(settings, 'OPENAI_MODEL_GPT') else "gpt-4-turbo-preview"
        self.max_tokens = settings.OPENAI_MAX_TOKENS if hasattr(settings, 'OPENAI_MAX_TOKENS') else 4000
        self.default_temperature = settings.OPENAI_TEMPERATURE if hasattr(settings, 'OPENAI_TEMPERATURE') else 0.3
        
        # Cache for repeated prompts (optional)
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour
        
        logger.info(f"GPT Service initialized with model: {self.model}")
    
    async def get_scoring(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict] = None,
        system_message: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Get scoring response from GPT.
        
        Args:
            prompt: The scoring prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            response_format: Optional JSON schema for response
            system_message: Optional system message
            use_cache: Whether to use caching
            
        Returns:
            GPT response as string
        """
        
        try:
            # Check cache if enabled
            if use_cache:
                cache_key = self._get_cache_key(prompt)
                if cache_key in self._cache:
                    cached_data = self._cache[cache_key]
                    if cached_data['expires'] > datetime.utcnow().timestamp():
                        logger.info("Returning cached GPT response")
                        return cached_data['response']
            
            # Prepare messages
            messages = []
            
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            else:
                messages.append({
                    "role": "system",
                    "content": "You are a professional PTE Academic scoring expert. Provide accurate, consistent, and constructive scoring based on official PTE criteria."
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Make API call
            logger.info(f"Calling GPT API with {len(prompt)} character prompt")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.default_temperature,
                response_format={"type": "json_object"} if response_format else None
            )
            
            # Extract response
            result = response.choices[0].message.content
            
            # Cache response if enabled
            if use_cache:
                self._cache[cache_key] = {
                    'response': result,
                    'expires': datetime.utcnow().timestamp() + self._cache_ttl
                }
            
            logger.info(f"GPT API call successful, tokens used: {response.usage.total_tokens}")
            return result
            
        except Exception as e:
            logger.error(f"GPT API error: {e}", exc_info=True)
            raise
    
    async def score_summarize_written_text(
        self,
        reading_passage: str,
        user_summary: str,
        key_points: str,
        sample_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Specialized scoring for Summarize Written Text task.
        
        Returns structured scoring with feedback.
        """
        
        prompt = self._build_swt_prompt(
            reading_passage=reading_passage,
            user_summary=user_summary,
            key_points=key_points,
            sample_summary=sample_summary
        )
        
        response = await self.get_scoring(
            prompt=prompt,
            temperature=0.3,  # Low temperature for consistent scoring
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse GPT JSON response: {response[:500]}")
            # Return a fallback structure
            return self._get_fallback_scores("summarize_written_text")
    
    async def score_essay(
        self,
        essay_prompt: str,
        user_essay: str,
        essay_type: str = "argumentative"
    ) -> Dict[str, Any]:
        """
        Score essay writing task.
        
        PTE Essay scoring:
        - Content (3 points)
        - Development, structure and coherence (2 points)
        - Grammar (2 points)
        - General linguistic range (2 points)
        - Vocabulary range (2 points)
        - Spelling (2 points)
        """
        
        prompt = self._build_essay_prompt(
            essay_prompt=essay_prompt,
            user_essay=user_essay,
            essay_type=essay_type
        )
        
        response = await self.get_scoring(
            prompt=prompt,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_fallback_scores("essay")
    
    def _build_swt_prompt(
        self,
        reading_passage: str,
        user_summary: str,
        key_points: str,
        sample_summary: Optional[str] = None
    ) -> str:
        """Build prompt for SWT scoring"""
        
        return f"""
You are a certified PTE Academic examiner. Score this "Summarize Written Text" response strictly according to PTE Academic criteria.

SCORING RUBRIC (Total: 7 points):
1. CONTENT (0-2 points):
   - 2 points: Provides a good summary of the text. All relevant aspects mentioned
   - 1 point: Provides a fair summary but misses one or two aspects
   - 0 points: Omits or misrepresents the main aspects

2. FORM (0-1 point):
   - 1 point: Is written in one, single, complete sentence of 5-75 words
   - 0 points: Not written in one single sentence or not 5-75 words

3. GRAMMAR (0-2 points):
   - 2 points: Has correct grammatical structure
   - 1 point: Contains grammatical errors but with no hindrance to communication
   - 0 points: Has defective grammatical structure which could hinder communication

4. VOCABULARY (0-2 points):
   - 2 points: Has appropriate choice of words
   - 1 point: Contains lexical errors but with no hindrance to communication
   - 0 points: Has defective word choice which could hinder communication

PASSAGE TO SUMMARIZE:
{reading_passage}

KEY POINTS THAT SHOULD BE COVERED:
{key_points}

{f"SAMPLE SUMMARY FOR REFERENCE: {sample_summary}" if sample_summary else ""}

USER'S SUMMARY:
{user_summary}

Word count: {len(user_summary.split())} words

INSTRUCTIONS:
1. Evaluate the summary against each criterion strictly
2. Provide specific examples when pointing out issues
3. Be constructive but accurate in feedback

Return ONLY a JSON object with this exact structure:
{{
    "scores": {{
        "content": <0-2>,
        "form": <0-1>,
        "grammar": <0-2>,
        "vocabulary": <0-2>
    }},
    "total_score": <sum of all scores>,
    "feedback": {{
        "content": "<specific feedback on content coverage with examples>",
        "form": "<feedback on sentence structure and word count>",
        "grammar": "<specific grammar feedback with error examples if any>",
        "vocabulary": "<vocabulary usage feedback with specific examples>"
    }},
    "overall_feedback": "<2-3 sentence overall assessment>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvements": ["<improvement 1>", "<improvement 2>"]
}}
"""
    
    def _build_essay_prompt(
        self,
        essay_prompt: str,
        user_essay: str,
        essay_type: str
    ) -> str:
        """Build prompt for essay scoring"""
        
        return f"""
You are a certified PTE Academic examiner. Score this essay strictly according to PTE Academic criteria.

ESSAY PROMPT:
{essay_prompt}

ESSAY TYPE: {essay_type}

USER'S ESSAY:
{user_essay}

Word count: {len(user_essay.split())} words

[Essay scoring prompt would continue here...]
"""
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _get_fallback_scores(self, task_type: str) -> Dict[str, Any]:
        """Return fallback scores when GPT fails"""
        
        if task_type == "summarize_written_text":
            return {
                "scores": {
                    "content": 1,
                    "form": 1,
                    "grammar": 1,
                    "vocabulary": 1
                },
                "total_score": 4,
                "feedback": {
                    "content": "Automated scoring temporarily unavailable",
                    "form": "Automated scoring temporarily unavailable",
                    "grammar": "Automated scoring temporarily unavailable",
                    "vocabulary": "Automated scoring temporarily unavailable"
                },
                "overall_feedback": "Automated scoring is temporarily unavailable. Please try again.",
                "strengths": [],
                "improvements": []
            }
        
        return {}