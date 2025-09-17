#!/usr/bin/env python3
"""
Write Essay Scorer - 3-Layer Hybrid System
Layer 1: Rule-based (Grammar, Spelling, Form)
Layer 2: Statistical & Embeddings (Content, Coherence, Vocabulary)
Layer 3: GPT-4o Ultimate Language Expert (100+ point verification)

Total: 26 points across 7 components
"""

import re
import json
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

# Core ML imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    GECTOR_AVAILABLE = True
except ImportError:
    GECTOR_AVAILABLE = False

# NLP tools
import nltk
import spacy
from spellchecker import SpellChecker
import language_tool_python

# Sentence embeddings
from sentence_transformers import SentenceTransformer

# Keyword extraction
from keybert import KeyBERT
from yake import KeywordExtractor

# Vocabulary analysis
from lexicalrichness import LexicalRichness

# OpenAI for GPT-4o
from openai import OpenAI

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy initialization for global instance
_global_scorer = None

def get_essay_scorer():
    """Get or create global essay scorer instance"""
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = WriteEssayScorer()
    return _global_scorer

class WriteEssayScorer:
    """3-Layer Hybrid Scoring System for PTE Write Essay"""
    
    def __init__(self):
        """Initialize all scoring layers"""
        logger.info("🚀 Initializing Write Essay Scorer...")
        
        # Configuration
        self.use_gpt = bool(os.getenv('OPENAI_API_KEY'))
        self.openai_client = None
        self.total_api_cost = 0.0
        
        # Initialize layers
        self._init_grammar_layer()
        self._init_vocabulary_layer()
        self._init_content_layer()
        self._init_gpt_layer()
        
        logger.info("✅ Write Essay Scorer initialized successfully")
    
    def _init_grammar_layer(self):
        """Initialize Layer 1: Grammar checking"""
        logger.info("Initializing Grammar Layer...")
        
        # GECToR for primary grammar correction
        self.gector_model = None
        self.gector_tokenizer = None
        
        if GECTOR_AVAILABLE:
            try:
                model_name = "vennify/t5-base-grammar-correction"
                self.gector_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.gector_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                logger.info("✅ GECToR loaded")
            except Exception as e:
                logger.warning(f"⚠️ GECToR failed: {e}")
        
        # LanguageTool as fallback
        try:
            self.language_tool = language_tool_python.LanguageTool('en-US')
            logger.info("✅ LanguageTool loaded")
        except:
            self.language_tool = None
            logger.warning("⚠️ LanguageTool unavailable")
        
        # spaCy for parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy loaded")
        except:
            self.nlp = None
            logger.warning("⚠️ spaCy unavailable")
    
    def _init_vocabulary_layer(self):
        """Initialize vocabulary and spelling analysis"""
        logger.info("Initializing Vocabulary Layer...")
        
        # Spell checker
        self.spell = SpellChecker()
        
        # CEFR vocabulary levels
        self.cefr_levels = {
            'A1': set(['the', 'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'can', 'will']),
            'A2': set(['should', 'would', 'could', 'must', 'might', 'perhaps', 'probably']),
            'B1': set(['although', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless']),
            'B2': set(['consequently', 'subsequently', 'predominantly', 'substantially', 'remarkably']),
            'C1': set(['notwithstanding', 'nonetheless, 'albeit', 'whereby', 'wherein', 'henceforth']),
            'C2': set(['heretofore', 'inasmuch', 'insofar', 'vis-à-vis', 'paradigm', 'ubiquitous'])
        }
        
        # Academic vocabulary
        self.academic_words = set([
            'analyze', 'analysis', 'concept', 'conceptual', 'constitute', 'context', 'contextual',
            'derive', 'distribution', 'establish', 'estimate', 'evidence', 'export', 'factor',
            'finance', 'financial', 'formula', 'function', 'identify', 'income', 'indicate',
            'individual', 'interpret', 'involve', 'issue', 'labour', 'legal', 'legislate',
            'major', 'method', 'occur', 'percent', 'period', 'policy', 'principle', 'proceed',
            'process', 'require', 'research', 'respond', 'role', 'section', 'sector', 'significant',
            'similar', 'source', 'specific', 'structure', 'theory', 'vary'
        ])
        
        # Bad collocations database
        self.bad_collocations = {
            'do mistake': 'make a mistake',
            'do research': 'conduct research',
            'take decision': 'make a decision',
            'say lies': 'tell lies',
            'do homework': 'do homework',  # This is actually correct
            'make homework': 'do homework',  # This is wrong
        }
    
    def _init_content_layer(self):
        """Initialize Layer 2: Content and coherence analysis"""
        logger.info("Initializing Content Layer...")
        
        # Sentence embeddings for similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
        except:
            self.sentence_model = None
            logger.warning("⚠️ Sentence transformer unavailable")
        
        # Keyword extraction
        try:
            self.keybert = KeyBERT()
            self.yake = KeywordExtractor()
            logger.info("✅ Keyword extractors loaded")
        except:
            self.keybert = None
            self.yake = None
            logger.warning("⚠️ Keyword extraction unavailable")
        
        # Discourse markers for coherence
        self.discourse_markers = {
            'addition': ['furthermore', 'moreover', 'additionally', 'besides', 'also'],
            'contrast': ['however', 'nevertheless', 'nonetheless', 'conversely', 'whereas'],
            'cause': ['because', 'since', 'as', 'due to', 'owing to'],
            'effect': ['therefore', 'thus', 'consequently', 'hence', 'as a result'],
            'example': ['for example', 'for instance', 'such as', 'namely', 'specifically'],
            'conclusion': ['in conclusion', 'to conclude', 'in summary', 'overall', 'ultimately']
        }
    
    def _init_gpt_layer(self):
        """Initialize GPT-4o layer"""
        if self.use_gpt:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("✅ GPT-4o initialized")
            except Exception as e:
                logger.warning(f"⚠️ GPT-4o unavailable: {e}")
                self.use_gpt = False
    
    def score_essay(self, user_essay: str, essay_prompt: str) -> Dict:
        """
        Main scoring function - orchestrates all 3 layers
        Returns comprehensive scoring with 26-point total
        """
        try:
            logger.info("="*50)
            logger.info("🎯 STARTING WRITE ESSAY SCORING")
            logger.info("="*50)
            
            # Layer 1: Rule-based checks
            grammar_results = self.check_grammar(user_essay)
            spelling_results = self.check_spelling(user_essay)
            form_results = self.check_form_requirements(user_essay)
            
            # Layer 2: Statistical & Embeddings
            content_results = self.analyze_content(user_essay, essay_prompt)
            coherence_results = self.analyze_coherence(user_essay)
            linguistic_results = self.analyze_linguistic_range(user_essay)
            vocabulary_results = self.analyze_vocabulary_range(user_essay)
            
            # Layer 3: GPT-4o Ultimate Judge
            gpt_verification = self.gpt_final_verification(
                essay_prompt, user_essay,
                grammar_results, spelling_results, form_results,
                content_results, coherence_results, linguistic_results, vocabulary_results
            )
            
            # Compile final scores
            if gpt_verification.get("success"):
                final_scores = gpt_verification["verified_scores"]
                final_feedback = gpt_verification.get("final_feedback", {})
            else:
                # Fallback to ML scores if GPT fails
                final_scores = {
                    'content': content_results[0],
                    'form': form_results[0],
                    'development': coherence_results[0],
                    'grammar': grammar_results[0],
                    'linguistic': linguistic_results[0],
                    'vocabulary': vocabulary_results[0],
                    'spelling': spelling_results[0]
                }
                final_feedback = {}
            
            # Calculate total
            total_score = sum(final_scores.values())
            percentage = round((total_score / 26) * 100)
            
            # Determine band
            if percentage >= 90:
                band = "Expert"
            elif percentage >= 79:
                band = "Very Good"
            elif percentage >= 65:
                band = "Good"
            elif percentage >= 50:
                band = "Competent"
            else:
                band = "Limited"
            
            # Return comprehensive response
            return {
                "success": True,
                "total_score": total_score,
                "percentage": percentage,
                "band": band,
                "scores": final_scores,
                "component_feedback": {
                    "content": f"Content: {final_scores['content']}/6",
                    "form": f"Formal Requirements: {final_scores['form']}/2",
                    "development": f"Development & Coherence: {final_scores['development']}/6",
                    "grammar": f"Grammar: {final_scores['grammar']}/2",
                    "linguistic": f"General Linguistic Range: {final_scores['linguistic']}/6",
                    "vocabulary": f"Vocabulary Range: {final_scores['vocabulary']}/2",
                    "spelling": f"Spelling: {final_scores['spelling']}/2"
                },
                "errors": {
                    "grammar": grammar_results[1],
                    "spelling": spelling_results[1],
                    "vocabulary": vocabulary_results[1],
                    "coherence": coherence_results[1]
                },
                "suggestions": gpt_verification.get("detailed_suggestions", {}),
                "strengths": final_feedback.get("strengths", []),
                "improvements": final_feedback.get("critical_improvements", []),
                "harsh_assessment": final_feedback.get("harsh_assessment", ""),
                "api_cost": self.total_api_cost
            }
            
        except Exception as e:
            logger.error(f"Essay scoring failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def check_grammar(self, text: str) -> Tuple[float, List[str]]:
        """Layer 1: Grammar checking (0-2 points)"""
        errors = []
        
        # Try GECToR first
        if self.gector_model and self.gector_tokenizer:
            try:
                inputs = self.gector_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                outputs = self.gector_model.generate(**inputs, max_length=512)
                corrected = self.gector_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if corrected != text:
                    # Find differences
                    import difflib
                    diff = list(difflib.unified_diff(text.split(), corrected.split(), lineterm=''))
                    for line in diff:
                        if line.startswith('-') and not line.startswith('---'):
                            errors.append(f"Grammar: {line[1:]}")
            except:
                pass
        
        # Fallback to LanguageTool
        if self.language_tool and len(errors) < 3:
            matches = self.language_tool.check(text)
            for match in matches[:10]:  # Limit to 10 errors
                if match.ruleId not in ['WHITESPACE_RULE', 'UPPERCASE_SENTENCE_START']:
                    errors.append(f"{match.message}")
        
        # Calculate score (max 2 points, -0.2 per error)
        score = max(0, 2.0 - (len(errors) * 0.2))
        return round(score, 1), errors
    
    def check_spelling(self, text: str) -> Tuple[float, List[str]]:
        """Layer 1: Spelling check (0-2 points)"""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        misspelled = self.spell.unknown(words)
        
        errors = []
        for word in misspelled:
            correction = self.spell.correction(word)
            if correction and correction != word:
                errors.append(f"{word} → {correction}")
        
        # Score calculation (max 2 points, -0.5 per spelling error)
        score = max(0, 2.0 - (len(errors) * 0.5))
        return round(score, 1), errors
    
    def check_form_requirements(self, text: str) -> Tuple[float, List[str]]:
        """Layer 1: Check formal requirements (0-2 points)"""
        issues = []
        score = 2.0
        
        # Word count (200-300 words)
        word_count = len(text.split())
        if word_count < 200:
            issues.append(f"Too short: {word_count} words (minimum 200)")
            score = 0
        elif word_count > 300:
            issues.append(f"Too long: {word_count} words (maximum 300)")
            score = 0
        
        # Paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            issues.append("Missing proper paragraph structure (need intro, body, conclusion)")
            score = max(0, score - 1)
        
        return score, issues
    
    def analyze_content(self, essay: str, prompt: str) -> Tuple[float, List[str]]:
        """Layer 2: Content analysis (0-6 points)"""
        gaps = []
        
        if not self.sentence_model:
            return 3.0, ["Content analysis unavailable"]
        
        # Semantic similarity between essay and prompt
        essay_emb = self.sentence_model.encode(essay, convert_to_tensor=True)
        prompt_emb = self.sentence_model.encode(prompt, convert_to_tensor=True)
        
        from torch.nn import functional as F
        similarity = F.cosine_similarity(essay_emb.unsqueeze(0), prompt_emb.unsqueeze(0)).item()
        
        # Extract key concepts from prompt
        if self.keybert:
            prompt_keywords = self.keybert.extract_keywords(prompt, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
            prompt_keywords = [kw[0] for kw in prompt_keywords]
            
            # Check coverage
            essay_lower = essay.lower()
            for keyword in prompt_keywords:
                if keyword.lower() not in essay_lower:
                    gaps.append(f"Missing key concept: {keyword}")
        
        # Score based on similarity and coverage
        coverage_penalty = len(gaps) * 0.5
        base_score = similarity * 6.0  # Convert to 0-6 scale
        content_score = max(0, min(6.0, base_score - coverage_penalty))
        
        # Apply lenient percentage-based scoring like SWT
        coverage_percentage = (1 - len(gaps) * 0.15) * 100
        
        if coverage_percentage >= 60:  # Good coverage
            content_score = 6.0
        elif coverage_percentage >= 50:
            content_score = 4.5
        elif coverage_percentage >= 40:
            content_score = 3.0
        elif coverage_percentage >= 30:
            content_score = 1.5
        else:
            content_score = 0.0
        
        return round(content_score, 1), gaps
    
    def analyze_coherence(self, essay: str) -> Tuple[float, List[str]]:
        """Layer 2: Development, structure and coherence (0-6 points)"""
        issues = []
        score = 6.0
        
        # Check discourse markers
        discourse_found = []
        essay_lower = essay.lower()
        
        for category, markers in self.discourse_markers.items():
            found = False
            for marker in markers:
                if marker in essay_lower:
                    discourse_found.append(marker)
                    found = True
                    break
            if not found and category in ['contrast', 'cause', 'effect']:
                issues.append(f"Missing {category} discourse markers")
                score -= 1.0
        
        # Check paragraph development
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        for i, para in enumerate(paragraphs):
            sentences = para.split('.')
            if len(sentences) < 3:
                issues.append(f"Paragraph {i+1} underdeveloped")
                score -= 0.5
        
        return max(0, round(score, 1)), issues
    
    def analyze_linguistic_range(self, essay: str) -> Tuple[float, List[str]]:
        """Layer 2: General linguistic range (0-6 points)"""
        features = []
        
        # Sentence variety
        sentences = [s.strip() for s in essay.split('.') if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences]
        
        # Check for variety
        if len(set(sentence_lengths)) < 5:
            features.append("Limited sentence variety")
        
        # Lexical diversity
        try:
            lex = LexicalRichness(essay)
            ttr = lex.ttr  # Type-token ratio
            if ttr < 0.4:
                features.append(f"Low lexical diversity (TTR: {ttr:.2f})")
        except:
            pass
        
        # Complex structures
        complex_patterns = [
            r'\b(although|whereas|while|despite|in spite of)\b',
            r'\b(having|being|having been)\b',
            r'\b(which|whom|whose|whereby)\b',
            r'\b(not only.*but also|either.*or|neither.*nor)\b'
        ]
        
        complex_count = 0
        for pattern in complex_patterns:
            complex_count += len(re.findall(pattern, essay, re.I))
        
        # Score based on complexity
        if complex_count >= 8:
            score = 6.0
        elif complex_count >= 6:
            score = 4.5
        elif complex_count >= 4:
            score = 3.0
        elif complex_count >= 2:
            score = 1.5
        else:
            score = 0.5
            features.append("Lack of complex structures")
        
        return round(score, 1), features
    
    def analyze_vocabulary_range(self, essay: str) -> Tuple[float, List[str]]:
        """Layer 2: Vocabulary range (0-2 points)"""
        issues = []
        
        words = re.findall(r'\b[a-zA-Z]+\b', essay.lower())
        
        # Check academic vocabulary usage
        academic_count = sum(1 for word in words if word in self.academic_words)
        academic_percentage = (academic_count / len(words)) * 100 if words else 0
        
        if academic_percentage < 5:
            issues.append(f"Low academic vocabulary ({academic_percentage:.1f}%)")
        
        # Check for bad collocations
        essay_lower = essay.lower()
        for bad, good in self.bad_collocations.items():
            if bad in essay_lower:
                issues.append(f"Collocation: '{bad}' → '{good}'")
        
        # CEFR level assessment
        cefr_distribution = {}
        for level, vocab in self.cefr_levels.items():
            count = sum(1 for word in words if word in vocab)
            cefr_distribution[level] = count
        
        # Score based on sophistication
        if cefr_distribution.get('C1', 0) + cefr_distribution.get('C2', 0) >= 5:
            score = 2.0
        elif cefr_distribution.get('B2', 0) >= 5:
            score = 1.5
        elif cefr_distribution.get('B1', 0) >= 5:
            score = 1.0
        else:
            score = 0.5
            issues.append("Basic vocabulary level")
        
        return round(score, 1), issues
    
    def gpt_final_verification(self, prompt: str, essay: str, 
                              grammar_results: Tuple, spelling_results: Tuple,
                              form_results: Tuple, content_results: Tuple,
                              coherence_results: Tuple, linguistic_results: Tuple,
                              vocabulary_results: Tuple) -> Dict:
        """
        Layer 3: GPT-4o Ultimate Language Expert
        Performs 100+ point inspection with superior intelligence
        """
        if not self.use_gpt or not self.openai_client:
            return {"success": False, "reason": "GPT unavailable"}
        
        try:
            # Prepare comprehensive prompt
            gpt_prompt = f"""
You are the ULTIMATE PTE WRITE ESSAY EXPERT performing comprehensive 100+ point analysis.
Ignore ML findings - do your own COMPLETE inspection using GPT intelligence.

ESSAY PROMPT: {prompt}

USER ESSAY:
{essay}

🔍 COMPREHENSIVE ANALYSIS REQUIRED (26 POINTS TOTAL):

1. CONTENT (0-6 POINTS) - Check ALL:
   - Direct prompt relevance and task fulfillment
   - Argument depth and sophistication
   - Supporting evidence quality
   - Critical thinking demonstration
   - Logical reasoning chains
   - Counterargument consideration
   - Conclusion strength
   
2. FORMAL REQUIREMENTS (0-2 POINTS) - Check ALL:
   - Word count (200-300 strict)
   - Paragraph structure (intro, body paragraphs, conclusion)
   - Essay format conventions
   
3. DEVELOPMENT, STRUCTURE & COHERENCE (0-6 POINTS) - Check ALL:
   - Introduction effectiveness (hook, thesis, preview)
   - Body paragraph unity (topic sentences, support, transitions)
   - Logical flow between ideas
   - Discourse markers usage
   - Cohesive devices effectiveness
   - Conclusion synthesis
   - Overall essay architecture
   
4. GRAMMAR (0-2 POINTS) - Check 50+ RULES:
   - Tense consistency and accuracy
   - Subject-verb agreement (including tricky cases)
   - Article usage (a/an/the/zero article)
   - Preposition accuracy
   - Pronoun reference clarity
   - Conditional structures
   - Passive voice appropriateness
   - Modal verb usage
   - Gerund/infinitive patterns
   - Parallel structures
   - Comma splices and run-ons
   - Fragment detection
   - Dangling modifiers
   
5. GENERAL LINGUISTIC RANGE (0-6 POINTS) - Check ALL:
   - Sentence variety (simple, compound, complex, compound-complex)
   - Syntactic complexity
   - Clause variety (relative, adverbial, nominal)
   - Inversion and fronting
   - Emphatic structures
   - Academic register maintenance
   - Tone consistency
   - Stylistic sophistication
   
6. VOCABULARY RANGE (0-2 POINTS) - Check ALL:
   - Academic vocabulary deployment
   - Precise word choice
   - Collocation accuracy
   - Idiomatic expressions
   - Synonym variation
   - Word form accuracy (derivations)
   - Semantic precision
   - Register appropriateness
   - Avoiding repetition
   - Sophisticated lexis
   
7. SPELLING (0-2 POINTS) - Check EVERY WORD:
   - All spelling accuracy
   - Commonly confused words
   - British/American consistency
   - Proper noun spelling
   - Academic terminology

SCORING INSTRUCTION:
- Be ULTRA-HARSH like APEUni standards
- Deduct for EVERY tiny error found
- Content: Focus on addressing the EXACT prompt requirements
- Look for sophisticated academic writing features
- Check for plagiarism patterns or memorized templates

Return STRICT JSON:
{{
    "success": true,
    "verified_scores": {{
        "content": <0-6>,
        "form": <0-2>,
        "development": <0-6>,
        "grammar": <0-2>,
        "linguistic": <0-6>,
        "vocabulary": <0-2>,
        "spelling": <0-2>
    }},
    "detailed_errors_with_suggestions": {{
        "content": [
            {{
                "issue": "specific content gap",
                "suggestion": "how to improve",
                "impact": "why it matters"
            }}
        ],
        "grammar": [
            {{
                "error": "exact error found",
                "correction": "correct version",
                "rule": "grammar rule violated"
            }}
        ],
        "vocabulary": [
            {{
                "error": "word choice issue",
                "correction": "better alternative",
                "reason": "why change needed"
            }}
        ],
        "coherence": [
            {{
                "issue": "flow problem",
                "suggestion": "improvement",
                "location": "where in essay"
            }}
        ]
    }},
    "final_feedback": {{
        "strengths": ["strength 1", "strength 2"],
        "critical_improvements": ["must fix 1", "must fix 2", "must fix 3"],
        "harsh_assessment": "Brutally honest overall assessment focusing on weaknesses",
        "band_justification": "Why this score band was assigned"
    }}
}}"""

            # Call GPT-4o
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a strict PTE essay examiner. Return valid JSON only."},
                    {"role": "user", "content": gpt_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            gpt_text = response.choices[0].message.content
            
            # Clean and parse JSON
            gpt_text = re.sub(r'^```json\s*|\s*```$', '', gpt_text.strip())
            result = json.loads(gpt_text)
            
            # Track cost
            self.total_api_cost += (response.usage.prompt_tokens * 0.00001) + (response.usage.completion_tokens * 0.00003)
            
            logger.info("✅ GPT verification complete")
            return result
            
        except Exception as e:
            logger.error(f"GPT verification failed: {e}")
            return {"success": False, "reason": str(e)}


# Export function for easy import
def score_write_essay(user_essay: str, essay_prompt: str) -> Dict:
    """
    Convenience function to score an essay
    
    Args:
        user_essay: Student's essay (200-300 words)
        essay_prompt: Essay question/topic
    
    Returns:
        Comprehensive scoring dictionary with 26-point total
    """
    scorer = get_essay_scorer()
    return scorer.score_essay(user_essay, essay_prompt)