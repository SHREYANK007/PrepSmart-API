"""
Enhanced Hybrid Scoring System - 3-Layer Architecture with GPT Verification
Layer 1: Grammar (GECToR + LanguageTool)
Layer 2: Vocabulary (CEFR + Collocations + Spelling)
Layer 3: Content (Embeddings + Keyword Extraction)
Final: GPT Verification with structured output
"""

import re
import json
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from scipy.stats import pearsonr

# Core ML imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    GECTOR_AVAILABLE = True
except ImportError:
    GECTOR_AVAILABLE = False
    
from sentence_transformers import SentenceTransformer, util
import language_tool_python

# Keyword extraction
try:
    from keybert import KeyBERT
    from rake_nltk import Rake
    KEYWORD_EXTRACTION_AVAILABLE = True
except ImportError:
    KEYWORD_EXTRACTION_AVAILABLE = False

# OpenAI
from openai import OpenAI

# NLTK for additional NLP
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GrammarError:
    """Structure for grammar errors"""
    original: str
    corrected: str
    error_type: str
    position: int
    
@dataclass
class VocabularyIssue:
    """Structure for vocabulary issues"""
    word: str
    issue_type: str  # spelling, informal, cefr_mismatch, collocation
    suggestion: str
    severity: float  # 0.1 to 0.5

@dataclass
class ContentGap:
    """Structure for content gaps"""
    missing_keyword: str
    importance: float
    category: str  # main_idea, supporting_detail, relationship


class HybridScorer:
    """
    Production-ready 3-Layer Hybrid Scoring System with GPT verification
    """
    
    def __init__(self):
        """Initialize all scoring engines with graceful fallbacks"""
        logger.info("Initializing Enhanced Hybrid Scorer...")
        
        # Layer 1: Grammar
        self._init_grammar_layer()
        
        # Layer 2: Vocabulary  
        self._init_vocabulary_layer()
        
        # Layer 3: Content
        self._init_content_layer()
        
        # Final: GPT
        self._init_gpt_layer()
        
        # Calibration data storage
        self.calibration_data = []
        
        logger.info("✅ Enhanced Hybrid Scorer initialized successfully")
    
    def _init_grammar_layer(self):
        """Initialize grammar checking models"""
        logger.info("Initializing Grammar Layer...")
        
        # Try GECToR first
        self.gector_model = None
        self.gector_tokenizer = None
        
        if GECTOR_AVAILABLE:
            try:
                # Use a lighter grammar correction model from HuggingFace
                model_name = "vennify/t5-base-grammar-correction"
                self.gector_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.gector_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                logger.info("✅ GECToR grammar model loaded")
            except Exception as e:
                logger.warning(f"Failed to load GECToR model: {e}")
                self.gector_model = None
        
        # LanguageTool as fallback
        try:
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
            logger.info("✅ LanguageTool initialized as fallback")
        except Exception as e:
            logger.warning(f"LanguageTool initialization failed: {e}")
            self.grammar_tool = None
    
    def _init_vocabulary_layer(self):
        """Initialize vocabulary analysis tools"""
        logger.info("Initializing Vocabulary Layer...")
        
        # CEFR word lists (simplified version for demo)
        self.cefr_levels = {
            'A1': {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'do', 'does', 
                   'go', 'come', 'get', 'make', 'see', 'know', 'think', 'good', 'bad', 'big', 'small'},
            'A2': {'because', 'but', 'so', 'when', 'where', 'why', 'how', 'can', 'could', 'will', 
                   'would', 'should', 'must', 'may', 'might', 'want', 'need', 'like', 'love', 'hate'},
            'B1': {'although', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
                   'consequently', 'additionally', 'specifically', 'generally', 'particularly'},
            'B2': {'significant', 'substantial', 'considerable', 'fundamental', 'essential', 
                   'crucial', 'vital', 'demonstrate', 'illustrate', 'emphasize', 'facilitate'},
            'C1': {'notwithstanding', 'albeit', 'whereby', 'inasmuch', 'nonetheless', 'henceforth',
                   'paradigm', 'methodology', 'framework', 'synthesis', 'hypothesis', 'empirical'},
            'C2': {'quintessential', 'ubiquitous', 'dichotomy', 'paradoxical', 'epistemological',
                   'hermeneutical', 'phenomenological', 'ontological', 'axiological', 'teleological'}
        }
        
        # Collocation rules
        self.collocation_errors = {
            r'\bmake\s+(a\s+)?research\b': 'conduct research',
            r'\bdo\s+(a\s+)?mistake\b': 'make a mistake',
            r'\bdo\s+(an?\s+)?error\b': 'make an error',
            r'\bsay\s+(an?\s+)?opinion\b': 'express an opinion',
            r'\btake\s+(a\s+)?decision\b': 'make a decision',
            r'\bmake\s+(a\s+)?photo\b': 'take a photo',
            r'\bgive\s+(an?\s+)?advice\b': 'give advice (no article)',
            r'\bdo\s+(a\s+)?progress\b': 'make progress',
        }
        
        # Common misspellings database
        self.common_misspellings = {
            'recieve': 'receive', 'occured': 'occurred', 'seperate': 'separate',
            'definately': 'definitely', 'begining': 'beginning', 'enviroment': 'environment',
            'goverment': 'government', 'necessery': 'necessary', 'tomorow': 'tomorrow',
            'wich': 'which', 'thier': 'their', 'freind': 'friend', 'beleive': 'believe',
            'acheive': 'achieve', 'concience': 'conscience', 'existance': 'existence',
            'independant': 'independent', 'maintainance': 'maintenance', 'occurance': 'occurrence',
            'perseverence': 'perseverance', 'priviledge': 'privilege', 'recomend': 'recommend',
            'arguement': 'argument', 'judgement': 'judgment', 'acknowledgement': 'acknowledgment',
            'accomodate': 'accommodate', 'embarass': 'embarrass', 'millenium': 'millennium',
            'mispell': 'misspell', 'noticable': 'noticeable', 'paralel': 'parallel',
            'questionaire': 'questionnaire', 'refered': 'referred', 'succesful': 'successful',
            'tommorow': 'tomorrow', 'untill': 'until', 'wellcome': 'welcome'
        }
        
        logger.info("✅ Vocabulary Layer initialized")
    
    def _init_content_layer(self):
        """Initialize content analysis models"""
        logger.info("Initializing Content Layer...")
        
        # Sentence embeddings
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # Keyword extraction
        self.keyword_extractor = None
        if KEYWORD_EXTRACTION_AVAILABLE:
            try:
                self.keyword_extractor = KeyBERT(model=self.sentence_model)
                self.rake_extractor = Rake()
                logger.info("✅ Keyword extractors initialized")
            except Exception as e:
                logger.warning(f"Keyword extraction initialization failed: {e}")
        
        # Logical connectors that indicate relational understanding
        self.logical_connectors = {
            'contrast': ['however', 'although', 'though', 'nevertheless', 'nonetheless', 
                        'on the other hand', 'in contrast', 'conversely', 'whereas', 'while'],
            'cause_effect': ['because', 'therefore', 'thus', 'consequently', 'as a result',
                           'hence', 'accordingly', 'due to', 'owing to', 'since'],
            'addition': ['furthermore', 'moreover', 'additionally', 'besides', 'also',
                        'in addition', 'equally important', 'likewise'],
            'conclusion': ['in conclusion', 'to sum up', 'in summary', 'overall', 'ultimately',
                          'finally', 'in essence', 'to conclude']
        }
        
        logger.info("✅ Content Layer initialized")
    
    def _init_gpt_layer(self):
        """Initialize GPT client for final verification"""
        logger.info("Initializing GPT Layer...")
        
        # Ensure environment variables are loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = None
        self.use_gpt = False
        
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                self.use_gpt = True
                logger.info("✅ GPT client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GPT client: {e}")
        else:
            logger.warning("No OpenAI API key found - GPT verification disabled")
        
        # API cost tracking
        self.total_api_cost = 0.0
    
    # ==================== LAYER 1: GRAMMAR ====================
    
    def score_grammar(self, text: str) -> Tuple[float, List[GrammarError], str]:
        """
        Layer 1: Grammar scoring with GECToR/LanguageTool
        Returns: (score, list of errors, corrected text)
        """
        logger.info("🔍 Layer 1: Grammar Analysis")
        errors = []
        corrected_text = text
        
        # Try GECToR first
        if self.gector_model and self.gector_tokenizer:
            try:
                corrected_text, errors = self._gector_grammar_check(text)
                logger.info(f"  GECToR found {len(errors)} grammar issues")
            except Exception as e:
                logger.warning(f"  GECToR failed: {e}, falling back to LanguageTool")
                corrected_text, errors = self._languagetool_grammar_check(text)
        
        # Fallback to LanguageTool
        elif self.grammar_tool:
            corrected_text, errors = self._languagetool_grammar_check(text)
            logger.info(f"  LanguageTool found {len(errors)} grammar issues")
        
        # Last resort: regex-based
        else:
            errors = self._regex_grammar_check(text)
            logger.info(f"  Regex checker found {len(errors)} grammar issues")
        
        # Calculate score (harsh: -0.5 per error)
        grammar_score = max(0.0, 2.0 - (len(errors) * 0.5))
        
        logger.info(f"  Grammar Score: {grammar_score}/2.0")
        return round(grammar_score, 1), errors, corrected_text
    
    def _gector_grammar_check(self, text: str) -> Tuple[str, List[GrammarError]]:
        """Use GECToR model for grammar correction"""
        errors = []
        
        # Tokenize and generate correction
        inputs = self.gector_tokenizer(
            "grammar: " + text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )
        
        outputs = self.gector_model.generate(**inputs, max_length=256, num_beams=4)
        corrected_text = self.gector_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Compare to find differences
        import difflib
        diff = difflib.SequenceMatcher(None, text.split(), corrected_text.split())
        
        for tag, i1, i2, j1, j2 in diff.get_opcodes():
            if tag != 'equal':
                original = ' '.join(text.split()[i1:i2])
                corrected = ' '.join(corrected_text.split()[j1:j2])
                errors.append(GrammarError(
                    original=original,
                    corrected=corrected,
                    error_type='grammar',
                    position=i1
                ))
        
        return corrected_text, errors
    
    def _languagetool_grammar_check(self, text: str) -> Tuple[str, List[GrammarError]]:
        """Use LanguageTool for grammar checking"""
        errors = []
        corrected_text = text
        
        matches = self.grammar_tool.check(text)
        
        # Process matches in reverse order to maintain positions
        for match in reversed(matches):
            if match.replacements:
                error = GrammarError(
                    original=text[match.offset:match.offset + match.errorLength],
                    corrected=match.replacements[0] if match.replacements else "",
                    error_type=match.ruleId,
                    position=match.offset
                )
                errors.append(error)
                
                # Apply correction
                corrected_text = (
                    corrected_text[:match.offset] + 
                    match.replacements[0] + 
                    corrected_text[match.offset + match.errorLength:]
                )
        
        return corrected_text, list(reversed(errors))
    
    def _regex_grammar_check(self, text: str) -> List[GrammarError]:
        """Fallback regex-based grammar checking"""
        errors = []
        
        # Check for missing comma after introductory words
        intro_pattern = r'\b(However|Therefore|Furthermore|Moreover|Additionally|Nevertheless|Consequently|Subsequently|Finally|Initially|Meanwhile|Thus|Hence) [a-z]'
        for match in re.finditer(intro_pattern, text):
            errors.append(GrammarError(
                original=match.group(0),
                corrected=match.group(1) + ', ' + match.group(0)[-1],
                error_type='missing_comma_after_intro',
                position=match.start()
            ))
        
        # Check for article errors
        if re.search(r'\ba\s+[aeiouAEIOU]', text):
            errors.append(GrammarError(
                original="a + vowel",
                corrected="an + vowel",
                error_type='article_error',
                position=0
            ))
        
        return errors
    
    # ==================== LAYER 2: VOCABULARY ====================
    
    def score_vocabulary(self, text: str, passage: str) -> Tuple[float, List[VocabularyIssue]]:
        """
        Layer 2: Vocabulary scoring with CEFR analysis
        Returns: (score, list of vocabulary issues)
        """
        logger.info("🔍 Layer 2: Vocabulary Analysis")
        issues = []
        total_deduction = 0.0
        
        words = re.findall(r'\b\w+\b', text.lower())
        passage_words = re.findall(r'\b\w+\b', passage.lower())
        
        # 1. Spelling check
        spelling_issues = self._check_spelling(words)
        issues.extend(spelling_issues)
        total_deduction += sum(issue.severity for issue in spelling_issues)
        logger.info(f"  Found {len(spelling_issues)} spelling errors")
        
        # 2. CEFR level analysis
        cefr_issues = self._analyze_cefr_level(words, passage_words)
        issues.extend(cefr_issues)
        total_deduction += sum(issue.severity for issue in cefr_issues)
        logger.info(f"  CEFR analysis: {len(cefr_issues)} issues")
        
        # 3. Collocation errors
        collocation_issues = self._check_collocations(text)
        issues.extend(collocation_issues)
        total_deduction += sum(issue.severity for issue in collocation_issues)
        logger.info(f"  Found {len(collocation_issues)} collocation errors")
        
        # 4. Informal words
        informal_issues = self._check_formality(words)
        issues.extend(informal_issues)
        total_deduction += sum(issue.severity for issue in informal_issues)
        logger.info(f"  Found {len(informal_issues)} informal words")
        
        # Calculate score
        vocab_score = max(0.0, 2.0 - total_deduction)
        
        logger.info(f"  Vocabulary Score: {vocab_score}/2.0")
        return round(vocab_score, 1), issues
    
    def _check_spelling(self, words: List[str]) -> List[VocabularyIssue]:
        """Check for spelling errors"""
        issues = []
        for word in words:
            if word in self.common_misspellings:
                issues.append(VocabularyIssue(
                    word=word,
                    issue_type='spelling',
                    suggestion=self.common_misspellings[word],
                    severity=0.5  # Harsh for spelling
                ))
        return issues
    
    def _analyze_cefr_level(self, text_words: List[str], passage_words: List[str]) -> List[VocabularyIssue]:
        """Analyze CEFR level mismatch"""
        issues = []
        
        # Count words by CEFR level
        text_level_counts = {level: 0 for level in self.cefr_levels}
        passage_level_counts = {level: 0 for level in self.cefr_levels}
        
        for word in text_words:
            for level, word_list in self.cefr_levels.items():
                if word in word_list:
                    text_level_counts[level] += 1
                    break
        
        for word in passage_words:
            for level, word_list in self.cefr_levels.items():
                if word in word_list:
                    passage_level_counts[level] += 1
                    break
        
        # Calculate percentages
        total_text_words = len(text_words)
        total_passage_words = len(passage_words)
        
        if total_text_words > 0:
            text_basic_ratio = (text_level_counts['A1'] + text_level_counts['A2']) / total_text_words
            passage_advanced_ratio = (passage_level_counts['B2'] + passage_level_counts['C1'] + passage_level_counts['C2']) / total_passage_words if total_passage_words > 0 else 0
            
            # Penalize if text is too basic compared to passage
            if text_basic_ratio > 0.5 and passage_advanced_ratio > 0.3:
                issues.append(VocabularyIssue(
                    word='overall_vocabulary',
                    issue_type='cefr_mismatch',
                    suggestion='Use more advanced vocabulary to match passage sophistication',
                    severity=0.3
                ))
        
        return issues
    
    def _check_collocations(self, text: str) -> List[VocabularyIssue]:
        """Check for collocation errors"""
        issues = []
        for pattern, correction in self.collocation_errors.items():
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(VocabularyIssue(
                    word=pattern,
                    issue_type='collocation',
                    suggestion=correction,
                    severity=0.3
                ))
        return issues
    
    def _check_formality(self, words: List[str]) -> List[VocabularyIssue]:
        """Check for informal words"""
        informal_words = {
            'kids': 'children', 'stuff': 'materials/content', 
            'things': 'aspects/elements', 'guys': 'people',
            'gonna': 'going to', 'wanna': 'want to',
            'kinda': 'somewhat', 'sorta': 'somewhat'
        }
        
        issues = []
        for word in words:
            if word in informal_words:
                issues.append(VocabularyIssue(
                    word=word,
                    issue_type='informal',
                    suggestion=informal_words[word],
                    severity=0.2
                ))
        return issues
    
    # ==================== LAYER 3: CONTENT ====================
    
    def score_content(self, user_summary: str, passage: str, key_points: str = "") -> Tuple[float, List[ContentGap]]:
        """
        Layer 3: Content scoring with embeddings + keyword extraction
        Returns: (score, list of content gaps)
        """
        logger.info("🔍 Layer 3: Content Analysis")
        
        # 1. Semantic similarity
        similarity_score = self._calculate_semantic_similarity(user_summary, passage)
        logger.info(f"  Semantic similarity: {similarity_score:.3f}")
        
        # 2. Keyword coverage
        keyword_gaps = self._analyze_keyword_coverage(user_summary, passage)
        logger.info(f"  Missing {len(keyword_gaps)} key concepts")
        
        # 3. Logical connectors
        connector_score = self._analyze_logical_connectors(user_summary)
        logger.info(f"  Logical connector score: {connector_score:.2f}")
        
        # Calculate final content score
        base_score = similarity_score * 2.0  # Convert to 0-2 scale
        keyword_penalty = len(keyword_gaps) * 0.2  # -0.2 per missing keyword
        connector_bonus = connector_score * 0.2  # Small bonus for good connectors
        
        content_score = max(0.0, min(2.0, base_score - keyword_penalty + connector_bonus))
        
        logger.info(f"  Content Score: {content_score}/2.0")
        return round(content_score, 1), keyword_gaps
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        if not self.sentence_model:
            return 0.5  # Default middle score
        
        emb1 = self.sentence_model.encode(text1, convert_to_tensor=True)
        emb2 = self.sentence_model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        
        return similarity
    
    def _analyze_keyword_coverage(self, user_summary: str, passage: str) -> List[ContentGap]:
        """Extract and compare keywords between passage and summary"""
        gaps = []
        
        if self.keyword_extractor:
            try:
                # Extract keywords from passage
                passage_keywords = self.keyword_extractor.extract_keywords(
                    passage, 
                    keyphrase_ngram_range=(1, 2), 
                    stop_words='english',
                    top_n=10
                )
                
                # Extract keywords from summary
                summary_keywords = self.keyword_extractor.extract_keywords(
                    user_summary,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=10
                )
                
                # Find missing keywords
                passage_keys = set(kw[0] for kw in passage_keywords[:5])  # Top 5 most important
                summary_keys = set(kw[0] for kw in summary_keywords)
                
                missing = passage_keys - summary_keys
                for keyword in missing:
                    gaps.append(ContentGap(
                        missing_keyword=keyword,
                        importance=0.8,  # High importance for top keywords
                        category='main_idea'
                    ))
                    
            except Exception as e:
                logger.warning(f"Keyword extraction failed: {e}")
        
        # Fallback: simple word frequency
        else:
            passage_words = set(w.lower() for w in passage.split() if len(w) > 4)
            summary_words = set(w.lower() for w in user_summary.split() if len(w) > 4)
            missing = list(passage_words - summary_words)[:3]
            
            for word in missing:
                gaps.append(ContentGap(
                    missing_keyword=word,
                    importance=0.5,
                    category='supporting_detail'
                ))
        
        return gaps
    
    def _analyze_logical_connectors(self, text: str) -> float:
        """Analyze use of logical connectors"""
        text_lower = text.lower()
        connector_count = 0
        categories_used = set()
        
        for category, connectors in self.logical_connectors.items():
            for connector in connectors:
                if connector in text_lower:
                    connector_count += 1
                    categories_used.add(category)
        
        # Score based on variety and count
        variety_score = len(categories_used) / len(self.logical_connectors)
        count_score = min(1.0, connector_count / 3)  # Expect at least 3 connectors
        
        return (variety_score + count_score) / 2
    
    # ==================== GPT FINAL VERIFICATION ====================
    
    def gpt_final_verification(self, 
                              passage: str,
                              user_summary: str,
                              grammar_results: Tuple,
                              vocabulary_results: Tuple,
                              content_results: Tuple) -> Dict:
        """
        Final GPT verification layer with structured output
        """
        if not self.use_gpt:
            return {"success": False, "reason": "GPT not available"}
        
        logger.info("🤖 Final Layer: GPT Verification")
        
        # Prepare findings summary
        grammar_score, grammar_errors, corrected_text = grammar_results
        vocab_score, vocab_issues = vocabulary_results
        content_score, content_gaps = content_results
        
        prompt = f"""You are an expert PTE examiner performing FINAL VERIFICATION of automated scoring.

PASSAGE:
{passage}

USER SUMMARY:
{user_summary}

AUTOMATED FINDINGS:
1. GRAMMAR LAYER:
   - Score: {grammar_score}/2.0
   - Errors found: {len(grammar_errors)}
   - Corrected version: {corrected_text}
   - Specific errors: {[f"{e.original}→{e.corrected}" for e in grammar_errors[:3]]}

2. VOCABULARY LAYER:
   - Score: {vocab_score}/2.0
   - Issues found: {len(vocab_issues)}
   - Types: {set(v.issue_type for v in vocab_issues)}
   - Specific issues: {[f"{v.word} ({v.issue_type})" for v in vocab_issues[:3]]}

3. CONTENT LAYER:
   - Score: {content_score}/2.0
   - Missing keywords: {len(content_gaps)}
   - Key gaps: {[g.missing_keyword for g in content_gaps[:3]]}

YOUR TASK:
1. VERIFY all automated findings - are they correct?
2. FIND any errors the system missed
3. PROVIDE final scores with ultra-harsh PTE standards
4. Give specific, actionable feedback

SCORING RULES (PTE OFFICIAL):
- Grammar: 2.0 max, deduct 0.5 per error
- Vocabulary: 2.0 max, deduct 0.3-0.5 per issue
- Content: 2.0 max, based on coverage
- Form: 1.0 for single sentence 5-75 words

Return STRICT JSON:
{{
    "verified_scores": {{
        "grammar": 0.0-2.0,
        "vocabulary": 0.0-2.0,
        "content": 0.0-2.0,
        "form": 0.0-1.0
    }},
    "confirmed_errors": {{
        "grammar": ["error1", "error2"],
        "vocabulary": ["issue1", "issue2"],
        "content": ["missing1", "missing2"]
    }},
    "additional_errors_found": {{
        "grammar": ["new_error1"],
        "vocabulary": ["new_issue1"],
        "content": ["new_gap1"]
    }},
    "final_feedback": {{
        "strengths": ["strength1", "strength2"],
        "critical_improvements": ["improvement1", "improvement2"],
        "harsh_assessment": "Detailed paragraph with specific issues"
    }}
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Track cost
            if hasattr(response, 'usage'):
                self._track_api_cost(response.usage)
            
            # Parse and auto-repair JSON
            result = self._auto_repair_json(response.choices[0].message.content)
            result["success"] = True
            
            logger.info("  ✅ GPT verification complete")
            return result
            
        except Exception as e:
            logger.error(f"GPT verification failed: {e}")
            return {"success": False, "reason": str(e)}
    
    def _auto_repair_json(self, content: str) -> Dict:
        """Auto-repair malformed JSON from GPT"""
        import re
        
        # Clean common issues
        content = content.strip()
        
        # Remove markdown code blocks
        content = re.sub(r'```json?\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        
        # Try to extract JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        # Attempt to parse
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
            
            # Try to fix common issues
            content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
            content = re.sub(r',\s*]', ']', content)
            
            try:
                return json.loads(content)
            except:
                # Return safe default
                return {
                    "verified_scores": {"grammar": 1.0, "vocabulary": 1.0, "content": 1.0, "form": 0.5},
                    "confirmed_errors": {"grammar": [], "vocabulary": [], "content": []},
                    "additional_errors_found": {"grammar": [], "vocabulary": [], "content": []},
                    "final_feedback": {
                        "strengths": ["JSON parsing failed"],
                        "critical_improvements": ["Review automated findings"],
                        "harsh_assessment": "Manual review recommended"
                    }
                }
    
    def _track_api_cost(self, usage_data):
        """Track OpenAI API costs"""
        if not usage_data:
            return
        
        # GPT-4o pricing
        input_cost = (usage_data.prompt_tokens / 1000) * 0.005
        output_cost = (usage_data.completion_tokens / 1000) * 0.015
        total_cost = input_cost + output_cost
        
        self.total_api_cost += total_cost
        logger.info(f"  💰 API Cost: ${total_cost:.4f} (Total: ${self.total_api_cost:.4f})")
    
    # ==================== MAIN SCORING METHOD ====================
    
    def comprehensive_score(self, user_summary: str, passage: str, key_points: str = "") -> Dict:
        """
        Main scoring method - runs all layers and returns final scores
        """
        logger.info("="*50)
        logger.info("🚀 Starting Comprehensive 3-Layer Scoring")
        logger.info("="*50)
        
        try:
            # Layer 1: Grammar
            grammar_results = self.score_grammar(user_summary)
            
            # Layer 2: Vocabulary
            vocabulary_results = self.score_vocabulary(user_summary, passage)
            
            # Layer 3: Content
            content_results = self.score_content(user_summary, passage, key_points)
            
            # Form check (simple)
            form_score, form_feedback = self._score_form(user_summary)
            
            # GPT Final Verification
            gpt_verification = self.gpt_final_verification(
                passage, user_summary,
                grammar_results, vocabulary_results, content_results
            )
            
            # Compile final scores
            if gpt_verification.get("success"):
                # Use GPT-verified scores
                final_scores = gpt_verification["verified_scores"]
                additional_errors = gpt_verification.get("additional_errors_found", {})
                final_feedback = gpt_verification.get("final_feedback", {})
                logger.info("✅ Using GPT-verified scores")
            else:
                # Use automated scores
                logger.warning(f"❌ GPT verification failed: {gpt_verification.get('reason', 'Unknown')}")
                final_scores = {
                    "grammar": grammar_results[0],
                    "vocabulary": vocabulary_results[0],
                    "content": content_results[0],
                    "form": form_score
                }
                additional_errors = {}
                final_feedback = {
                    "strengths": [],
                    "critical_improvements": [],
                    "harsh_assessment": "Automated scoring only - GPT verification unavailable"
                }
            
            # Calculate totals
            total_score = sum(final_scores.values())
            percentage = round((total_score / 7) * 100)
            
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
            
            # Format errors for output
            grammar_errors = [f"{e.original}→{e.corrected}" for e in grammar_results[1]]
            vocabulary_errors = [f"{v.word}: {v.suggestion}" for v in vocabulary_results[1]]
            content_feedback = [f"Missing: {g.missing_keyword}" for g in content_results[1]]
            
            # Add any additional errors found by GPT
            if additional_errors:
                grammar_errors.extend(additional_errors.get("grammar", []))
                vocabulary_errors.extend(additional_errors.get("vocabulary", []))
                content_feedback.extend(additional_errors.get("content", []))
            
            logger.info("="*50)
            logger.info(f"✅ Scoring Complete: {total_score:.1f}/7 ({percentage}%) - {band}")
            logger.info("="*50)
            
            return {
                "success": True,
                "scores": final_scores,
                "total_score": round(total_score, 1),
                "percentage": percentage,
                "band": band,
                "grammar_errors": grammar_errors,
                "vocabulary_errors": vocabulary_errors,
                "content_feedback": content_feedback,
                "form_feedback": [form_feedback],
                "detailed_analysis": {
                    "total_grammar_errors": len(grammar_errors),
                    "total_vocabulary_errors": len(vocabulary_errors),
                    "total_content_gaps": len(content_feedback),
                    "corrected_text": grammar_results[2] if len(grammar_results) > 2 else user_summary,
                    "api_cost": self.total_api_cost
                },
                "strengths": final_feedback.get("strengths", []),
                "improvements": final_feedback.get("critical_improvements", []),
                "harsh_assessment": final_feedback.get("harsh_assessment", ""),
                "feedback": {
                    "grammar": f"Grammar: {final_scores['grammar']}/2.0 - {len(grammar_errors)} errors found",
                    "vocabulary": f"Vocabulary: {final_scores['vocabulary']}/2.0 - {len(vocabulary_errors)} issues",
                    "content": f"Content: {final_scores['content']}/2.0 - {len(content_feedback)} gaps",
                    "form": f"Form: {final_scores['form']}/1.0 - {form_feedback}"
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive scoring failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "scores": {"grammar": 0, "vocabulary": 0, "content": 0, "form": 0},
                "total_score": 0,
                "percentage": 0,
                "band": "Error"
            }
    
    def _score_form(self, text: str) -> Tuple[float, str]:
        """Simple form scoring"""
        words = text.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', text.strip())
        sentence_count = len([s for s in sentences if s.strip()])
        
        if 5 <= word_count <= 75 and sentence_count == 1:
            return 1.0, f"Perfect form: {word_count} words, single sentence"
        elif word_count < 5:
            return 0.0, f"Too short: {word_count} words (min 5)"
        elif word_count > 75:
            return 0.0, f"Too long: {word_count} words (max 75)"
        elif sentence_count != 1:
            return 0.0, f"Must be single sentence (found {sentence_count})"
        else:
            return 0.5, f"Form issues detected"
    
    # ==================== CALIBRATION ====================
    
    def calibrate(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calibrate scorer against known Pearson scores
        Input: [{passage, user_summary, pearson_score}, ...]
        Output: {correlation, rmse, bias}
        """
        logger.info("🎯 Starting Calibration")
        
        our_scores = []
        pearson_scores = []
        
        for sample in dataset:
            try:
                result = self.comprehensive_score(
                    user_summary=sample['user_summary'],
                    passage=sample['passage']
                )
                our_scores.append(result['total_score'])
                pearson_scores.append(sample['pearson_score'])
            except Exception as e:
                logger.warning(f"Calibration sample failed: {e}")
                continue
        
        if len(our_scores) < 2:
            return {"error": "Insufficient calibration data"}
        
        # Calculate correlation
        correlation, p_value = pearsonr(our_scores, pearson_scores)
        
        # Calculate RMSE
        our_scores_np = np.array(our_scores)
        pearson_scores_np = np.array(pearson_scores)
        rmse = np.sqrt(np.mean((our_scores_np - pearson_scores_np) ** 2))
        
        # Calculate bias
        bias = np.mean(our_scores_np - pearson_scores_np)
        
        logger.info(f"Calibration Results: Correlation={correlation:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}")
        
        return {
            "correlation": correlation,
            "p_value": p_value,
            "rmse": rmse,
            "bias": bias,
            "n_samples": len(our_scores)
        }


# Create global instance
enhanced_hybrid_scorer = HybridScorer()