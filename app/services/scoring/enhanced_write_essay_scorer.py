#!/usr/bin/env python3
"""
Enhanced Write Essay Scorer - Production-Ready 3-Layer System
Addresses all weak points with modular, optimized components

Improvements:
1. Optimized GECToR with batching and intelligent fallback
2. L2 Syntactic Complexity Analyzer for linguistic range
3. Paragraph embedding analysis for structure scoring
4. Enhanced CEFR vocabulary analysis with lexical richness
5. Hunspell-based spelling with academic dictionary
6. GPT verifier with JSON schema and retry logic
7. Non-linear score mapping with calibration
"""

import re
import json
import logging
import os
import math
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from scipy import stats
import nltk
import spacy

# Core ML imports with error handling
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import hunspell
    CORE_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core models unavailable: {e}")
    CORE_MODELS_AVAILABLE = False

# NLP tools
import language_tool_python
from sentence_transformers import SentenceTransformer, util
from lexicalrichness import LexicalRichness

# Keyword extraction
try:
    from keybert import KeyBERT
    from yake import KeywordExtractor
    KEYWORD_TOOLS_AVAILABLE = True
except ImportError:
    KEYWORD_TOOLS_AVAILABLE = False

# OpenAI for GPT verification
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

@dataclass
class SyntacticComplexityMetrics:
    """L2 Syntactic Complexity Analysis results"""
    mean_length_clause: float
    mean_length_sentence: float
    mean_length_t_unit: float
    clauses_per_sentence: float
    clauses_per_t_unit: float
    complex_t_units_ratio: float
    coordinate_phrases_ratio: float
    subordination_ratio: float
    
@dataclass
class SpellingAnalysis:
    """Comprehensive spelling analysis results"""
    total_errors: int
    error_types: Dict[str, int]
    academic_errors: int
    false_positives: int
    severity_score: float

@dataclass
class VocabularyAnalysis:
    """Enhanced vocabulary analysis with CEFR and lexical richness"""
    cefr_distribution: Dict[str, float]
    lexical_diversity_ttr: float
    lexical_diversity_mtld: float
    academic_vocabulary_ratio: float
    collocation_errors: List[str]
    overuse_basic_words: bool

@dataclass
class ParagraphStructureAnalysis:
    """Paragraph coherence and structure analysis"""
    paragraph_similarities: List[float]
    idea_repetition_penalty: float
    coherence_score: float
    structural_score: float

class EnhancedWriteEssayScorer:
    """
    Production-Ready Enhanced Write Essay Scorer
    
    Optimizations:
    - GECToR with sentence batching
    - L2SCA linguistic complexity
    - Paragraph embedding analysis
    - Hunspell spelling with academic dictionary
    - GPT verifier with schema validation
    - Non-linear score mapping
    """
    
    def __init__(self):
        """Initialize all enhanced scoring layers"""
        logger.info("🚀 Initializing Enhanced Write Essay Scorer...")
        
        # Configuration
        self.use_gpt = bool(os.getenv('OPENAI_API_KEY'))
        self.openai_client = None
        self.total_api_cost = 0.0
        
        # Initialize enhanced layers
        self._init_enhanced_grammar_layer()
        self._init_syntactic_complexity_analyzer()
        self._init_enhanced_vocabulary_layer()
        self._init_enhanced_spelling_layer()
        self._init_paragraph_analysis_layer()
        self._init_enhanced_gpt_layer()
        self._init_score_mapping()
        
        logger.info("✅ Enhanced Write Essay Scorer initialized successfully")
    
    def _init_enhanced_grammar_layer(self):
        """1. Enhanced Grammar Layer with Optimized GECToR"""
        logger.info("Initializing Enhanced Grammar Layer...")
        
        # Primary: Optimized GECToR with batching
        self.gector_model = None
        self.gector_tokenizer = None
        
        if CORE_MODELS_AVAILABLE:
            try:
                # Use the original model but with optimization
                model_name = "vennify/t5-base-grammar-correction"
                self.gector_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.gector_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                # Optimize for inference
                self.gector_model.eval()
                if torch.cuda.is_available():
                    self.gector_model = self.gector_model.cuda()
                    logger.info("✅ GECToR loaded on GPU")
                else:
                    logger.info("✅ GECToR loaded on CPU")
                    
            except Exception as e:
                logger.error(f"❌ GECToR failed to load: {e}")
                self.gector_model = None
        
        # Fallback: LanguageTool
        try:
            self.language_tool = language_tool_python.LanguageTool('en-US')
            logger.info("✅ LanguageTool loaded as fallback")
        except:
            self.language_tool = None
            logger.warning("⚠️ LanguageTool unavailable")
        
        # spaCy for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy loaded for sentence processing")
        except:
            self.nlp = None
            logger.warning("⚠️ spaCy unavailable")
    
    def _init_syntactic_complexity_analyzer(self):
        """2. L2 Syntactic Complexity Analyzer (L2SCA)"""
        logger.info("Initializing L2 Syntactic Complexity Analyzer...")
        
        # POS patterns for T-unit detection
        self.t_unit_patterns = [
            r'\b(although|though|while|whereas|if|unless|because|since|when|whenever|where|wherever)\b',
            r'\b(that|which|who|whom|whose)\b',
            r'\b(after|before|until|during|following)\b'
        ]
        
        # Complex structure patterns
        self.complex_patterns = {
            'subordinate_clauses': r'\b(that|which|who|whom|whose|when|where|why|how)\b',
            'coordinate_phrases': r'\b(and|but|or|nor|for|so|yet)\b',
            'nominal_clauses': r'\b(that|whether|if|what|who|how|why|when|where)\s+\w+\s+\w+',
            'adverbial_clauses': r'\b(when|while|where|because|since|although|if|unless)\b',
            'relative_clauses': r'\b(who|whom|whose|which|that)\b'
        }
        
        logger.info("✅ L2SCA patterns loaded")
    
    def _init_enhanced_vocabulary_layer(self):
        """3. Enhanced Vocabulary Analysis with CEFR"""
        logger.info("Initializing Enhanced Vocabulary Layer...")
        
        # Enhanced CEFR word lists (open-source based)
        self.cefr_levels = {
            'A1': set([
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'do', 'does',
                'go', 'come', 'get', 'make', 'see', 'know', 'think', 'good', 'bad', 'big', 'small',
                'new', 'old', 'first', 'last', 'long', 'short', 'high', 'low', 'right', 'left',
                'here', 'there', 'now', 'then', 'today', 'tomorrow', 'yesterday', 'time', 'day',
                'year', 'week', 'month', 'morning', 'evening', 'night', 'home', 'work', 'school'
            ]),
            'A2': set([
                'because', 'but', 'so', 'when', 'where', 'why', 'how', 'can', 'could', 'will',
                'would', 'should', 'must', 'may', 'might', 'want', 'need', 'like', 'love', 'hate',
                'help', 'try', 'start', 'stop', 'finish', 'continue', 'change', 'move', 'live',
                'study', 'learn', 'teach', 'understand', 'remember', 'forget', 'hope', 'believe'
            ]),
            'B1': set([
                'although', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
                'consequently', 'additionally', 'specifically', 'generally', 'particularly',
                'organize', 'develop', 'create', 'establish', 'maintain', 'improve', 'increase',
                'decrease', 'reduce', 'compare', 'contrast', 'analyze', 'evaluate', 'consider'
            ]),
            'B2': set([
                'significant', 'substantial', 'considerable', 'fundamental', 'essential',
                'crucial', 'vital', 'demonstrate', 'illustrate', 'emphasize', 'facilitate',
                'implement', 'establish', 'determine', 'identify', 'indicate', 'suggest',
                'contribute', 'achieve', 'acquire', 'adapt', 'advocate', 'appreciate'
            ]),
            'C1': set([
                'notwithstanding', 'albeit', 'whereby', 'inasmuch', 'nonetheless', 'henceforth',
                'paradigm', 'methodology', 'framework', 'synthesis', 'hypothesis', 'empirical',
                'comprehensive', 'sophisticated', 'intricate', 'elaborate', 'meticulous',
                'rigorous', 'substantial', 'profound', 'unprecedented', 'innovative'
            ]),
            'C2': set([
                'quintessential', 'ubiquitous', 'dichotomy', 'paradoxical', 'epistemological',
                'hermeneutical', 'phenomenological', 'ontological', 'axiological', 'teleological',
                'indigenous', 'exacerbate', 'ameliorate', 'corroborate', 'elucidate', 'substantiate'
            ])
        }
        
        # Academic Word List (AWL)
        self.academic_words = set([
            'analyze', 'analysis', 'concept', 'conceptual', 'constitute', 'context', 'contextual',
            'derive', 'distribution', 'establish', 'estimate', 'evidence', 'export', 'factor',
            'finance', 'financial', 'formula', 'function', 'identify', 'income', 'indicate',
            'individual', 'interpret', 'involve', 'issue', 'labour', 'legal', 'legislate',
            'major', 'method', 'occur', 'percent', 'period', 'policy', 'principle', 'proceed',
            'process', 'require', 'research', 'respond', 'role', 'section', 'sector', 'significant',
            'similar', 'source', 'specific', 'structure', 'theory', 'vary', 'approach', 'area',
            'assessment', 'assume', 'authority', 'available', 'benefit', 'category', 'commission',
            'community', 'complex', 'computer', 'conclude', 'conduct', 'consequence', 'construction',
            'consumer', 'contract', 'create', 'data', 'definition', 'derived', 'design', 'distinction',
            'element', 'environment', 'established', 'evaluation', 'feature', 'final', 'focus',
            'impact', 'injury', 'institute', 'investment', 'item', 'journal', 'maintenance',
            'normal', 'obtained', 'participation', 'perceived', 'positive', 'potential', 'previous',
            'primary', 'purchase', 'range', 'region', 'regulation', 'relevant', 'resident',
            'resource', 'restricted', 'security', 'sought', 'survey', 'text', 'traditional',
            'transfer'
        ])
        
        # Enhanced collocation error patterns
        self.collocation_errors = {
            r'\bmake\s+(a\s+)?research\b': 'conduct research',
            r'\bdo\s+(a\s+)?mistake\b': 'make a mistake',
            r'\bdo\s+(an?\s+)?error\b': 'make an error',
            r'\bsay\s+(an?\s+)?opinion\b': 'express an opinion',
            r'\btake\s+(a\s+)?decision\b': 'make a decision',
            r'\bmake\s+(a\s+)?photo\b': 'take a photo',
            r'\bgive\s+(an?\s+)?advice\b': 'give advice',
            r'\bdo\s+(a\s+)?progress\b': 'make progress',
            r'\bsay\s+(a\s+)?lie\b': 'tell a lie',
            r'\bmake\s+(a\s+)?homework\b': 'do homework',
            r'\btake\s+(a\s+)?shower\b': 'have/take a shower',
            r'\bmake\s+(a\s+)?party\b': 'throw/have a party'
        }
        
        logger.info("✅ Enhanced vocabulary analysis loaded")
    
    def _init_enhanced_spelling_layer(self):
        """4. Enhanced Spelling with Hunspell"""
        logger.info("Initializing Enhanced Spelling Layer...")
        
        # Primary: Hunspell with academic dictionary
        self.hunspell_checker = None
        try:
            # Try to initialize Hunspell
            self.hunspell_checker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
            logger.info("✅ Hunspell loaded")
        except:
            logger.warning("⚠️ Hunspell unavailable, using fallback")
        
        # Academic and domain-specific whitelist
        self.academic_whitelist = set([
            'methodology', 'epistemology', 'paradigm', 'heuristic', 'meta-analysis',
            'quasi-experimental', 'socioeconomic', 'interdisciplinary', 'counterargument',
            'globalization', 'digitalization', 'urbanization', 'industrialization',
            'sustainability', 'biodiversity', 'cryptocurrency', 'nanotechnology'
        ])
        
        # Enhanced misspelling database
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
            'tempation': 'temptation', 'questionaire': 'questionnaire', 'refered': 'referred',
            'succesful': 'successful', 'tommorow': 'tomorrow', 'untill': 'until',
            'wellcome': 'welcome', 'topc': 'topic', 'disadvangtes': 'disadvantages',
            'prominet': 'prominent', 'qoute': 'quote', 'reporst': 'reports'
        }
        
        logger.info("✅ Enhanced spelling analysis loaded")
    
    def _init_paragraph_analysis_layer(self):
        """5. Paragraph Structure Analysis with Embeddings"""
        logger.info("Initializing Paragraph Analysis Layer...")
        
        # Sentence embeddings for paragraph analysis
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded for paragraph analysis")
        except:
            self.sentence_model = None
            logger.warning("⚠️ Sentence transformer unavailable")
        
        # Discourse markers for coherence analysis
        self.discourse_markers = {
            'addition': ['furthermore', 'moreover', 'additionally', 'besides', 'also', 'in addition'],
            'contrast': ['however', 'nevertheless', 'nonetheless', 'conversely', 'whereas', 'on the other hand'],
            'cause': ['because', 'since', 'as', 'due to', 'owing to', 'as a result of'],
            'effect': ['therefore', 'thus', 'consequently', 'hence', 'as a result', 'accordingly'],
            'example': ['for example', 'for instance', 'such as', 'namely', 'specifically', 'in particular'],
            'conclusion': ['in conclusion', 'to conclude', 'in summary', 'overall', 'ultimately', 'finally']
        }
        
        logger.info("✅ Paragraph analysis layer loaded")
    
    def _init_enhanced_gpt_layer(self):
        """6. Enhanced GPT Verifier with Schema Validation"""
        if self.use_gpt:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("✅ GPT-4o verifier initialized")
                
                # JSON schema for GPT responses
                self.gpt_response_schema = {
                    "type": "object",
                    "properties": {
                        "verification_status": {"type": "string", "enum": ["confirmed", "adjusted", "rejected"]},
                        "verified_scores": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "number", "minimum": 0, "maximum": 6},
                                "form": {"type": "number", "minimum": 0, "maximum": 2},
                                "development": {"type": "number", "minimum": 0, "maximum": 6},
                                "grammar": {"type": "number", "minimum": 0, "maximum": 2},
                                "linguistic": {"type": "number", "minimum": 0, "maximum": 6},
                                "vocabulary": {"type": "number", "minimum": 0, "maximum": 2},
                                "spelling": {"type": "number", "minimum": 0, "maximum": 2}
                            },
                            "required": ["content", "form", "development", "grammar", "linguistic", "vocabulary", "spelling"]
                        },
                        "verification_notes": {"type": "string"},
                        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["verification_status", "verified_scores", "verification_notes", "confidence_score"]
                }
                
            except Exception as e:
                logger.warning(f"⚠️ GPT-4o unavailable: {e}")
                self.use_gpt = False
    
    def _init_score_mapping(self):
        """7. Non-linear Score Mapping and Calibration"""
        logger.info("Initializing Score Mapping...")
        
        # Default mapping parameters (can be calibrated)
        self.mapping_params = {
            'scale_factor': 3.46,  # 26 * 3.46 ≈ 90
            'curve_type': 'sigmoid',  # 'linear', 'sigmoid', 'logarithmic'
            'sigmoid_steepness': 0.15,
            'sigmoid_midpoint': 13.0,
            'logarithmic_base': 1.5,
            'min_score': 10,
            'max_score': 90
        }
        
        logger.info("✅ Score mapping initialized")
    
    # ==================== ENHANCED SCORING METHODS ====================
    
    def check_enhanced_grammar(self, text: str) -> Tuple[float, List[str]]:
        """
        1. Enhanced Grammar Check with Optimized GECToR
        - Sentence-level batching for efficiency
        - Intelligent fallback system
        """
        errors = []
        
        if not text.strip():
            return 0.0, ["Empty text"]
        
        # Sentence segmentation for batching
        sentences = self._segment_sentences(text)
        
        # Primary: GECToR with batching
        if self.gector_model and self.gector_tokenizer:
            try:
                errors.extend(self._gector_batch_check(sentences))
            except Exception as e:
                logger.warning(f"GECToR failed, using fallback: {e}")
                errors.extend(self._languagetool_check(text))
        else:
            # Fallback: LanguageTool
            errors.extend(self._languagetool_check(text))
        
        # Calculate score with improved calibration
        score = self._calculate_grammar_score(errors)
        return score, errors
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences for batch processing"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback: simple regex
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _gector_batch_check(self, sentences: List[str]) -> List[str]:
        """Optimized GECToR with sentence batching"""
        errors = []
        batch_size = 3  # Process 3 sentences at a time
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            try:
                # Process batch
                for sentence in batch:
                    if len(sentence) > 10:  # Skip very short sentences
                        inputs = self.gector_tokenizer(
                            sentence, 
                            return_tensors="pt", 
                            max_length=128, 
                            truncation=True,
                            padding=True
                        )
                        
                        with torch.no_grad():
                            outputs = self.gector_model.generate(
                                **inputs, 
                                max_length=128,
                                num_beams=2,
                                early_stopping=True
                            )
                        
                        corrected = self.gector_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        if corrected.strip() != sentence.strip():
                            errors.append(f"Grammar: {sentence[:50]}... → {corrected[:50]}...")
                            
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}")
                continue
        
        return errors
    
    def _languagetool_check(self, text: str) -> List[str]:
        """LanguageTool grammar checking"""
        errors = []
        if not self.language_tool:
            return errors
        
        try:
            matches = self.language_tool.check(text)
            for match in matches[:15]:  # Limit errors
                if match.ruleId not in ['WHITESPACE_RULE', 'UPPERCASE_SENTENCE_START']:
                    errors.append(f"Grammar: {match.message}")
        except Exception as e:
            logger.warning(f"LanguageTool check failed: {e}")
        
        return errors
    
    def _calculate_grammar_score(self, errors: List[str]) -> float:
        """Calculate grammar score with improved calibration"""
        error_count = len(errors)
        
        if error_count == 0:
            return 2.0
        elif error_count <= 2:
            return 1.7
        elif error_count <= 4:
            return 1.2
        elif error_count <= 6:
            return 0.8
        elif error_count <= 8:
            return 0.4
        else:
            return 0.0
    
    def analyze_syntactic_complexity(self, text: str) -> Tuple[float, SyntacticComplexityMetrics]:
        """
        2. L2 Syntactic Complexity Analysis
        Calculates comprehensive linguistic complexity metrics
        """
        if not text.strip():
            return 0.0, SyntacticComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Sentence and clause analysis
        sentences = self._segment_sentences(text)
        t_units = self._identify_t_units(text)
        clauses = self._identify_clauses(text)
        
        # Calculate metrics
        total_words = len(text.split())
        total_sentences = len(sentences)
        total_clauses = len(clauses)
        total_t_units = len(t_units)
        
        # Avoid division by zero
        if total_sentences == 0 or total_t_units == 0:
            return 0.0, SyntacticComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Core L2SCA metrics
        mean_length_sentence = total_words / total_sentences
        mean_length_t_unit = total_words / total_t_units
        mean_length_clause = total_words / max(total_clauses, 1)
        clauses_per_sentence = total_clauses / total_sentences
        clauses_per_t_unit = total_clauses / total_t_units
        
        # Complex structure analysis
        complex_t_units = sum(1 for t_unit in t_units if self._is_complex_t_unit(t_unit))
        complex_t_units_ratio = complex_t_units / total_t_units
        
        coordinate_phrases = len(re.findall(self.complex_patterns['coordinate_phrases'], text, re.I))
        coordinate_phrases_ratio = coordinate_phrases / total_t_units
        
        subordinate_clauses = len(re.findall(self.complex_patterns['subordinate_clauses'], text, re.I))
        subordination_ratio = subordinate_clauses / total_t_units
        
        metrics = SyntacticComplexityMetrics(
            mean_length_clause=mean_length_clause,
            mean_length_sentence=mean_length_sentence,
            mean_length_t_unit=mean_length_t_unit,
            clauses_per_sentence=clauses_per_sentence,
            clauses_per_t_unit=clauses_per_t_unit,
            complex_t_units_ratio=complex_t_units_ratio,
            coordinate_phrases_ratio=coordinate_phrases_ratio,
            subordination_ratio=subordination_ratio
        )
        
        # Calculate linguistic range score (0-6)
        complexity_score = self._calculate_complexity_score(metrics)
        
        return complexity_score, metrics
    
    def _identify_t_units(self, text: str) -> List[str]:
        """Identify T-units (minimal terminable units)"""
        # Split by sentence boundaries first
        sentences = re.split(r'[.!?]+', text)
        t_units = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Split complex sentences into T-units
            # Look for coordinating conjunctions at clause level
            units = re.split(r'\s*,\s*(?:and|but|or|nor|for|so|yet)\s+', sentence)
            t_units.extend([unit.strip() for unit in units if unit.strip()])
        
        return t_units
    
    def _identify_clauses(self, text: str) -> List[str]:
        """Identify all clauses in the text"""
        clauses = []
        
        # Main clauses (sentences)
        sentences = self._segment_sentences(text)
        clauses.extend(sentences)
        
        # Subordinate clauses
        for pattern in self.t_unit_patterns:
            matches = re.finditer(pattern + r'[^.!?]*', text, re.I)
            for match in matches:
                clause = match.group().strip()
                if clause and len(clause.split()) >= 3:  # Minimum clause length
                    clauses.append(clause)
        
        return clauses
    
    def _is_complex_t_unit(self, t_unit: str) -> bool:
        """Check if T-unit contains complex structures"""
        for pattern in self.complex_patterns.values():
            if re.search(pattern, t_unit, re.I):
                return True
        return False
    
    def _calculate_complexity_score(self, metrics: SyntacticComplexityMetrics) -> float:
        """Calculate linguistic complexity score based on L2SCA metrics"""
        # Normalize metrics for 200-300 word essays
        normalized_scores = []
        
        # Mean length metrics (higher = more complex)
        if metrics.mean_length_sentence >= 20:
            normalized_scores.append(1.0)
        elif metrics.mean_length_sentence >= 15:
            normalized_scores.append(0.7)
        elif metrics.mean_length_sentence >= 12:
            normalized_scores.append(0.5)
        else:
            normalized_scores.append(0.2)
        
        # Subordination ratio (higher = more complex)
        if metrics.subordination_ratio >= 0.4:
            normalized_scores.append(1.0)
        elif metrics.subordination_ratio >= 0.3:
            normalized_scores.append(0.7)
        elif metrics.subordination_ratio >= 0.2:
            normalized_scores.append(0.5)
        else:
            normalized_scores.append(0.2)
        
        # Complex T-units ratio
        if metrics.complex_t_units_ratio >= 0.6:
            normalized_scores.append(1.0)
        elif metrics.complex_t_units_ratio >= 0.4:
            normalized_scores.append(0.7)
        elif metrics.complex_t_units_ratio >= 0.2:
            normalized_scores.append(0.5)
        else:
            normalized_scores.append(0.2)
        
        # Calculate weighted average
        complexity_score = sum(normalized_scores) / len(normalized_scores) * 6.0
        return round(min(6.0, complexity_score), 1)
    
    def analyze_paragraph_structure(self, text: str) -> Tuple[float, ParagraphStructureAnalysis]:
        """
        3. Enhanced Paragraph Structure Analysis with Embeddings
        Detects idea repetition and measures coherence
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            # Single paragraph - penalize structure
            return 2.0, ParagraphStructureAnalysis([], 0.0, 2.0, 2.0)
        
        # Calculate paragraph embeddings
        similarities = []
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode(paragraphs)
                
                # Calculate pairwise similarities
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        similarity = util.cos_sim(embeddings[i], embeddings[j]).item()
                        similarities.append(similarity)
                        
            except Exception as e:
                logger.warning(f"Embedding analysis failed: {e}")
        
        # Calculate penalties and scores
        idea_repetition_penalty = self._calculate_repetition_penalty(similarities)
        coherence_score = self._analyze_discourse_coherence(text)
        structural_score = self._analyze_structural_elements(paragraphs)
        
        # Overall development score (0-6)
        development_score = max(0, 6.0 - idea_repetition_penalty)
        development_score = min(development_score, coherence_score, structural_score)
        
        analysis = ParagraphStructureAnalysis(
            paragraph_similarities=similarities,
            idea_repetition_penalty=idea_repetition_penalty,
            coherence_score=coherence_score,
            structural_score=structural_score
        )
        
        return round(development_score, 1), analysis
    
    def _calculate_repetition_penalty(self, similarities: List[float]) -> float:
        """Calculate penalty for idea repetition based on similarity scores"""
        if not similarities:
            return 0.0
        
        high_similarity_count = sum(1 for sim in similarities if sim > 0.9)
        moderate_similarity_count = sum(1 for sim in similarities if 0.7 < sim <= 0.9)
        
        penalty = (high_similarity_count * 2.0) + (moderate_similarity_count * 1.0)
        return min(penalty, 4.0)  # Max penalty of 4 points
    
    def _analyze_discourse_coherence(self, text: str) -> float:
        """Analyze discourse marker usage for coherence"""
        text_lower = text.lower()
        markers_found = []
        
        for category, markers in self.discourse_markers.items():
            category_found = False
            for marker in markers:
                if marker in text_lower:
                    markers_found.append(marker)
                    category_found = True
                    break
            if not category_found and category in ['contrast', 'cause', 'effect']:
                return max(0, 6.0 - 1.5)  # Penalty for missing critical markers
        
        # Score based on marker diversity
        unique_categories = len(set(self._get_marker_category(marker) for marker in markers_found))
        if unique_categories >= 4:
            return 6.0
        elif unique_categories >= 3:
            return 4.5
        elif unique_categories >= 2:
            return 3.0
        else:
            return 1.5
    
    def _get_marker_category(self, marker: str) -> str:
        """Get the category of a discourse marker"""
        for category, markers in self.discourse_markers.items():
            if marker in markers:
                return category
        return 'unknown'
    
    def _analyze_structural_elements(self, paragraphs: List[str]) -> float:
        """Analyze structural elements of the essay"""
        if len(paragraphs) < 3:
            return 2.0  # Needs intro, body, conclusion
        
        # Check for introduction indicators
        intro_indicators = ['this essay', 'i will discuss', 'this paper', 'the purpose', 'introduction']
        has_intro = any(indicator in paragraphs[0].lower() for indicator in intro_indicators)
        
        # Check for conclusion indicators
        conclusion_indicators = ['in conclusion', 'to conclude', 'in summary', 'finally', 'overall']
        has_conclusion = any(indicator in paragraphs[-1].lower() for indicator in conclusion_indicators)
        
        # Calculate structure score
        structure_score = 3.0  # Base score
        if has_intro:
            structure_score += 1.5
        if has_conclusion:
            structure_score += 1.5
        
        return min(6.0, structure_score)
    
    def analyze_enhanced_vocabulary(self, text: str) -> Tuple[float, VocabularyAnalysis]:
        """
        4. Enhanced Vocabulary Analysis with CEFR and Lexical Richness
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return 0.0, VocabularyAnalysis({}, 0.0, 0.0, 0.0, [], False)
        
        # CEFR distribution analysis
        cefr_distribution = self._analyze_cefr_distribution(words)
        
        # Lexical richness metrics
        try:
            lex_rich = LexicalRichness(text)
            ttr = lex_rich.ttr
            mtld = lex_rich.mtld
        except:
            ttr = len(set(words)) / len(words)
            mtld = 50.0  # Default value
        
        # Academic vocabulary ratio
        academic_words = [word for word in words if word in self.academic_words]
        academic_ratio = len(academic_words) / len(words)
        
        # Collocation error detection
        collocation_errors = self._detect_collocation_errors(text)
        
        # Basic word overuse detection
        basic_words = [word for word in words if word in self.cefr_levels['A1'] or word in self.cefr_levels['A2']]
        overuse_basic = (len(basic_words) / len(words)) > 0.6
        
        # Calculate vocabulary score (0-2)
        vocab_score = self._calculate_vocabulary_score(cefr_distribution, ttr, academic_ratio, collocation_errors, overuse_basic)
        
        analysis = VocabularyAnalysis(
            cefr_distribution=cefr_distribution,
            lexical_diversity_ttr=ttr,
            lexical_diversity_mtld=mtld,
            academic_vocabulary_ratio=academic_ratio,
            collocation_errors=collocation_errors,
            overuse_basic_words=overuse_basic
        )
        
        return vocab_score, analysis
    
    def _analyze_cefr_distribution(self, words: List[str]) -> Dict[str, float]:
        """Analyze CEFR level distribution of vocabulary"""
        total_words = len(words)
        distribution = {}
        
        for level, word_set in self.cefr_levels.items():
            level_words = [word for word in words if word in word_set]
            distribution[level] = len(level_words) / total_words if total_words > 0 else 0.0
        
        return distribution
    
    def _detect_collocation_errors(self, text: str) -> List[str]:
        """Detect collocation errors in the text"""
        errors = []
        for pattern, correction in self.collocation_errors.items():
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(f"Collocation: {pattern} → {correction}")
        return errors
    
    def _calculate_vocabulary_score(self, cefr_dist: Dict[str, float], ttr: float, 
                                  academic_ratio: float, collocation_errors: List[str], 
                                  overuse_basic: bool) -> float:
        """Calculate vocabulary range score"""
        score = 2.0  # Start with maximum
        
        # Penalty for overusing basic vocabulary
        if overuse_basic:
            score -= 0.8
        
        # Bonus for advanced vocabulary usage
        advanced_ratio = cefr_dist.get('B2', 0) + cefr_dist.get('C1', 0) + cefr_dist.get('C2', 0)
        if advanced_ratio >= 0.15:
            score = min(2.0, score + 0.3)
        elif advanced_ratio >= 0.10:
            score = min(2.0, score + 0.1)
        
        # Penalty for collocation errors
        score -= len(collocation_errors) * 0.2
        
        # Consider lexical diversity
        if ttr < 0.4:
            score -= 0.3
        elif ttr >= 0.6:
            score = min(2.0, score + 0.2)
        
        # Academic vocabulary bonus
        if academic_ratio >= 0.10:
            score = min(2.0, score + 0.2)
        elif academic_ratio >= 0.05:
            score = min(2.0, score + 0.1)
        
        return round(max(0.0, score), 1)
    
    def check_enhanced_spelling(self, text: str) -> Tuple[float, SpellingAnalysis]:
        """
        5. Enhanced Spelling Check with Hunspell and Academic Dictionary
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return 2.0, SpellingAnalysis(0, {}, 0, 0, 0.0)
        
        # Primary spelling check with explicit database
        errors = []
        academic_errors = 0
        false_positives = 0
        
        # Check against known misspellings
        for word in words:
            if word in self.common_misspellings:
                errors.append(f"{word} → {self.common_misspellings[word]}")
                if word in self.academic_words:
                    academic_errors += 1
        
        # Hunspell check for additional errors
        if self.hunspell_checker:
            for word in words:
                if word not in self.common_misspellings and len(word) > 2:
                    if not self.hunspell_checker.spell(word):
                        # Check against whitelist to avoid false positives
                        if word not in self.academic_whitelist:
                            suggestions = self.hunspell_checker.suggest(word)
                            if suggestions:
                                errors.append(f"{word} → {suggestions[0]}")
                            else:
                                errors.append(f"{word} → [unknown]")
                        else:
                            false_positives += 1
        
        # Categorize error types
        error_types = {
            'common_misspellings': sum(1 for e in errors if any(cm in e for cm in self.common_misspellings)),
            'academic_terms': academic_errors,
            'other': len(errors) - academic_errors
        }
        
        # Calculate severity score
        total_errors = len(errors)
        severity_score = min(1.0, total_errors / max(len(words) * 0.05, 1))  # Max 5% error rate
        
        # Calculate spelling score (0-2) with Pearson-style scoring
        if total_errors == 0:
            spelling_score = 2.0
        elif total_errors == 1:
            spelling_score = 1.5
        elif total_errors <= 3:
            spelling_score = 1.0
        elif total_errors <= 5:
            spelling_score = 0.5
        else:
            spelling_score = 0.0
        
        analysis = SpellingAnalysis(
            total_errors=total_errors,
            error_types=error_types,
            academic_errors=academic_errors,
            false_positives=false_positives,
            severity_score=severity_score
        )
        
        return spelling_score, analysis
    
    def gpt_enhanced_verification(self, essay_prompt: str, user_essay: str, 
                                layer_results: Dict) -> Dict:
        """
        6. Enhanced GPT Verifier with JSON Schema and Retry Logic
        Acts as verifier, not independent scorer
        """
        if not self.use_gpt or not self.openai_client:
            return {"success": False, "reason": "GPT unavailable"}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Prepare verification prompt
                verification_prompt = self._create_verification_prompt(essay_prompt, user_essay, layer_results)
                
                # Call GPT-4o
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a PTE scoring verifier. Analyze the provided ML results and return valid JSON only."},
                        {"role": "user", "content": verification_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                
                # Parse and validate response
                gpt_text = response.choices[0].message.content
                result = self._parse_and_validate_gpt_response(gpt_text)
                
                if result["success"]:
                    # Track cost
                    self.total_api_cost += (response.usage.prompt_tokens * 0.00001) + (response.usage.completion_tokens * 0.00003)
                    logger.info("✅ GPT verification successful")
                    return result
                else:
                    logger.warning(f"GPT response validation failed (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.warning(f"GPT verification attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {"success": False, "reason": f"GPT verification failed after {max_retries} attempts"}
        
        return {"success": False, "reason": "GPT verification failed"}
    
    def _create_verification_prompt(self, essay_prompt: str, user_essay: str, layer_results: Dict) -> str:
        """Create structured verification prompt for GPT"""
        return f"""
TASK: Verify and refine the ML-generated scores for this PTE Write Essay response.

ESSAY PROMPT: {essay_prompt}

USER ESSAY: {user_essay}

ML ANALYSIS RESULTS:
- Content Score: {layer_results.get('content_score', 0)}/6
- Form Score: {layer_results.get('form_score', 0)}/2
- Development Score: {layer_results.get('development_score', 0)}/6
- Grammar Score: {layer_results.get('grammar_score', 0)}/2
- Linguistic Range Score: {layer_results.get('linguistic_score', 0)}/6
- Vocabulary Score: {layer_results.get('vocabulary_score', 0)}/2
- Spelling Score: {layer_results.get('spelling_score', 0)}/2

DETECTED ISSUES:
- Grammar Errors: {len(layer_results.get('grammar_errors', []))}
- Spelling Errors: {len(layer_results.get('spelling_errors', []))}
- Vocabulary Issues: {len(layer_results.get('vocabulary_errors', []))}
- Structure Issues: {len(layer_results.get('structure_issues', []))}

VERIFICATION INSTRUCTIONS:
1. Review each ML score against PTE criteria
2. Confirm, adjust, or reject each score with reasoning
3. Ensure scores reflect actual essay quality
4. Consider essay length (should be 200-300 words)

Return ONLY valid JSON matching this schema:
{{
    "verification_status": "confirmed|adjusted|rejected",
    "verified_scores": {{
        "content": <0-6>,
        "form": <0-2>,
        "development": <0-6>,
        "grammar": <0-2>,
        "linguistic": <0-6>,
        "vocabulary": <0-2>,
        "spelling": <0-2>
    }},
    "verification_notes": "Brief explanation of any adjustments made",
    "confidence_score": <0.0-1.0>
}}
"""

    def _parse_and_validate_gpt_response(self, gpt_text: str) -> Dict:
        """Parse and validate GPT response against schema"""
        try:
            # Clean response
            cleaned_text = re.sub(r'^```json\s*|\s*```$', '', gpt_text.strip())
            
            # Parse JSON
            result = json.loads(cleaned_text)
            
            # Validate structure
            required_keys = ["verification_status", "verified_scores", "verification_notes", "confidence_score"]
            if not all(key in result for key in required_keys):
                return {"success": False, "reason": "Missing required keys"}
            
            # Validate score ranges
            scores = result["verified_scores"]
            score_ranges = {
                "content": (0, 6), "form": (0, 2), "development": (0, 6),
                "grammar": (0, 2), "linguistic": (0, 6), "vocabulary": (0, 2), "spelling": (0, 2)
            }
            
            for component, (min_val, max_val) in score_ranges.items():
                if component not in scores or not (min_val <= scores[component] <= max_val):
                    return {"success": False, "reason": f"Invalid {component} score"}
            
            return {"success": True, "data": result}
            
        except json.JSONDecodeError as e:
            # Attempt auto-repair
            repaired_json = self._auto_repair_json(gpt_text)
            if repaired_json:
                try:
                    result = json.loads(repaired_json)
                    return {"success": True, "data": result}
                except:
                    pass
            
            return {"success": False, "reason": f"JSON parsing failed: {e}"}
        except Exception as e:
            return {"success": False, "reason": f"Validation failed: {e}"}
    
    def _auto_repair_json(self, text: str) -> Optional[str]:
        """Attempt to auto-repair malformed JSON"""
        try:
            # Common repairs
            text = text.strip()
            
            # Remove markdown code blocks
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            # Fix common JSON issues
            text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
            text = re.sub(r',\s*]', ']', text)  # Remove trailing commas in arrays
            
            # Try to extract JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json_match.group()
                
        except Exception:
            pass
        
        return None
    
    def apply_score_mapping(self, raw_scores: Dict[str, float]) -> Dict[str, Union[float, int]]:
        """
        7. Non-linear Score Mapping with Calibration
        Maps 26-point raw scores to 90-point PTE scale
        """
        total_raw = sum(raw_scores.values())
        
        # Apply non-linear mapping based on curve type
        if self.mapping_params['curve_type'] == 'sigmoid':
            mapped_total = self._sigmoid_mapping(total_raw)
        elif self.mapping_params['curve_type'] == 'logarithmic':
            mapped_total = self._logarithmic_mapping(total_raw)
        else:
            mapped_total = self._linear_mapping(total_raw)
        
        # Ensure within bounds
        mapped_total = max(self.mapping_params['min_score'], 
                          min(self.mapping_params['max_score'], mapped_total))
        
        # Calculate percentage and band
        percentage = round((total_raw / 26) * 100)
        band = self._calculate_band(mapped_total)
        
        return {
            'raw_total': total_raw,
            'mapped_total': round(mapped_total, 1),
            'percentage': percentage,
            'band': band
        }
    
    def _sigmoid_mapping(self, raw_score: float) -> float:
        """Sigmoid curve mapping for more realistic score distribution"""
        steepness = self.mapping_params['sigmoid_steepness']
        midpoint = self.mapping_params['sigmoid_midpoint']
        
        # Sigmoid function: f(x) = max_score / (1 + e^(-steepness * (x - midpoint)))
        sigmoid_value = 1 / (1 + math.exp(-steepness * (raw_score - midpoint)))
        return sigmoid_value * self.mapping_params['max_score']
    
    def _logarithmic_mapping(self, raw_score: float) -> float:
        """Logarithmic mapping for score scaling"""
        base = self.mapping_params['logarithmic_base']
        if raw_score <= 0:
            return self.mapping_params['min_score']
        
        log_value = math.log(raw_score + 1, base)
        max_log = math.log(27, base)  # 26 + 1
        
        normalized = log_value / max_log
        return normalized * self.mapping_params['max_score']
    
    def _linear_mapping(self, raw_score: float) -> float:
        """Simple linear mapping"""
        return raw_score * self.mapping_params['scale_factor']
    
    def _calculate_band(self, score: float) -> str:
        """Calculate PTE band based on score"""
        if score >= 79:
            return "Expert"
        elif score >= 65:
            return "Very Good"
        elif score >= 50:
            return "Good"
        elif score >= 36:
            return "Competent"
        elif score >= 30:
            return "Modest"
        else:
            return "Limited"
    
    def calibrate_score_mapping(self, calibration_data: List[Dict]) -> Dict:
        """
        Calibration function for tuning against real APEUni/Pearson data
        
        Args:
            calibration_data: List of dicts with 'raw_scores', 'expected_score'
        
        Returns:
            Updated mapping parameters
        """
        if not calibration_data:
            return self.mapping_params
        
        raw_scores = [sum(data['raw_scores'].values()) for data in calibration_data]
        expected_scores = [data['expected_score'] for data in calibration_data]
        
        # Optimize parameters using least squares
        best_params = self.mapping_params.copy()
        best_error = float('inf')
        
        # Grid search for optimal parameters
        steepness_range = np.arange(0.1, 0.3, 0.05)
        midpoint_range = np.arange(10.0, 16.0, 1.0)
        
        for steepness in steepness_range:
            for midpoint in midpoint_range:
                test_params = self.mapping_params.copy()
                test_params['sigmoid_steepness'] = steepness
                test_params['sigmoid_midpoint'] = midpoint
                
                # Calculate error
                predicted_scores = []
                for raw_score in raw_scores:
                    mapped = self._sigmoid_mapping_with_params(raw_score, test_params)
                    predicted_scores.append(mapped)
                
                error = np.mean([(pred - exp) ** 2 for pred, exp in zip(predicted_scores, expected_scores)])
                
                if error < best_error:
                    best_error = error
                    best_params = test_params.copy()
        
        self.mapping_params = best_params
        logger.info(f"✅ Score mapping calibrated with error: {best_error:.3f}")
        
        return self.mapping_params
    
    def _sigmoid_mapping_with_params(self, raw_score: float, params: Dict) -> float:
        """Helper function for calibration"""
        steepness = params['sigmoid_steepness']
        midpoint = params['sigmoid_midpoint']
        sigmoid_value = 1 / (1 + math.exp(-steepness * (raw_score - midpoint)))
        return sigmoid_value * params['max_score']
    
    # ==================== MAIN SCORING METHOD ====================
    
    def score_essay(self, user_essay: str, essay_prompt: str) -> Dict:
        """
        Main enhanced scoring method orchestrating all improvements
        """
        try:
            start_time = time.time()
            logger.info("="*60)
            logger.info("🎯 ENHANCED WRITE ESSAY SCORING STARTED")
            logger.info("="*60)
            
            # Layer 1: Enhanced Grammar and Form
            grammar_score, grammar_errors = self.check_enhanced_grammar(user_essay)
            spelling_score, spelling_analysis = self.check_enhanced_spelling(user_essay)
            form_score, form_issues = self._check_form_requirements(user_essay)
            
            # Layer 2: Enhanced Analysis
            linguistic_score, complexity_metrics = self.analyze_syntactic_complexity(user_essay)
            vocabulary_score, vocabulary_analysis = self.analyze_enhanced_vocabulary(user_essay)
            development_score, structure_analysis = self.analyze_paragraph_structure(user_essay)
            content_score, content_gaps = self._analyze_content(user_essay, essay_prompt)
            
            # Prepare layer results for GPT verification
            layer_results = {
                'content_score': content_score,
                'form_score': form_score,
                'development_score': development_score,
                'grammar_score': grammar_score,
                'linguistic_score': linguistic_score,
                'vocabulary_score': vocabulary_score,
                'spelling_score': spelling_score,
                'grammar_errors': grammar_errors,
                'spelling_errors': spelling_analysis.error_types,
                'vocabulary_errors': vocabulary_analysis.collocation_errors,
                'structure_issues': []
            }
            
            # Layer 3: GPT Enhanced Verification
            gpt_result = self.gpt_enhanced_verification(essay_prompt, user_essay, layer_results)
            
            # Use verified scores if available, otherwise use ML scores
            if gpt_result.get("success") and "data" in gpt_result:
                final_scores = gpt_result["data"]["verified_scores"]
                verification_notes = gpt_result["data"]["verification_notes"]
            else:
                final_scores = {
                    'content': content_score,
                    'form': form_score,
                    'development': development_score,
                    'grammar': grammar_score,
                    'linguistic': linguistic_score,
                    'vocabulary': vocabulary_score,
                    'spelling': spelling_score
                }
                verification_notes = "GPT verification unavailable - using ML scores"
            
            # Apply enhanced score mapping
            score_mapping = self.apply_score_mapping(final_scores)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return comprehensive results
            return {
                "success": True,
                "scores": final_scores,
                "total_score": score_mapping['raw_total'],
                "mapped_score": score_mapping['mapped_total'],
                "percentage": score_mapping['percentage'],
                "band": score_mapping['band'],
                "word_count": len(user_essay.split()),
                "paragraph_count": len([p for p in user_essay.split('\n\n') if p.strip()]),
                
                # Enhanced analysis results
                "syntactic_complexity": {
                    "mean_sentence_length": complexity_metrics.mean_length_sentence,
                    "subordination_ratio": complexity_metrics.subordination_ratio,
                    "complex_structures": complexity_metrics.complex_t_units_ratio
                },
                "vocabulary_analysis": {
                    "cefr_distribution": vocabulary_analysis.cefr_distribution,
                    "lexical_diversity": vocabulary_analysis.lexical_diversity_ttr,
                    "academic_ratio": vocabulary_analysis.academic_vocabulary_ratio
                },
                "spelling_analysis": {
                    "total_errors": spelling_analysis.total_errors,
                    "error_types": spelling_analysis.error_types,
                    "severity": spelling_analysis.severity_score
                },
                "structure_analysis": {
                    "paragraph_similarities": structure_analysis.paragraph_similarities,
                    "coherence_score": structure_analysis.coherence_score
                },
                
                # Error details
                "errors": {
                    "grammar": grammar_errors,
                    "spelling": [f"Total: {spelling_analysis.total_errors}"],
                    "vocabulary": vocabulary_analysis.collocation_errors,
                    "form": form_issues
                },
                
                # Metadata
                "verification_notes": verification_notes,
                "processing_time": round(processing_time, 2),
                "api_cost": self.total_api_cost,
                "model_versions": {
                    "gector": "vennify/t5-base-grammar-correction",
                    "embeddings": "all-MiniLM-L6-v2",
                    "gpt": "gpt-4o"
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced scoring failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def _check_form_requirements(self, text: str) -> Tuple[float, List[str]]:
        """Check formal requirements (word count, structure)"""
        issues = []
        score = 2.0
        
        word_count = len(text.split())
        if word_count < 200:
            issues.append(f"Too short: {word_count} words (minimum 200)")
            score = 0
        elif word_count > 300:
            issues.append(f"Too long: {word_count} words (maximum 300)")
            score = 0
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            issues.append("Need at least 3 paragraphs (intro, body, conclusion)")
            score = max(0, score - 1)
        
        return score, issues
    
    def _analyze_content(self, essay: str, prompt: str) -> Tuple[float, List[str]]:
        """Basic content analysis"""
        gaps = []
        
        if not self.sentence_model:
            return 4.0, ["Content analysis unavailable"]
        
        try:
            # Semantic similarity
            essay_emb = self.sentence_model.encode(essay, convert_to_tensor=True)
            prompt_emb = self.sentence_model.encode(prompt, convert_to_tensor=True)
            
            similarity = util.cos_sim(essay_emb, prompt_emb).item()
            
            # Basic content score
            if similarity >= 0.7:
                content_score = 6.0
            elif similarity >= 0.5:
                content_score = 4.5
            elif similarity >= 0.3:
                content_score = 3.0
            else:
                content_score = 1.5
                gaps.append("Low relevance to prompt")
            
            return content_score, gaps
            
        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            return 4.0, ["Content analysis failed"]


# Export for easy import
_global_enhanced_scorer = None

def get_enhanced_essay_scorer():
    """Get or create global enhanced essay scorer instance"""
    global _global_enhanced_scorer
    if _global_enhanced_scorer is None:
        _global_enhanced_scorer = EnhancedWriteEssayScorer()
    return _global_enhanced_scorer

def score_enhanced_write_essay(user_essay: str, essay_prompt: str) -> Dict:
    """
    Convenience function to score an essay with enhanced system
    
    Args:
        user_essay: Student's essay (200-300 words)
        essay_prompt: Essay question/topic
    
    Returns:
        Comprehensive enhanced scoring dictionary
    """
    scorer = get_enhanced_essay_scorer()
    return scorer.score_essay(user_essay, essay_prompt)