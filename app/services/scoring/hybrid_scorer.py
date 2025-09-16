"""
Hybrid Scoring System - Rule-based + ML for Pearson-level accuracy
Combines grammar engines, spell checkers, and embeddings
"""

import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List, Tuple
import logging

# Try to import LanguageTool, fallback to regex if Java not available
try:
    import language_tool_python
    LANGUAGETOOL_AVAILABLE = True
except Exception:
    LANGUAGETOOL_AVAILABLE = False

logger = logging.getLogger(__name__)

class HybridScorer:
    def __init__(self):
        """Initialize all scoring engines"""
        try:
            # Grammar checker (with fallback)
            if LANGUAGETOOL_AVAILABLE:
                try:
                    self.grammar_tool = language_tool_python.LanguageTool('en-US')
                    self.use_languagetool = True
                    logger.info("LanguageTool initialized successfully")
                except Exception as e:
                    logger.warning(f"LanguageTool failed (Java required): {e}")
                    self.use_languagetool = False
            else:
                self.use_languagetool = False
                logger.info("LanguageTool not available, using regex fallback")
            
            # Sentence embeddings for content scoring
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Hybrid scorer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid scorer: {e}")
            raise
    
    def score_grammar(self, text: str) -> Tuple[float, List[str]]:
        """
        Grammar scoring using LanguageTool or regex fallback - APEUni level strictness
        Returns: (score out of 2.0, list of errors)
        """
        try:
            if self.use_languagetool:
                # Use LanguageTool for advanced grammar checking
                matches = self.grammar_tool.check(text)
                
                # Categorize errors with different penalties (harsher like APEUni)
                critical_errors = []  # -0.5 each
                minor_errors = []     # -0.5 each (make all errors equal penalty)
                tiny_errors = []      # -0.5 each
                
                for match in matches:
                    error_type = match.ruleId
                    error_msg = f"{match.message} at position {match.offset}"
                    
                    # Critical grammar errors
                    if any(x in error_type.lower() for x in ['subject_verb', 'agreement', 'tense']):
                        critical_errors.append(error_msg)
                    # Minor punctuation errors
                    elif any(x in error_type.lower() for x in ['comma', 'punctuation', 'apostrophe']):
                        minor_errors.append(error_msg)
                    # Tiny errors
                    else:
                        tiny_errors.append(error_msg)
                
                # Calculate deductions (APEUni-style harsh penalties - 0.5 per error)
                total_deduction = (
                    len(critical_errors) * 0.5 +
                    len(minor_errors) * 0.5 +
                    len(tiny_errors) * 0.5
                )
                
                # Cap at 2.0 maximum
                grammar_score = max(0.0, 2.0 - total_deduction)
                
                # Format error messages like APEUni
                all_errors = []
                for err in critical_errors:
                    all_errors.append(f"CRITICAL: {err}")
                for err in minor_errors:
                    all_errors.append(f"MINOR: {err}")
                for err in tiny_errors:
                    all_errors.append(f"TINY: {err}")
                
                return round(grammar_score, 1), all_errors
            else:
                # Fallback regex-based grammar checking
                return self._regex_grammar_check(text)
            
        except Exception as e:
            logger.error(f"Grammar scoring failed: {e}")
            return 1.5, ["Grammar check unavailable"]
    
    def _regex_grammar_check(self, text: str) -> Tuple[float, List[str]]:
        """
        Fallback regex-based grammar checking for common errors
        """
        errors = []
        total_deduction = 0.0
        
        # Check for common missing comma patterns
        comma_patterns = [
            (r'\b(however|therefore|moreover|furthermore|consequently|nevertheless|meanwhile)\s+[a-z]', 'Missing comma after transition word'),
            (r'\b(also|too|as well)\s+[a-z]', 'Missing comma after "also/too/as well"'),
            (r'\b(first|second|third|finally|lastly)\s+[a-z]', 'Missing comma after sequence word'),
            (r'\b(for example|for instance|in fact|in addition|in contrast)\s+[a-z]', 'Missing comma after phrase'),
        ]
        
        for pattern, message in comma_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                errors.append(f"MINOR: {message} at position {match.start()}")
                total_deduction += 0.5
        
        # Check for basic subject-verb agreement issues
        agreement_patterns = [
            (r'\b(children|people|students)\s+is\b', 'Subject-verb disagreement: plural subject with singular verb'),
            (r'\b(child|person|student)\s+are\b', 'Subject-verb disagreement: singular subject with plural verb'),
        ]
        
        for pattern, message in agreement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                errors.append(f"CRITICAL: {message} at position {match.start()}")
                total_deduction += 0.5
        
        # Check for missing apostrophes
        apostrophe_patterns = [
            (r'\b(dont|doesnt|wont|cant|isnt|arent|wasnt|werent|havent|hasnt|shouldnt|wouldnt|couldnt)\b', 'Missing apostrophe in contraction'),
        ]
        
        for pattern, message in apostrophe_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                errors.append(f"MINOR: {message} at position {match.start()}")
                total_deduction += 0.5
        
        # Calculate final score
        grammar_score = max(0.0, 2.0 - total_deduction)
        
        if not errors:
            errors = ["Basic grammar check passed (install Java for advanced checking)"]
        
        return round(grammar_score, 1), errors
    
    def score_vocabulary(self, text: str, passage: str) -> Tuple[float, List[str]]:
        """
        Vocabulary scoring: spelling + redundancy + appropriateness
        Returns: (score out of 2.0, list of errors)
        """
        try:
            errors = []
            total_deduction = 0.0
            
            # 1. Basic spell checking (common misspellings)
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Common misspellings
            common_misspellings = {
                'recieve': 'receive', 'occured': 'occurred', 'seperate': 'separate',
                'definately': 'definitely', 'begining': 'beginning', 'enviroment': 'environment',
                'goverment': 'government', 'necessery': 'necessary', 'tomorow': 'tomorrow',
                'wich': 'which', 'thier': 'their', 'freind': 'friend'
            }
            
            misspelled = []
            for word in words:
                if word in common_misspellings:
                    misspelled.append(f"{word} → {common_misspellings[word]}")
                elif len(word) > 20:  # Extremely long words likely errors
                    misspelled.append(word)
            
            # 2. Redundancy check (copied from passage)
            passage_words = set(re.findall(r'\b\w+\b', passage.lower()))
            text_words = re.findall(r'\b\w+\b', text.lower())
            
            copied_words = [w for w in text_words if w in passage_words and len(w) > 4]
            redundancy_ratio = len(copied_words) / max(len(text_words), 1)
            
            if redundancy_ratio > 0.7:
                errors.append("Excessive copying from passage")
                total_deduction += 0.5
            elif redundancy_ratio > 0.5:
                errors.append("High word repetition from passage")
                total_deduction += 0.5
            
            # 3. Inappropriate informal words
            informal_words = ['kids', 'stuff', 'things', 'guys', 'gonna', 'wanna']
            found_informal = [w for w in words if w in informal_words]
            for word in found_informal:
                errors.append(f"Informal word: '{word}' (use formal alternative)")
                total_deduction += 0.2
            
            # 4. Spelling errors
            for word in misspelled:
                errors.append(f"Spelling error: '{word}'")
                total_deduction += 0.2
            
            vocab_score = max(0.0, 2.0 - total_deduction)
            return round(vocab_score, 1), errors
            
        except Exception as e:
            logger.error(f"Vocabulary scoring failed: {e}")
            return 1.5, ["Vocabulary check unavailable"]
    
    def score_content(self, user_summary: str, key_points: str, passage: str) -> Tuple[float, List[str]]:
        """
        Content scoring using semantic similarity embeddings
        Returns: (score out of 2.0, coverage details)
        """
        try:
            # Generate embeddings
            user_emb = self.sentence_model.encode(user_summary, convert_to_tensor=True)
            key_emb = self.sentence_model.encode(key_points, convert_to_tensor=True)
            passage_emb = self.sentence_model.encode(passage, convert_to_tensor=True)
            
            # Calculate similarities
            key_similarity = util.pytorch_cos_sim(user_emb, key_emb).item()
            passage_similarity = util.pytorch_cos_sim(user_emb, passage_emb).item()
            
            # Weighted content score (Pearson-style)
            if key_similarity > 0.8 and passage_similarity > 0.7:
                content_score = 2.0
                feedback = ["Excellent coverage of key points"]
            elif key_similarity > 0.7 and passage_similarity > 0.6:
                content_score = 1.8
                feedback = ["Good coverage with minor gaps"]
            elif key_similarity > 0.6 and passage_similarity > 0.5:
                content_score = 1.5
                feedback = ["Adequate coverage, missing some key points"]
            elif key_similarity > 0.5 and passage_similarity > 0.4:
                content_score = 1.2
                feedback = ["Basic coverage, several key points missing"]
            elif key_similarity > 0.4:
                content_score = 1.0
                feedback = ["Limited coverage of main ideas"]
            else:
                content_score = 0.5
                feedback = ["Poor understanding of passage content"]
            
            # Add specific feedback
            if key_similarity < 0.7:
                feedback.append(f"Key points similarity: {key_similarity:.2f} (needs improvement)")
            if passage_similarity < 0.6:
                feedback.append(f"Passage alignment: {passage_similarity:.2f} (needs improvement)")
            
            return content_score, feedback
            
        except Exception as e:
            logger.error(f"Content scoring failed: {e}")
            return 1.5, ["Content analysis unavailable"]
    
    def score_form(self, text: str) -> Tuple[float, List[str]]:
        """
        Form scoring: strict regex validation (not GPT guessing)
        Returns: (score out of 1.0, form issues)
        """
        try:
            issues = []
            
            # Count words
            words = re.findall(r'\b\w+\b', text)
            word_count = len(words)
            
            # Count sentences (basic detection)
            sentences = re.split(r'[.!?]+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            sentence_count = len(sentences)
            
            # Form validation
            if word_count < 5:
                issues.append(f"Too short: {word_count} words (minimum 5)")
                return 0.0, issues
            elif word_count > 75:
                issues.append(f"Too long: {word_count} words (maximum 75)")
                return 0.0, issues
            
            if sentence_count != 1:
                issues.append(f"Must be single sentence, found {sentence_count}")
                return 0.0, issues
            
            # Perfect form
            return 1.0, [f"Perfect form: {word_count} words, single sentence"]
            
        except Exception as e:
            logger.error(f"Form scoring failed: {e}")
            return 0.5, ["Form check unavailable"]
    
    def comprehensive_score(self, user_summary: str, passage: str, key_points: str) -> Dict:
        """
        Complete hybrid scoring - combines all engines
        Returns full scoring breakdown like APEUni
        """
        try:
            # Individual component scoring
            grammar_score, grammar_errors = self.score_grammar(user_summary)
            vocab_score, vocab_errors = self.score_vocabulary(user_summary, passage)
            content_score, content_feedback = self.score_content(user_summary, key_points, passage)
            form_score, form_feedback = self.score_form(user_summary)
            
            # Calculate total
            total_score = grammar_score + vocab_score + content_score + form_score
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
            
            return {
                "success": True,
                "scores": {
                    "grammar": grammar_score,
                    "vocabulary": vocab_score,
                    "content": content_score,
                    "form": form_score
                },
                "total_score": round(total_score, 1),
                "percentage": percentage,
                "band": band,
                "grammar_errors": grammar_errors,
                "vocabulary_errors": vocab_errors,
                "content_feedback": content_feedback,
                "form_feedback": form_feedback,
                "detailed_analysis": {
                    "total_grammar_errors": len(grammar_errors),
                    "total_vocabulary_errors": len(vocab_errors),
                    "error_breakdown": f"Grammar: {len(grammar_errors)}, Vocabulary: {len(vocab_errors)}, Content gaps: {2.0 - content_score}, Form issues: {1.0 - form_score}"
                },
                # Add fields that main.py expects
                "grammar_justification": f"Grammar score: {grammar_score}/2.0. " + ("; ".join(grammar_errors[:3]) if grammar_errors else "No grammar errors found"),
                "vocabulary_justification": f"Vocabulary score: {vocab_score}/2.0. " + ("; ".join(vocab_errors[:3]) if vocab_errors else "No vocabulary errors found"),
                "content_justification": f"Content score: {content_score}/2.0. " + ("; ".join(content_feedback[:2]) if content_feedback else "Good content coverage"),
                "form_justification": f"Form score: {form_score}/1.0. " + ("; ".join(form_feedback) if form_feedback else "Perfect form"),
                "feedback": {
                    "grammar": f"Grammar score: {grammar_score}/2.0. " + "; ".join(grammar_errors[:3]),
                    "vocabulary": f"Vocabulary score: {vocab_score}/2.0. " + "; ".join(vocab_errors[:3]),
                    "content": f"Content score: {content_score}/2.0. " + "; ".join(content_feedback[:2]),
                    "form": f"Form score: {form_score}/1.0. " + "; ".join(form_feedback)
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive scoring failed: {e}")
            return {"success": False, "error": str(e)}

# Global instance
hybrid_scorer = HybridScorer()