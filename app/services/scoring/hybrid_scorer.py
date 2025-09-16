"""
Hybrid Scoring System - Rule-based + ML for Pearson-level accuracy
Combines grammar engines, spell checkers, and embeddings
"""

import re
import language_tool_python
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List, Tuple
import logging
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class HybridScorer:
    def __init__(self):
        """Initialize all scoring engines"""
        try:
            # Grammar checker (LanguageTool required) - direct initialization
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
            logger.info("LanguageTool initialized successfully")
            
            # Sentence embeddings for content scoring
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # GPT client for intelligent analysis
            api_key = os.getenv('OPENAI_API_KEY')
            self.openai_client = OpenAI(api_key=api_key) if api_key else None
            self.use_gpt = bool(api_key)
            
            logger.info(f"Hybrid scorer initialized successfully (GPT: {'enabled' if self.use_gpt else 'disabled'})")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid scorer: {e}")
            raise
    
    def score_grammar(self, text: str) -> Tuple[float, List[str]]:
        """
        Grammar scoring using LanguageTool - APEUni level strictness
        Returns: (score out of 2.0, list of errors)
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Grammar scoring failed: {e}")
            return 1.5, ["Grammar check unavailable"]
    
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
            print(f"DEBUG Content: user_summary length: {len(user_summary)}")
            print(f"DEBUG Content: key_points: '{key_points[:100]}...' (length: {len(key_points)})")
            print(f"DEBUG Content: passage length: {len(passage)}")
            
            # Generate embeddings
            user_emb = self.sentence_model.encode(user_summary, convert_to_tensor=True)
            key_emb = self.sentence_model.encode(key_points, convert_to_tensor=True)
            passage_emb = self.sentence_model.encode(passage, convert_to_tensor=True)
            
            # Calculate similarities
            key_similarity = util.pytorch_cos_sim(user_emb, key_emb).item()
            passage_similarity = util.pytorch_cos_sim(user_emb, passage_emb).item()
            
            print(f"DEBUG Content: key_similarity = {key_similarity:.3f}")
            print(f"DEBUG Content: passage_similarity = {passage_similarity:.3f}")
            
            # Check if key_points is just placeholder text
            is_placeholder_keypoints = key_points.lower().startswith("main ideas") or len(key_points) < 50
            
            if is_placeholder_keypoints:
                print("DEBUG: Detected placeholder key_points, using passage similarity only")
                # Use passage similarity only when key_points are placeholder
                if passage_similarity > 0.8:
                    content_score = 2.0
                    feedback = ["Excellent understanding of passage content"]
                elif passage_similarity > 0.7:
                    content_score = 1.8
                    feedback = ["Very good understanding with minor gaps"]
                elif passage_similarity > 0.6:
                    content_score = 1.5
                    feedback = ["Good understanding, some details missing"]
                elif passage_similarity > 0.5:
                    content_score = 1.2
                    feedback = ["Basic understanding of main concepts"]
                elif passage_similarity > 0.4:
                    content_score = 1.0
                    feedback = ["Limited understanding of passage"]
                else:
                    content_score = 0.5
                    feedback = ["Poor understanding of passage content"]
            else:
                # Original logic when key_points are real
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
    
    def gpt_enhanced_analysis(self, user_summary: str, passage: str, key_points: str) -> Dict:
        """
        GPT-powered intelligent analysis for content depth and vocabulary sophistication
        """
        if not self.use_gpt:
            return {"success": False, "reason": "GPT not available"}
            
        try:
            prompt = f"""You are an expert PTE scorer analyzing a 'Summarize Written Text' response with APEUni-level strictness.

PASSAGE (original text):
{passage[:500]}...

USER SUMMARY:
{user_summary}

Analyze this summary for:

1. CONTENT ACCURACY & COMPLETENESS:
   - Does it capture the main idea accurately?
   - Are key details included?
   - Any factual errors or misinterpretations?
   
2. VOCABULARY SOPHISTICATION:
   - Word choice appropriateness
   - Academic vocabulary usage
   - Any informal/inappropriate words
   - Redundancy or repetition issues

3. COHERENCE & FLOW:
   - Logical organization
   - Smooth transitions
   - Clear relationships between ideas

Provide scores out of 2.0 for each category and specific actionable feedback.

Respond in JSON format:
{{
    "content_accuracy": 1.8,
    "vocabulary_quality": 1.9,
    "coherence": 1.7,
    "detailed_feedback": {{
        "strengths": ["strength1", "strength2"],
        "improvements": ["improvement1", "improvement2"],
        "vocabulary_issues": ["issue1", "issue2"],
        "content_gaps": ["gap1", "gap2"]
    }}
}}"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
                timeout=8  # 8 second timeout
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            result["success"] = True
            return result
            
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return {"success": False, "reason": str(e)}
    
    def gpt_double_check_analysis(self, user_summary: str, passage: str, potential_issues: Dict) -> Dict:
        """
        STEP 2: GPT-4o Double-Check Verification System
        GPT verifies all rule-based findings and provides detailed feedback
        """
        if not self.use_gpt:
            return {"success": False, "reason": "GPT not available"}
            
        try:
            initial_scores = potential_issues["initial_scores"]
            grammar_issues = potential_issues["grammar_errors"]
            vocab_issues = potential_issues["vocabulary_errors"]
            
            prompt = f"""You are an expert PTE scorer with APEUni-level strictness. You must DOUBLE-CHECK and VERIFY the findings from automated tools.

ORIGINAL PASSAGE:
{passage}

USER SUMMARY:
{user_summary}

STEP 1 AUTOMATED FINDINGS:
- Grammar Issues Found: {grammar_issues}
- Vocabulary Issues Found: {vocab_issues}
- Initial Scores: Grammar={initial_scores['grammar']}/2, Vocabulary={initial_scores['vocabulary']}/2, Content={initial_scores['content']}/2, Form={initial_scores['form']}/1

YOUR TASK - VERIFY AND ENHANCE:
1. Check EVERY grammar issue found - are they real errors?
2. Find ANY additional errors the tools missed
3. Verify vocabulary problems and find more
4. Provide detailed explanations for EVERY deduction
5. Be STRICT like APEUni - every small error counts

SCORING RULES:
- Grammar: Start at 2.0, deduct 0.5 per error
- Vocabulary: Start at 2.0, deduct 0.5 per spelling/word choice error  
- Content: Evaluate understanding depth
- Form: 5-75 words, single sentence

Respond in JSON format:
{{
    "verified_grammar_score": 1.5,
    "verified_vocabulary_score": 1.5,
    "verified_content_score": 1.8,
    "verified_form_score": 1.0,
    "detailed_issues": {{
        "grammar_details": ["Specific error 1 with location", "Specific error 2 with location"],
        "vocabulary_details": ["Vocab issue 1", "Vocab issue 2"],
        "additional_errors_found": ["New error GPT found", "Another new error"]
    }},
    "strengths": ["What the user did well"],
    "improvements": ["Specific actionable improvements"],
    "harsh_feedback": "Detailed paragraph explaining every deduction like APEUni would"
}}

BE HARSH AND DETAILED. Find every tiny error."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1,
                timeout=10  # 10 second timeout
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            result["success"] = True
            return result
            
        except Exception as e:
            logger.error(f"GPT double-check failed: {e}")
            return {"success": False, "reason": str(e)}
    
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
        2-STEP VERIFICATION SYSTEM:
        Step 1: Rule-based engines find ALL potential issues
        Step 2: GPT-4o double-checks everything and provides detailed feedback
        """
        try:
            print("DEBUG: Starting 2-STEP VERIFICATION SYSTEM")
            
            # STEP 1: RULE-BASED DETECTION (Find ALL potential issues)
            print("DEBUG: STEP 1 - Rule-based detection")
            grammar_score, grammar_errors = self.score_grammar(user_summary)
            vocab_score, vocab_errors = self.score_vocabulary(user_summary, passage)
            content_score, content_feedback = self.score_content(user_summary, key_points, passage)
            form_score, form_feedback = self.score_form(user_summary)
            
            print(f"DEBUG: Step 1 results - G:{grammar_score}, V:{vocab_score}, C:{content_score}, F:{form_score}")
            print(f"DEBUG: Found issues - Grammar: {len(grammar_errors)}, Vocab: {len(vocab_errors)}")
            
            # Collect ALL potential issues for GPT verification
            all_potential_issues = {
                "grammar_errors": grammar_errors,
                "vocabulary_errors": vocab_errors, 
                "content_gaps": content_feedback,
                "form_issues": form_feedback,
                "initial_scores": {
                    "grammar": grammar_score,
                    "vocabulary": vocab_score,
                    "content": content_score,
                    "form": form_score
                }
            }
            
            # STEP 2: GPT-4O VERIFICATION & DETAILED ANALYSIS (with timeout protection)
            print("DEBUG: STEP 2 - GPT-4o verification and detailed analysis")
            try:
                gpt_verification = self.gpt_double_check_analysis(user_summary, passage, all_potential_issues)
            except Exception as e:
                print(f"DEBUG: GPT step failed with error: {e}")
                gpt_verification = {"success": False, "reason": f"Timeout or error: {str(e)}"}
            
            if gpt_verification.get("success"):
                print("DEBUG: GPT verification successful - applying verified scores")
                
                # Use GPT's verified scores (GPT has final say)
                final_grammar = gpt_verification.get("verified_grammar_score", grammar_score)
                final_vocab = gpt_verification.get("verified_vocabulary_score", vocab_score)
                final_content = gpt_verification.get("verified_content_score", content_score)
                final_form = gpt_verification.get("verified_form_score", form_score)
                
                # Get detailed feedback from GPT
                detailed_issues = gpt_verification.get("detailed_issues", {})
                strengths = gpt_verification.get("strengths", [])
                improvements = gpt_verification.get("improvements", [])
                
                print(f"DEBUG: Final verified scores - G:{final_grammar}, V:{final_vocab}, C:{final_content}, F:{final_form}")
                
                # Update scores and errors with GPT verification
                grammar_score = final_grammar
                vocab_score = final_vocab
                content_score = final_content
                form_score = final_form
                
                # Enhanced error lists with GPT details
                grammar_errors = detailed_issues.get("grammar_details", grammar_errors)
                vocab_errors = detailed_issues.get("vocabulary_details", vocab_errors)
                
            else:
                print(f"DEBUG: GPT verification failed: {gpt_verification.get('reason', 'Unknown')}")
                print("DEBUG: Using rule-based scores with stricter deductions")
                
                # Apply stricter rule-based scoring when GPT fails
                if len(grammar_errors) > 0:
                    grammar_score = max(0.0, 2.0 - (len(grammar_errors) * 0.5))
                if len(vocab_errors) > 0:
                    vocab_score = max(0.0, 2.0 - (len(vocab_errors) * 0.5))
                    
                strengths = ["Grammar analysis completed", "Vocabulary check completed"] if grammar_score >= 1.5 and vocab_score >= 1.5 else []
                improvements = ["Review grammar rules", "Check spelling and word choice"] if grammar_score < 1.5 or vocab_score < 1.5 else []
            
            # Calculate total with enhanced scores
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
                # Enhanced fields with GPT insights
                "grammar_justification": f"Grammar score: {grammar_score}/2.0. " + ("; ".join(grammar_errors[:3]) if grammar_errors else "No grammar errors found"),
                "vocabulary_justification": f"Vocabulary score: {vocab_score}/2.0. " + ("; ".join(vocab_errors[:3]) if vocab_errors else "No vocabulary errors found") + (f" GPT insights: {'; '.join(vocab_issues[:2])}" if vocab_issues else ""),
                "content_justification": f"Content score: {content_score}/2.0. " + ("; ".join(content_feedback[:2]) if content_feedback else "Good content coverage"),
                "form_justification": f"Form score: {form_score}/1.0. " + ("; ".join(form_feedback) if form_feedback else "Perfect form"),
                "strengths": strengths,
                "ai_recommendations": improvements,
                "gpt_analysis": gpt_analysis if gpt_analysis.get("success") else None,
                "feedback": {
                    "grammar": f"Grammar score: {grammar_score}/2.0. " + "; ".join(grammar_errors[:3]),
                    "vocabulary": f"Vocabulary score: {vocab_score}/2.0. " + "; ".join(vocab_errors[:3]) + (f" | GPT: {'; '.join(vocab_issues[:1])}" if vocab_issues else ""),
                    "content": f"Content score: {content_score}/2.0. " + "; ".join(content_feedback[:2]),
                    "form": f"Form score: {form_score}/1.0. " + "; ".join(form_feedback)
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive scoring failed: {e}")
            return {"success": False, "error": str(e)}

# Global instance
hybrid_scorer = HybridScorer()