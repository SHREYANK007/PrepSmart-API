#!/usr/bin/env python3
"""
Test script to verify calibrated Write Essay scorer against the user's sample
Should detect all 5 spelling errors and score appropriately
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.scoring.write_essay_scorer import score_write_essay

def test_calibrated_scorer():
    """Test the calibrated scorer with the user's essay sample"""
    
    # Sample essay with known spelling errors
    essay_prompt = "Write about the advantages and disadvantages of technology in education"
    
    user_essay = """
Technology has become an integral part of modern education, bringing both advantages and disadvantages. 
This essay will explore the topc of educational technology and its impact on learning.

One of the main advantages of technology in education is improved access to information. Students can now 
access vast amounts of educational resources online, making learning more efficient. Additionally, 
technology enables interactive learning experiences through multimedia content and virtual simulations.

However, there are also significant disadvangtes to consider. One major concern is the digital divide, 
where not all students have equal access to technology. This can create inequalities in educational 
opportunities. Furthermore, excessive reliance on technology can reduce face-to-face interaction 
between teachers and students.

Another prominet issue is the potential for distraction. With access to social media and entertainment 
platforms, students may struggle to focus on their studies. As one expert noted, "Technology is a 
double-edged sword" (qoute from Education Today, 2023).

In conclusion, while technology offers many benefits for education, it also presents challenges that 
must be addressed. Recent reporst suggest that a balanced approach, combining traditional teaching 
methods with technological tools, may be the most effective solution for modern education.
"""

    print("🎯 Testing Calibrated Write Essay Scorer")
    print("="*60)
    print(f"Essay word count: {len(user_essay.split())} words")
    print()
    
    # Test the scorer
    result = score_write_essay(user_essay, essay_prompt)
    
    if result.get("success"):
        scores = result["scores"]
        errors = result["errors"]
        
        print("📊 SCORING RESULTS:")
        print(f"Total Score: {result['total_score']}/26 ({result['percentage']}%)")
        print(f"Band: {result['band']}")
        print()
        
        print("📋 COMPONENT SCORES:")
        for component, score in scores.items():
            if component == "content":
                print(f"  Content: {score}/6")
            elif component == "form":
                print(f"  Form: {score}/2")
            elif component == "development":
                print(f"  Development & Coherence: {score}/6")
            elif component == "grammar":
                print(f"  Grammar: {score}/2")
            elif component == "linguistic":
                print(f"  General Linguistic Range: {score}/6")
            elif component == "vocabulary":
                print(f"  Vocabulary Range: {score}/2")
            elif component == "spelling":
                print(f"  Spelling: {score}/2")
        print()
        
        print("🔍 ERROR DETECTION:")
        
        # Check spelling errors specifically
        spelling_errors = errors.get("spelling", [])
        print(f"Spelling Errors Found: {len(spelling_errors)}")
        
        expected_errors = ["topc", "disadvangtes", "prominet", "qoute", "reporst"]
        found_errors = []
        
        for error in spelling_errors:
            print(f"  - {error}")
            # Extract the misspelled word
            if "→" in error:
                word = error.split("→")[0].strip()
                found_errors.append(word)
        
        print()
        print("✅ VALIDATION:")
        print(f"Expected to find: {expected_errors}")
        print(f"Actually found: {found_errors}")
        
        all_found = all(error in " ".join(spelling_errors) for error in expected_errors)
        print(f"All 5 spelling errors detected: {'✅ YES' if all_found else '❌ NO'}")
        
        # Check grammar errors
        grammar_errors = errors.get("grammar", [])
        print(f"Grammar Errors Found: {len(grammar_errors)}")
        for error in grammar_errors:
            print(f"  - {error}")
        
        print()
        print("🎯 CALIBRATION SUMMARY:")
        print(f"Spelling Score: {scores['spelling']}/2 (should detect all 5 errors but not be too harsh)")
        print(f"Grammar Score: {scores['grammar']}/2 (should be realistic like APEUni ~1.7)")
        
        # Check if calibration is working
        if all_found and scores['spelling'] >= 0.5:
            print("✅ Spelling calibration: SUCCESSFUL - Detects all errors but scoring is balanced")
        else:
            print("❌ Spelling calibration: NEEDS ADJUSTMENT")
            
        if 1.0 <= scores['grammar'] <= 2.0:
            print("✅ Grammar calibration: SUCCESSFUL - Realistic scoring")
        else:
            print("❌ Grammar calibration: NEEDS ADJUSTMENT")
    
    else:
        print(f"❌ Scoring failed: {result.get('error')}")
    
    return result

if __name__ == "__main__":
    test_calibrated_scorer()