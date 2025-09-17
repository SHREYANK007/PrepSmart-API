#!/usr/bin/env python3
"""
Test script for Enhanced Hybrid Scorer
Demonstrates the 3-layer scoring system with various test cases
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.scoring.hybrid_scorer_enhanced import get_enhanced_scorer

def test_case_1():
    """Test with grammar and vocabulary errors"""
    print("\n" + "="*60)
    print("TEST CASE 1: Grammar and Vocabulary Errors")
    print("="*60)
    
    passage = """Environmental conservation is crucial for future generations. 
    Children need clean air and water to thrive. We must implement sustainable 
    practices now to ensure a viable future. Without immediate action, 
    irreversible damage will occur to our ecosystems."""
    
    # Intentional errors: missing comma, spelling, informal word
    user_summary = """The passage discuss the importance of enviromental conservation 
    for future generations as kids need clean air and water to survive and we must 
    take action immedietly to prevent damage."""
    
    result = enhanced_hybrid_scorer.comprehensive_score(
        user_summary=user_summary,
        passage=passage
    )
    
    print_results(result)
    return result

def test_case_2():
    """Test with perfect summary"""
    print("\n" + "="*60)
    print("TEST CASE 2: Near-Perfect Summary")
    print("="*60)
    
    passage = """Artificial intelligence has revolutionized various industries 
    by automating complex tasks and improving decision-making processes. However, 
    concerns about job displacement and ethical implications require careful 
    consideration and regulatory frameworks."""
    
    user_summary = """Artificial intelligence has transformed industries through 
    automation and enhanced decision-making, although concerns regarding employment 
    and ethics necessitate thoughtful regulation."""
    
    result = enhanced_hybrid_scorer.comprehensive_score(
        user_summary=user_summary,
        passage=passage
    )
    
    print_results(result)
    return result

def test_case_3():
    """Test with missing content and poor vocabulary"""
    print("\n" + "="*60)
    print("TEST CASE 3: Poor Content Coverage")
    print("="*60)
    
    passage = """Climate change poses unprecedented challenges requiring global 
    cooperation. Rising temperatures, melting ice caps, and extreme weather events 
    threaten biodiversity and human settlements. Immediate transition to renewable 
    energy sources and sustainable practices is imperative."""
    
    # Missing key concepts, basic vocabulary
    user_summary = """Climate change is bad and causes problems like hot weather 
    and we need to use good energy sources."""
    
    result = enhanced_hybrid_scorer.comprehensive_score(
        user_summary=user_summary,
        passage=passage
    )
    
    print_results(result)
    return result

def test_calibration():
    """Test calibration with sample data"""
    print("\n" + "="*60)
    print("CALIBRATION TEST")
    print("="*60)
    
    # Sample calibration dataset
    calibration_data = [
        {
            "passage": "Technology advances rapidly.",
            "user_summary": "Technology is advancing quickly in modern times.",
            "pearson_score": 5.5  # Out of 7
        },
        {
            "passage": "Education is fundamental for development.",
            "user_summary": "Education forms the foundation of societal progress.",
            "pearson_score": 6.0
        }
    ]
    
    calibration_result = enhanced_hybrid_scorer.calibrate(calibration_data)
    
    print("\nCalibration Results:")
    for key, value in calibration_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

def print_results(result):
    """Pretty print scoring results"""
    if not result.get("success"):
        print(f"\n❌ Scoring failed: {result.get('error', 'Unknown error')}")
        return
    
    print(f"\n📊 SCORING RESULTS:")
    print(f"  Total Score: {result['total_score']}/7.0 ({result['percentage']}%)")
    print(f"  Band: {result['band']}")
    
    print(f"\n📈 Component Scores:")
    for component, score in result['scores'].items():
        print(f"  {component.capitalize()}: {score}")
    
    print(f"\n❌ Errors Found:")
    print(f"  Grammar: {len(result.get('grammar_errors', []))} errors")
    if result.get('grammar_errors'):
        for err in result['grammar_errors'][:3]:
            print(f"    • {err}")
    
    print(f"  Vocabulary: {len(result.get('vocabulary_errors', []))} issues")
    if result.get('vocabulary_errors'):
        for err in result['vocabulary_errors'][:3]:
            print(f"    • {err}")
    
    print(f"  Content: {len(result.get('content_feedback', []))} gaps")
    if result.get('content_feedback'):
        for gap in result['content_feedback'][:3]:
            print(f"    • {gap}")
    
    if result.get('strengths'):
        print(f"\n✅ Strengths:")
        for strength in result['strengths']:
            print(f"  • {strength}")
    
    if result.get('improvements'):
        print(f"\n⚠️ Critical Improvements:")
        for improvement in result['improvements']:
            print(f"  • {improvement}")
    
    if result.get('harsh_assessment'):
        print(f"\n🔍 Harsh Assessment:")
        print(f"  {result['harsh_assessment']}")
    
    detailed = result.get('detailed_analysis', {})
    if detailed.get('api_cost'):
        print(f"\n💰 API Cost: ${detailed['api_cost']:.4f}")

def main():
    """Run all tests"""
    print("\n" + "🚀 ENHANCED HYBRID SCORER TEST SUITE 🚀".center(60))
    
    # Run test cases
    test_case_1()
    test_case_2()
    test_case_3()
    
    # Run calibration test
    test_calibration()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()