#!/usr/bin/env python3
"""
Test script for Write Essay Scorer
Tests the 3-layer hybrid scoring system with various essay samples
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.scoring.write_essay_scorer import get_essay_scorer

def test_case_1():
    """Test with grammar and spelling errors"""
    print("\n" + "="*60)
    print("TEST CASE 1: Essay with Grammar/Spelling Errors")
    print("="*60)
    
    prompt = """Some people think that parents should teach children how to be good members 
    of society. Others, however, believe that school is the place to learn this. 
    Discuss both these views and give your own opinion."""
    
    # Essay with intentional errors
    essay = """In todays world their is a debate about who should be responsable for teaching 
    children to become good members of society. While some beleive that parents should take 
    this role others argue that schools are better equiped for this task. In my opinion both 
    parents and schools play crucial roles in shaping childrens social development.

    Parents are the first teachers in a childs life and they have the most influence on there 
    behavior and values. Children learn by observing their parents actions and this learning 
    begins from very early age. For example when parents show kindness respect and honesty 
    children naturaly adopt these qualities. Furthermore parents can provide personalized 
    guidance based on their childs unique personality and needs which schools cannot always do.

    On the other hand schools offer a structured enviroment where children can learn social 
    skills through interaction with peers from diverse backgrounds. Teachers are trained 
    professionals who can teach important concepts like cooperation teamwork and civic 
    responsibility through organized activities. Schools also provide oppurtunities for 
    children to practice these skills in real situations such as group projects and sports.

    In conclusion I believe that both parents and schools have essential roles in teaching 
    children to be good members of society. Parents provide the foundation of values while 
    schools offer the practical experience and formal education. The best outcomes occur when 
    parents and schools work together to guide childrens development."""
    
    scorer = get_essay_scorer()
    result = scorer.score_essay(essay, prompt)
    print_results(result)
    return result

def test_case_2():
    """Test with high-quality essay"""
    print("\n" + "="*60)
    print("TEST CASE 2: High-Quality Essay")
    print("="*60)
    
    prompt = """In many countries, the proportion of older people is steadily increasing. 
    Does this trend have positive or negative effects on society?"""
    
    essay = """The demographic shift towards an aging population is a phenomenon observed 
    across numerous developed nations. While this trend presents certain challenges, I believe 
    its effects on society are predominantly positive, particularly in terms of wisdom 
    preservation and economic stability.

    Firstly, older individuals constitute a repository of invaluable experience and knowledge. 
    Their accumulated wisdom, gained through decades of professional and personal experiences, 
    serves as an irreplaceable resource for younger generations. Moreover, many senior citizens 
    continue to contribute meaningfully to society through mentorship programs, volunteer work, 
    and consultancy roles, thereby bridging generational gaps and fostering social cohesion.

    Additionally, contrary to popular belief, the aging population can stimulate economic growth 
    in specific sectors. The healthcare, leisure, and service industries have expanded 
    significantly to cater to older demographics, creating numerous employment opportunities. 
    Furthermore, many seniors possess substantial savings and disposable income, which they 
    inject into the economy through consumption and investment.

    Nevertheless, critics rightfully highlight the strain on healthcare systems and pension 
    schemes. However, these challenges can be mitigated through progressive policies such as 
    raising the retirement age, encouraging healthy lifestyles, and implementing sustainable 
    pension reforms.

    In conclusion, while an aging population necessitates societal adaptations, the benefits 
    of wisdom preservation, continued productivity, and economic stimulation outweigh the 
    challenges. Societies that embrace this demographic transition and implement appropriate 
    policies will ultimately thrive."""
    
    scorer = get_essay_scorer()
    result = scorer.score_essay(essay, prompt)
    print_results(result)
    return result

def test_case_3():
    """Test with poor content and structure"""
    print("\n" + "="*60)
    print("TEST CASE 3: Poor Content and Structure")
    print("="*60)
    
    prompt = """Some people believe that unpaid community service should be a compulsory 
    part of high school programs. To what extent do you agree or disagree?"""
    
    essay = """I think community service is good for students. Students can learn many things 
    from doing community service. It helps them understand society better.

    Community service teaches responsibility. When students help others, they learn to care 
    about people. This is very important for their future. They also learn new skills that 
    can help them get jobs later. Many employers like to see community service on resumes.

    Some people say students are too busy with studies. They think adding community service 
    will make students more stressed. But I think if we plan it well, it won't be a problem. 
    Students can do service during holidays or weekends. Schools can make it fun and 
    interesting so students will enjoy it.

    In my country, some schools already have community service programs. The students seem 
    happy and they learn a lot. They help old people, clean parks, and teach younger children. 
    These activities make them better persons.

    To conclude, I agree that community service should be compulsory in high schools. It will 
    help students become better citizens and learn important life skills. Schools just need 
    to organize it properly so it doesn't interfere with academic studies."""
    
    scorer = get_essay_scorer()
    result = scorer.score_essay(essay, prompt)
    print_results(result)
    return result

def test_case_4():
    """Test with word count issues"""
    print("\n" + "="*60)
    print("TEST CASE 4: Word Count Issue (Too Short)")
    print("="*60)
    
    prompt = """Technology has made our lives easier. Do you agree or disagree?"""
    
    # Only about 150 words
    essay = """I strongly agree that technology has made our lives significantly easier in many ways. 

    First, communication has become instant and effortless. We can now video call family 
    members across the globe, send messages instantly, and share important information in 
    seconds. This was impossible just a few decades ago.

    Second, technology has revolutionized how we access information. Through the internet, 
    we can learn about any topic, take online courses, and solve problems quickly. Students 
    can research assignments efficiently, and professionals can stay updated with the latest 
    developments in their fields.

    Finally, daily tasks have been simplified through technological innovations. Online banking 
    saves us trips to the bank, GPS navigation prevents us from getting lost, and smart home 
    devices automate routine tasks.

    In conclusion, technology has undeniably made life more convenient and efficient, though 
    we must use it wisely to avoid becoming overly dependent on it."""
    
    scorer = get_essay_scorer()
    result = scorer.score_essay(essay, prompt)
    print_results(result)
    return result

def print_results(result):
    """Pretty print scoring results"""
    if not result.get("success"):
        print(f"\n❌ Scoring failed: {result.get('error', 'Unknown error')}")
        return
    
    print(f"\n📊 SCORING RESULTS:")
    print(f"  Total Score: {result['total_score']}/26 ({result['percentage']}%)")
    print(f"  Band: {result['band']}")
    
    print(f"\n📈 Component Scores:")
    for component, score in result['scores'].items():
        print(f"  {component.capitalize()}: {score}")
    
    print(f"\n❌ Errors Found:")
    for error_type, errors in result['errors'].items():
        if errors:
            print(f"  {error_type.capitalize()}: {len(errors)} issues")
            for err in errors[:3]:  # Show first 3
                print(f"    • {err}")
    
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
    
    if result.get('api_cost'):
        print(f"\n💰 API Cost: ${result['api_cost']:.4f}")

def main():
    """Run all tests"""
    print("\n" + "🚀 WRITE ESSAY SCORER TEST SUITE 🚀".center(60))
    
    # Check if scorer can initialize
    print("\n🔧 Initializing Essay Scorer...")
    try:
        scorer = get_essay_scorer()
        print("✅ Scorer initialized successfully")
        print(f"  GPT Available: {scorer.use_gpt}")
        print(f"  GECToR Available: {scorer.gector_model is not None}")
        print(f"  LanguageTool Available: {scorer.language_tool is not None}")
        print(f"  Sentence Transformer Available: {scorer.sentence_model is not None}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Run test cases
    test_case_1()  # Grammar/spelling errors
    test_case_2()  # High-quality essay
    test_case_3()  # Poor content/structure
    test_case_4()  # Word count issue
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()