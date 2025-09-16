#!/usr/bin/env python3
"""
Pre-download all ML models for PrepSmart Enhanced Scorer
Run this before starting the API to avoid startup hangs
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_transformers_models():
    """Download grammar correction models"""
    print("🔄 Downloading grammar models...")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "vennify/t5-base-grammar-correction"
        print(f"Downloading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print("✅ Grammar models downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Grammar model download failed: {e}")
        return False

def download_sentence_transformers():
    """Download sentence embedding models"""
    print("🔄 Downloading sentence transformer models...")
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = "all-MiniLM-L6-v2"
        print(f"Downloading {model_name}...")
        
        model = SentenceTransformer(model_name)
        
        print("✅ Sentence transformer models downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Sentence transformer download failed: {e}")
        return False

def download_spacy_models():
    """Download spaCy language models"""
    print("🔄 Downloading spaCy models...")
    try:
        import subprocess
        
        # Download English model
        result = subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ spaCy models downloaded successfully")
            return True
        else:
            print(f"❌ spaCy download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ spaCy download failed: {e}")
        return False

def download_nltk_data():
    """Download NLTK data"""
    print("🔄 Downloading NLTK data...")
    try:
        import nltk
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ NLTK download failed: {e}")
        return False

def initialize_language_tool():
    """Initialize LanguageTool (downloads grammar rules)"""
    print("🔄 Initializing LanguageTool...")
    try:
        import language_tool_python
        
        # This will download the LanguageTool JAR if needed
        tool = language_tool_python.LanguageTool('en-US')
        tool.close()
        
        print("✅ LanguageTool initialized successfully")
        return True
    except Exception as e:
        print(f"❌ LanguageTool initialization failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print("🔄 Testing OpenAI connection...")
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️ No OpenAI API key found in environment")
            return False
            
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        
        print("✅ OpenAI connection successful")
        return True
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def main():
    """Download all models and test connections"""
    print("🚀 PrepSmart Model Downloader")
    print("=" * 50)
    
    results = []
    
    # Download all models
    results.append(("Grammar Models", download_transformers_models()))
    results.append(("Sentence Transformers", download_sentence_transformers()))
    results.append(("spaCy Models", download_spacy_models()))
    results.append(("NLTK Data", download_nltk_data()))
    results.append(("LanguageTool", initialize_language_tool()))
    results.append(("OpenAI Connection", test_openai_connection()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 50)
    
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name:20} : {status}")
    
    total_success = sum(success for _, success in results)
    total_items = len(results)
    
    print(f"\nOverall: {total_success}/{total_items} components ready")
    
    if total_success == total_items:
        print("\n🎉 All models downloaded! API startup should be fast now.")
    else:
        print(f"\n⚠️ {total_items - total_success} components failed. Check errors above.")
    
    return total_success == total_items

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)