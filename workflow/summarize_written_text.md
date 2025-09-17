# Summarize Written Text (SWT) - Workflow

## Overview
3-layer hybrid scoring system with GPT-4o ultimate language expert verification.

## Workflow Steps

### 1. Input Processing
- **User Summary**: Student's written summary (5-75 words)
- **Reading Passage**: Original text to be summarized
- **Key Points**: Important concepts to be covered (optional)

### 2. Layer 1: Grammar Analysis
- **Primary**: GECToR (T5-based grammar correction)
- **Fallback**: LanguageTool for grammar checking
- **Detection**: Tense errors, subject-verb agreement, articles, prepositions
- **Scoring**: 0-2.0 points based on error count and severity

### 3. Layer 2: Vocabulary Analysis
- **CEFR Level Assessment**: A1-C2 sophistication scoring
- **Spelling Check**: Dictionary-based validation + custom misspellings
- **Collocation Rules**: Detect incorrect word combinations
- **Word Choice**: Formal vs informal language appropriateness
- **Scoring**: 0-2.0 points based on vocabulary quality

### 4. Layer 3: Content Analysis
- **Semantic Similarity**: Sentence-BERT embeddings (70% weight)
- **Keyword Coverage**: Extract and match key concepts (30% weight)
- **Logical Connectors**: Assess flow and coherence
- **Percentage-based Scoring**:
  - 50%+ coverage → 2.0 points
  - 40-50% coverage → 1.5 points
  - 30-40% coverage → 1.0 points
  - 25-30% coverage → 0.5 points
  - <25% coverage → 0.0 points

### 5. Form Validation
- **Word Count**: Must be 5-75 words
- **Single Sentence**: Required format
- **Scoring**: 1.0 if valid, 0.0 if invalid

### 6. GPT-4o Ultimate Language Expert
- **Comprehensive Analysis**: 100+ point intelligent inspection
- **Grammar Categories**: 10 comprehensive checks (tenses, articles, prepositions, etc.)
- **Vocabulary Categories**: 10 comprehensive checks (spelling, word choice, formality, etc.)
- **Content Review**: Missing concepts, logical flow, coherence
- **Verification**: Re-scores all components using AI intelligence
- **Suggestions**: Detailed corrections and learning tips

### 7. Final Scoring Compilation
- **Total Score**: /7.0 (Grammar 2.0 + Vocabulary 2.0 + Content 2.0 + Form 1.0)
- **Band Assignment**: 
  - 86%+ → Excellent
  - 71-85% → Very Good
  - 57-70% → Good
  - <57% → Needs Improvement

### 8. Response Generation
- **Component Scores**: Individual breakdowns
- **Error Lists**: Grammar, vocabulary, content gaps
- **Suggestions**: GPT-generated intelligent recommendations
- **Harsh Assessment**: Honest feedback for improvement
- **Strengths**: Positive aspects identified

## API Endpoints
- `POST /summarize-text` - Main scoring endpoint

## Models Required
- **GECToR**: `vennify/t5-base-grammar-correction`
- **Sentence-BERT**: `all-MiniLM-L6-v2`
- **spaCy**: `en_core_web_sm`
- **NLTK**: punkt, stopwords, wordnet, omw-1.4
- **GPT-4o**: OpenAI API integration

## Performance
- **Timeout**: 3 minutes for complex analysis
- **Cost Tracking**: OpenAI API usage monitoring
- **Caching**: Lazy model initialization

## Status
✅ **COMPLETE & LOCKED** - No changes until explicitly requested