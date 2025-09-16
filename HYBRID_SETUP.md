# Hybrid Scoring System Setup

This system combines rule-based checking + ML embeddings for APEUni/Pearson-level accuracy.

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements_hybrid.txt
```

### 2. Download Language Models
```bash
# LanguageTool (will download automatically on first use)
# Sentence Transformers model (will download on first use)

# SpaCy model (optional, for advanced parsing)
python -m spacy download en_core_web_sm
```

### 3. Setup SymSpell Dictionary (Optional)
```bash
# Download a frequency dictionary for better spell checking
# You can use the default or download from SymSpell repository
```

## What This Gives You

### 🎯 Grammar Scoring (LanguageTool Engine)
- **Micro-deductions**: -0.3 critical, -0.2 minor, -0.1 tiny errors
- **Error Categories**: Subject-verb agreement, comma placement, apostrophes
- **APEUni-style reporting**: "Missing comma after 'also'"

### 🔤 Vocabulary Scoring (Multi-layered)
- **Spell checking**: Character-level error detection
- **Redundancy detection**: Penalizes copying from passage
- **Formality checking**: Flags informal words (kids → children)
- **Weighted penalties**: Different deductions per error type

### 📝 Content Scoring (Sentence Embeddings)
- **Semantic similarity**: User summary vs key points
- **Passage alignment**: Checks understanding depth
- **Coverage scoring**: Partial credit based on similarity scores
- **Embedding model**: all-MiniLM-L6-v2 (fast, accurate)

### 📏 Form Scoring (Regex Validation)
- **Word count**: Exact 5-75 word validation
- **Sentence count**: Must be exactly 1 sentence
- **No GPT guessing**: Rule-based precision

## Scoring Breakdown

```python
# Example output
{
    "grammar": 1.8,  # 2.0 - 0.2 (one comma error)
    "vocabulary": 2.0,  # Perfect spelling/word choice
    "content": 1.7,  # Good coverage, minor gaps
    "form": 1.0  # Perfect form
}
# Total: 6.5/7 (93% - Very Good)
```

## Error Reporting Examples

### Grammar Errors (LanguageTool)
- "MINOR: Missing comma after 'also' at position 15"
- "CRITICAL: Subject-verb disagreement at position 42"
- "TINY: Capitalization error at sentence start"

### Vocabulary Errors (Multi-check)
- "Spelling error: 'recieve' → 'receive'"
- "Informal word: 'kids' (use 'children')"
- "Excessive copying from passage (70% overlap)"

### Content Feedback (Embeddings)
- "Key points similarity: 0.75 (good coverage)"
- "Missing discussion of marshmallow experiment details"
- "Good understanding of main concept"

## Advantages Over GPT-Only

1. **Consistency**: Same errors = same deductions every time
2. **Precision**: Micro-level error detection like APEUni
3. **Speed**: Faster than GPT-4 API calls
4. **Reliability**: No "AI mood swings" or inconsistency
5. **Transparency**: Clear rules for each deduction

## Fallback System

If hybrid scoring fails, system automatically falls back to original GPT method, ensuring no downtime.

## Performance Notes

- First run downloads models (~500MB total)
- Subsequent runs are fast (< 2 seconds per summary)
- Memory usage: ~1GB for all models loaded