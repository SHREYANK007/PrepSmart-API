# Repeat Sentence - Workflow

## Overview
Speaking task where students listen and repeat a sentence (10-12 seconds).

## Workflow Steps

### 1. Input Processing
- **Audio File**: Student's repeated sentence
- **Original Audio**: Reference sentence
- **Text Reference**: Original sentence text

### 2. Content Analysis (3/3 points)
- **Word Accuracy**: Exact word matching
- **Sequence Order**: Correct word order
- **Completeness**: All words included
- **No Additions**: No extra words

### 3. Oral Fluency (2/2 points)
- **Natural Pace**: Appropriate speaking speed
- **Smooth Delivery**: No excessive hesitations
- **Rhythm**: Natural sentence rhythm
- **Stress Patterns**: Correct emphasis

### 4. Pronunciation (2/2 points)
- **Sound Accuracy**: Clear articulation
- **Intonation**: Rising/falling patterns
- **Connected Speech**: Natural linking
- **Overall Clarity**: Understandability

## Technical Requirements
- **Speech Recognition**: Audio-to-text conversion
- **Phonetic Matching**: Sound comparison
- **Prosody Analysis**: Stress and intonation
- **Timing Analysis**: Pace evaluation

## API Endpoints
- `POST /repeat-sentence` - Audio analysis endpoint

## Scoring Scale
- **Total**: 7 points
- **Components**: Content (3) + Fluency (2) + Pronunciation (2)

## Status
🚧 **PENDING IMPLEMENTATION**