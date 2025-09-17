# Read Aloud - Workflow

## Overview
Speaking task where students read a passage aloud (35-40 seconds).

## Workflow Steps

### 1. Input Processing
- **Audio File**: Student's recorded speech
- **Text Passage**: Original text to be read
- **Duration Check**: Should be 35-40 seconds

### 2. Content Analysis (5/5 points)
- **Word Recognition**: Accuracy of pronunciation
- **Content Coverage**: All words attempted
- **Omissions**: Missing words penalty
- **Substitutions**: Incorrect word replacements

### 3. Oral Fluency (5/5 points)
- **Reading Rate**: Appropriate speed (130-150 WPM)
- **Phrasing**: Natural word groupings
- **Hesitations**: Pauses and fillers
- **Rhythm**: Natural speech patterns

### 4. Pronunciation (5/5 points)
- **Vowel Sounds**: Accuracy of vowel pronunciation
- **Consonant Sounds**: Clear consonant articulation
- **Word Stress**: Correct syllable emphasis
- **Sentence Stress**: Emphasis on content words

## Technical Requirements
- **Audio Processing**: Speech-to-text conversion
- **Phonetic Analysis**: IPA comparison
- **Timing Analysis**: Rate calculation
- **Quality Assessment**: Audio clarity check

## API Endpoints
- `POST /read-aloud` - Audio analysis endpoint

## Scoring Scale
- **Total**: 15 points
- **Components**: Content (5) + Fluency (5) + Pronunciation (5)

## Status
🚧 **PENDING IMPLEMENTATION**