# Answer Short Question - Workflow

## Overview
Speaking task where students answer a short factual question (10 seconds).

## Workflow Steps

### 1. Input Processing
- **Audio File**: Student's answer
- **Question Audio**: Original question
- **Expected Answer**: Reference response
- **Duration Check**: Should be under 10 seconds

### 2. Content Analysis (1/1 point)
- **Accuracy**: Correct factual answer
- **Relevance**: Response addresses question
- **Completeness**: Sufficient information provided
- **Appropriateness**: Suitable response type

### 3. Response Evaluation
- **Word Recognition**: Clear word identification
- **Answer Matching**: Comparison with expected responses
- **Semantic Understanding**: Meaning comprehension
- **Context Appropriateness**: Suitable for question type

## Technical Requirements
- **Speech Recognition**: Audio-to-text conversion
- **Answer Validation**: Correct response checking
- **Semantic Matching**: Meaning comparison
- **Quick Processing**: Fast evaluation needed

## API Endpoints
- `POST /answer-short-question` - Short answer analysis

## Scoring Scale
- **Total**: 1 point
- **Binary**: Correct (1) or Incorrect (0)

## Common Question Types
- **Factual**: Who, what, when, where questions
- **Definitional**: What is/are questions
- **Numerical**: How many, what time questions
- **Categorical**: Types, categories, classifications

## Status
🚧 **PENDING IMPLEMENTATION**