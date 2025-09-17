# PrepSmart API - Workflow Documentation

## Overview
This folder contains detailed workflow documentation for each PTE task supported by the PrepSmart API.

## Task Workflows

### ✅ Writing Tasks
- **[Summarize Written Text](summarize_written_text.md)** - ✅ **COMPLETE & LOCKED**
  - 3-layer hybrid scoring with GPT-4o expert
  - Grammar, Vocabulary, Content analysis
  - 7-point scoring system
- **[Write Essay](write_essay.md)** - 🚧 **PENDING**
  - 200-300 words essay scoring
  - 10-point scoring system

### 🚧 Speaking Tasks
- **[Read Aloud](read_aloud.md)** - 🚧 **PENDING**
  - Content, Fluency, Pronunciation analysis
  - 15-point scoring system
- **[Repeat Sentence](repeat_sentence.md)** - 🚧 **PENDING**
  - Audio repetition accuracy
  - 7-point scoring system
- **[Describe Image](describe_image.md)** - 🚧 **PENDING**
  - Visual description analysis
  - 15-point scoring system
- **[Retell Lecture](retell_lecture.md)** - 🚧 **PENDING**
  - Lecture content retelling
  - 15-point scoring system
- **[Answer Short Question](answer_short_question.md)** - 🚧 **PENDING**
  - Factual question responses
  - 1-point scoring system

### 📚 Reading Tasks
- **Reading Comprehension** - 🚧 **PENDING**
- **Fill in the Blanks** - 🚧 **PENDING**

### 🎧 Listening Tasks
- **Summarize Spoken Text** - 🚧 **PENDING**
- **Fill in the Blanks (Listening)** - 🚧 **PENDING**

## Implementation Status

| Task | Status | Scorer Type | API Endpoint |
|------|--------|-------------|--------------|
| Summarize Written Text | ✅ Complete | 3-Layer Hybrid + GPT | `/summarize-text` |
| Write Essay | 🚧 Pending | - | `/write-essay` |
| Read Aloud | 🚧 Pending | - | `/read-aloud` |
| Repeat Sentence | 🚧 Pending | - | `/repeat-sentence` |
| Describe Image | 🚧 Pending | - | `/describe-image` |
| Retell Lecture | 🚧 Pending | - | `/retell-lecture` |
| Answer Short Question | 🚧 Pending | - | `/answer-short-question` |

## Architecture Notes

### Completed: Summarize Written Text
- **3-Layer System**: Grammar → Vocabulary → Content
- **ML Models**: GECToR, LanguageTool, Sentence-BERT, spaCy
- **GPT Integration**: Ultimate language expert with 100+ point analysis
- **Scoring**: Percentage-based content thresholds
- **Response Mapping**: Frontend-compatible suggestions

### Next Priorities
1. **Write Essay** - Similar to SWT but longer form
2. **Speaking Tasks** - Require audio processing capabilities
3. **Reading/Listening** - Different analysis approaches needed

## Development Guidelines
- Each workflow file contains complete implementation specifications
- Follow the established 3-layer + GPT pattern where applicable
- Maintain consistent API response formats
- Include comprehensive error handling and logging
- Document all scoring thresholds and algorithms