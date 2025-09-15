# PrepSmart API - Python Edition

AI-powered scoring and analysis API for PrepSmart PTE practice platform using Python, FastAPI, and advanced audio processing.

## 🏗️ Architecture

```
PrepSmart-API/
├── app/
│   ├── api/             # FastAPI route definitions
│   │   ├── v1/          # API version 1 endpoints
│   │   └── middleware/  # Request/response middleware
│   ├── core/            # Core configuration and settings
│   ├── models/          # Pydantic models and schemas
│   ├── services/        # Business logic services
│   │   ├── audio/       # Audio processing (Whisper, analysis)
│   │   ├── scoring/     # GPT-based scoring engines
│   │   └── wordpress/   # WordPress integration
│   └── utils/           # Helper functions and utilities
├── tests/               # Unit and integration tests
├── docs/                # API documentation
├── config/              # Configuration files
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
└── main.py             # FastAPI application entry point
```

## 🎯 **Why Python for PrepSmart API?**

### **Audio Processing Advantages:**
- **Whisper Integration**: Native OpenAI Whisper for transcription + confidence scores
- **Speech Analysis**: Pause detection, speaking rate, pronunciation analysis
- **Audio Features**: Librosa for pitch, tone, rhythm analysis
- **Real-time Processing**: WebRTC VAD for voice activity detection

### **AI/ML Capabilities:**
- **OpenAI Python SDK**: Direct GPT-4 integration with structured outputs
- **Audio ML Models**: Hugging Face transformers for pronunciation scoring
- **Custom Models**: TensorFlow/PyTorch for specialized PTE scoring
- **NLP Processing**: NLTK/spaCy for text analysis

## 🚀 **Technology Stack**

- **Framework**: FastAPI (high performance, auto docs)
- **Audio Processing**: Whisper, librosa, pyaudio, webrtcvad
- **AI Integration**: OpenAI Python SDK, transformers
- **Database**: SQLAlchemy + PostgreSQL/MongoDB
- **File Storage**: boto3 (AWS S3) or cloudflare-python
- **Authentication**: JWT with python-jose
- **Deployment**: Docker + Gunicorn/Uvicorn

## 🎤 **Audio Analysis Features**

### **Whisper Integration:**
```python
import whisper
import openai

# Transcription with confidence and timing
def analyze_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(
        audio_file,
        word_timestamps=True,
        condition_on_previous_text=False
    )
    
    return {
        "transcript": result["text"],
        "confidence": result.get("confidence", 0),
        "word_timings": result["segments"],
        "speaking_rate": calculate_speaking_rate(result),
        "pause_analysis": detect_pauses(result),
        "pronunciation_score": score_pronunciation(result)
    }
```

### **Advanced Speech Metrics:**
- **Pause Detection**: Silence duration and frequency
- **Speaking Rate**: Words per minute calculation
- **Fluency Score**: Hesitation and filler word detection
- **Pronunciation**: Phoneme-level accuracy scoring
- **Confidence Levels**: Per-word confidence from Whisper
- **Prosody Analysis**: Pitch, stress, intonation patterns

## 🔄 **Scoring Flow Architecture**

```
WordPress Plugin → Python FastAPI → Whisper Analysis → GPT Scoring → Response
```

**Detailed Flow:**
1. **Audio Upload**: WordPress sends audio file + question context
2. **Audio Processing**: Whisper extracts transcript + speech features
3. **Context Enrichment**: Fetch question details from WordPress DB
4. **GPT Analysis**: Send structured prompt with audio features + text
5. **Score Calculation**: GPT returns detailed component scores
6. **Response**: JSON with scores + detailed feedback

## 📊 **API Endpoints Structure**

```python
# Speaking Tasks
POST /api/v1/score/speaking/read-aloud
POST /api/v1/score/speaking/repeat-sentence
POST /api/v1/score/speaking/describe-image
POST /api/v1/score/speaking/retell-lecture
POST /api/v1/score/speaking/answer-short-question

# Writing Tasks  
POST /api/v1/score/writing/summarize-written-text
POST /api/v1/score/writing/essay

# Listening Tasks
POST /api/v1/score/listening/summarize-spoken-text
POST /api/v1/score/listening/multiple-choice
POST /api/v1/score/listening/fill-blanks

# Reading Tasks
POST /api/v1/score/reading/multiple-choice
POST /api/v1/score/reading/reorder-paragraphs
POST /api/v1/score/reading/fill-blanks

# Utility Endpoints
GET  /api/v1/health
POST /api/v1/audio/analyze
GET  /api/v1/docs
```

## 🛠️ **Installation & Development**

```bash
# Clone repository
git clone https://github.com/username/PrepSmart-API.git
cd PrepSmart-API

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Configure your API keys and settings

# Run development server
uvicorn main:app --reload --port 8000

# API Documentation available at:
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

## 🐳 **Docker Deployment**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔒 **Security & Authentication**

- **JWT Tokens**: Integration with WordPress authentication
- **Rate Limiting**: Per-user and per-IP restrictions
- **File Validation**: Audio format and size checks
- **CORS**: Configured for WordPress origins
- **Input Sanitization**: Pydantic model validation

## 📈 **Performance Optimizations**

- **Async Processing**: FastAPI async endpoints
- **Audio Streaming**: Process audio files in chunks
- **Caching**: Redis for repeated analysis results
- **Background Tasks**: Celery for heavy processing
- **Model Loading**: Lazy loading of ML models

## 🌐 **Hostinger VPS Deployment**

```bash
# Server setup (Ubuntu 22.04)
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv nginx supervisor redis-server -y

# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev

# Deploy application
git clone https://github.com/username/PrepSmart-API.git
cd PrepSmart-API
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup systemd service
sudo cp deploy/prepsmart-api.service /etc/systemd/system/
sudo systemctl enable prepsmart-api
sudo systemctl start prepsmart-api

# Configure Nginx reverse proxy
sudo cp deploy/nginx.conf /etc/nginx/sites-available/prepsmart-api
sudo ln -s /etc/nginx/sites-available/prepsmart-api /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

This Python-based approach gives us much better audio processing capabilities with Whisper integration and detailed speech analysis features that are essential for accurate PTE scoring!

Ready to start with the specific task APIs whenever you are! 🚀