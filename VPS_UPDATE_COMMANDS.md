# VPS Update Commands

## 1. Connect to VPS and Navigate
```bash
# SSH into your VPS
ssh root@your-vps-ip

# Navigate to your project directory
cd /root/PrepSmart-API
```

## 2. Activate Virtual Environment
```bash
# Activate the virtual environment
source venv/bin/activate

# You should see (venv) in your prompt
```

## 3. Pull Latest Changes
```bash
# Pull latest code from GitHub
git pull origin main
```

## 4. Update Dependencies
```bash
# Install/update all requirements
pip install -r requirements.txt

# Download NLTK data (if not already done)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model (if not already done)
python -m spacy download en_core_web_sm
```

## 5. Restart Services
```bash
# If using systemd service
sudo systemctl restart prepsmart
sudo systemctl status prepsmart

# OR if running manually, stop current process and restart
# Press Ctrl+C to stop current process, then:
python main.py
```

## 6. Check if Running
```bash
# Check if API is responding
curl http://localhost:8000/health

# OR check what's running on port 8000
sudo netstat -tlnp | grep :8000
```

## Quick Update Script
```bash
#!/bin/bash
cd /root/PrepSmart-API
source venv/bin/activate
git pull origin main
pip install -r requirements.txt
sudo systemctl restart prepsmart
echo "✅ Update complete!"
```

## If You Get Errors:
```bash
# Check service logs
sudo journalctl -u prepsmart -f

# Check if virtual environment is active
which python
# Should show: /root/PrepSmart-API/venv/bin/python

# Reinstall problematic packages
pip uninstall torch transformers sentence-transformers
pip install torch transformers sentence-transformers
```