# VPS Deployment Guide for PrepSmart Enhanced Scoring System

## 1. Server Requirements

### Minimum Specifications:
- **CPU**: 4+ cores (8+ recommended for ML models)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 20GB+ available space
- **OS**: Ubuntu 20.04 LTS or newer
- **Python**: 3.8+ (3.10+ recommended)

## 2. Initial Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and essential packages
sudo apt install python3 python3-pip python3-venv git curl -y

# Install build tools for ML dependencies
sudo apt install build-essential python3-dev -y

# Create application user
sudo useradd -m -s /bin/bash prepsmart
sudo usermod -aG sudo prepsmart
```

## 3. Clone and Setup Application

```bash
# Switch to application user
sudo su - prepsmart

# Clone repository (replace with your GitHub repo URL)
git clone https://github.com/yourusername/PrepSmart-API.git
cd PrepSmart-API

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 4. Environment Configuration

```bash
# Create .env file
nano .env
```

Add the following content to `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

## 5. Install and Configure Nginx

```bash
# Install Nginx
sudo apt install nginx -y

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/prepsmart
```

Add this Nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

```bash
# Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/prepsmart /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 6. Create Systemd Service

```bash
# Create systemd service file
sudo nano /etc/systemd/system/prepsmart.service
```

Add this content:
```ini
[Unit]
Description=PrepSmart API
After=network.target

[Service]
Type=simple
User=prepsmart
WorkingDirectory=/home/prepsmart/PrepSmart-API
Environment=PATH=/home/prepsmart/PrepSmart-API/venv/bin
ExecStart=/home/prepsmart/PrepSmart-API/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable prepsmart
sudo systemctl start prepsmart

# Check service status
sudo systemctl status prepsmart
```

## 7. SSL Setup (Optional but Recommended)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d your-domain.com

# Auto-renewal setup (should be automatic)
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 8. Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## 9. Monitoring and Logs

```bash
# View application logs
sudo journalctl -u prepsmart -f

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Monitor system resources
htop
```

## 10. Frontend Integration

### API Endpoints:
- **Base URL**: `https://your-domain.com` or `http://your-ip:80`
- **Summarize Text**: `POST /api/v1/writing/summarize-written-text`

### Frontend Configuration:
Update your frontend's API base URL to point to your VPS:

```javascript
// In your frontend config
const API_BASE_URL = 'https://your-domain.com';

// Example API call
const response = await fetch(`${API_BASE_URL}/api/v1/writing/summarize-written-text`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question_title: "Environmental Conservation",
    reading_passage: "Your passage text here...",
    key_points: ["conservation", "future generations", "clean environment"],
    user_summary: "User's summary text here..."
  })
});
```

## 11. Performance Optimization

### For High Load:
```bash
# Increase worker processes in main.py
# Add to main.py: uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)

# Or use Gunicorn for production
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### Model Caching:
The enhanced scorer automatically caches models in memory. For better performance:
- Ensure adequate RAM (16GB+ recommended)
- Consider using Redis for session caching if needed

## 12. Backup Strategy

```bash
# Create backup script
nano ~/backup.sh
```

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/prepsmart/backups"
mkdir -p $BACKUP_DIR

# Backup application
tar -czf "$BACKUP_DIR/prepsmart_$DATE.tar.gz" -C /home/prepsmart PrepSmart-API

# Keep only last 7 backups
find $BACKUP_DIR -name "prepsmart_*.tar.gz" -mtime +7 -delete
```

```bash
chmod +x ~/backup.sh
# Add to crontab for daily backups
crontab -e
# Add: 0 2 * * * /home/prepsmart/backup.sh
```

## 13. Troubleshooting

### Common Issues:

1. **Service won't start**:
   ```bash
   sudo journalctl -u prepsmart -f
   # Check for missing dependencies or environment issues
   ```

2. **High memory usage**:
   ```bash
   # Monitor memory usage
   free -h
   # Restart service if needed
   sudo systemctl restart prepsmart
   ```

3. **API timeouts**:
   - Increase Nginx proxy timeout in `/etc/nginx/sites-available/prepsmart`
   - Add: `proxy_read_timeout 600s;`

4. **ML model loading errors**:
   ```bash
   # Reinstall transformers and torch
   source venv/bin/activate
   pip uninstall torch transformers sentence-transformers
   pip install torch transformers sentence-transformers
   ```

## 14. Deployment Commands Summary

```bash
# Quick deployment script
curl -sSL https://your-repo/deploy.sh | bash

# Or manual steps:
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart prepsmart
sudo systemctl restart nginx
```

## 15. Health Check

```bash
# Test API endpoint
curl -X GET "http://your-domain.com/health" 

# Test scoring endpoint
curl -X POST "http://your-domain.com/api/v1/writing/summarize-written-text" \
  -H "Content-Type: application/json" \
  -d '{"question_title":"Test","reading_passage":"Test passage","key_points":["test"],"user_summary":"Test summary"}'
```

---

**Note**: Replace `your-domain.com`, `your_openai_api_key_here`, and GitHub repository URL with your actual values.

**Security Reminder**: Never commit your `.env` file or expose your OpenAI API key. Use environment variables or secure secret management.