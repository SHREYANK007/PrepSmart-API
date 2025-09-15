# PrepSmart API - Hostinger VPS Deployment Guide

## Prerequisites
- Hostinger KVM 4 VPS access
- SSH client (PuTTY for Windows or Terminal for Mac/Linux)
- Your VPS IP address and root credentials

## Step 1: Connect to Your VPS

```bash
ssh root@YOUR_VPS_IP
```

## Step 2: Update System and Install Dependencies

```bash
# Update system packages
apt update && apt upgrade -y

# Install Python 3.11 and pip
apt install python3.11 python3.11-pip python3.11-venv -y

# Install Redis (for caching and rate limiting)
apt install redis-server -y

# Install Nginx (for reverse proxy)
apt install nginx -y

# Install Git
apt install git -y

# Install system dependencies for audio processing
apt install ffmpeg libsndfile1 -y
```

## Step 3: Create Application Directory and User

```bash
# Create application directory
mkdir -p /var/www/prepsmart-api
cd /var/www/prepsmart-api

# Create a dedicated user (optional but recommended)
adduser prepsmart --disabled-password --gecos ""
chown -R prepsmart:prepsmart /var/www/prepsmart-api
```

## Step 4: Clone and Setup Application

```bash
# Clone your repository (or upload files via SCP)
git clone https://github.com/yourusername/PrepSmart-API.git .
# OR upload files manually to /var/www/prepsmart-api/

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs uploads models
chmod 755 logs uploads models
```

## Step 5: Configure Environment

```bash
# Copy production environment file
cp .env.production .env

# Edit the .env file with your specific settings
nano .env
```

**Important: Update these values in your .env file:**
- `OPENAI_API_KEY=` (your actual OpenAI API key)
- `SECRET_KEY=` (generate a random 50+ character string)
- `HOST=0.0.0.0` (to accept external connections)
- `CORS_ORIGINS=https://pte.prepsmart.au` (your WordPress domain)

## Step 6: Test the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Test run the application
python main.py

# Check if it's running (should see "Application startup complete")
# Press Ctrl+C to stop
```

## Step 7: Create Systemd Service

```bash
# Create service file
nano /etc/systemd/system/prepsmart-api.service
```

**Add this content to the service file:**

```ini
[Unit]
Description=PrepSmart API Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/prepsmart-api
Environment=PATH=/var/www/prepsmart-api/venv/bin
ExecStart=/var/www/prepsmart-api/venv/bin/python main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start the service
systemctl daemon-reload
systemctl enable prepsmart-api
systemctl start prepsmart-api

# Check service status
systemctl status prepsmart-api
```

## Step 8: Configure Nginx Reverse Proxy

```bash
# Create Nginx configuration
nano /etc/nginx/sites-available/prepsmart-api
```

**Add this content:**

```nginx
server {
    listen 80;
    server_name YOUR_VPS_IP;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

```bash
# Enable the site
ln -s /etc/nginx/sites-available/prepsmart-api /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default

# Test and reload Nginx
nginx -t
systemctl restart nginx
```

## Step 9: Configure Firewall

```bash
# Allow HTTP and SSH
ufw allow 22
ufw allow 80
ufw allow 443
ufw enable
```

## Step 10: Test Your API

```bash
# Test from the VPS itself
curl http://localhost:8000/health

# Test from external (replace with your VPS IP)
curl http://YOUR_VPS_IP/health
```

## Step 11: Update Frontend Configuration

Once your API is running, update your WordPress site's API configuration:

In your `assets/js/api-config.js` file, change:

```javascript
baseURL: 'http://YOUR_VPS_IP', // Replace with your actual VPS IP
```

## Useful Commands for Management

```bash
# Check service status
systemctl status prepsmart-api

# View service logs
journalctl -u prepsmart-api -f

# Restart service
systemctl restart prepsmart-api

# Check if port 8000 is listening
netstat -tlnp | grep :8000

# Check Nginx status
systemctl status nginx

# View API logs
tail -f /var/www/prepsmart-api/logs/api.log
```

## Troubleshooting

1. **Service won't start**: Check logs with `journalctl -u prepsmart-api`
2. **Port not accessible**: Check firewall settings and Nginx configuration
3. **CORS errors**: Ensure CORS_ORIGINS in .env matches your WordPress domain exactly
4. **OpenAI errors**: Verify your API key is correct and has sufficient credits

## Security Notes

- Never commit your `.env` file to version control
- Use strong passwords and keep your system updated
- Consider setting up SSL certificates for HTTPS
- Monitor your OpenAI API usage to prevent unexpected charges

Your API should now be accessible at `http://YOUR_VPS_IP/` and ready to receive requests from your WordPress site at `https://pte.prepsmart.au/`