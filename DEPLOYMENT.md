# Deployment Guide

## Edge Device Deployment

### Raspberry Pi 4 Setup

**Requirements:**
- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi OS (64-bit)
- Python 3.9+
- Camera module or USB microscope

**Installation:**

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y python3-pip python3-venv git
sudo apt-get install -y libopencv-dev python3-opencv

# Clone repository
git clone <your-repo-url>
cd Dashboard

# Setup backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure for Pi
export DEVICE=cpu
export INFERENCE_WORKERS=2

# Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Performance Optimization:**
- Use TFLite model (5-7 FPS)
- Reduce FPS to 4-6
- Lower JPEG quality to 70-80
- Disable unnecessary services

### Coral Edge TPU Setup

**Installation:**

```bash
# Install Edge TPU runtime
bash scripts/install_edgetpu.sh

# Convert model to Edge TPU format
# (Requires TFLite model first)
edgetpu_compiler models/plankton_model_quant.tflite

# Enable Edge TPU in config
export USE_EDGE_TPU=true

# Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Performance:** 30-60 FPS with Edge TPU

## Cloud Deployment

### Docker Deployment

**Production deployment with Docker Compose:**

```bash
# Set environment variables
export API_KEY=your-secure-api-key
export MODEL_VERSION=1.0.0
export DEVICE=cpu
export INFERENCE_WORKERS=4

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

### Kubernetes Deployment

**Create deployment manifests:**

```yaml
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: plankton-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: plankton-backend
  template:
    metadata:
      labels:
        app: plankton-backend
    spec:
      containers:
      - name: backend
        image: your-registry/plankton-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: plankton-secrets
              key: api-key
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
```

## Security Hardening

### SSL/TLS Setup

```bash
# Install certbot
sudo apt-get install certbot

# Get certificate
sudo certbot certonly --standalone -d yourdomain.com

# Update nginx config to use SSL
```

### API Key Management

```bash
# Generate secure API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set in environment
export API_KEY=<generated-key>
```

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

## Monitoring

### System Monitoring

```bash
# Install monitoring tools
pip install prometheus-client

# Add to backend
from prometheus_client import start_http_server, Counter, Histogram

# Expose metrics
start_http_server(9090)
```

### Log Management

```bash
# Centralized logging with rsyslog
sudo apt-get install rsyslog

# Configure log rotation
sudo nano /etc/logrotate.d/plankton
```

## Backup & Recovery

### Backup Strategy

```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz backend/models/

# Backup data
tar -czf data-backup-$(date +%Y%m%d).tar.gz backend/data/

# Automated backup script
0 2 * * * /path/to/backup.sh
```

### Recovery

```bash
# Restore models
tar -xzf models-backup-YYYYMMDD.tar.gz -C backend/

# Restore data
tar -xzf data-backup-YYYYMMDD.tar.gz -C backend/
```

## Scaling

### Horizontal Scaling

```bash
# Use load balancer (nginx)
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

### Vertical Scaling

```bash
# Increase workers
export INFERENCE_WORKERS=8

# Allocate more memory
docker run -m 8g plankton-backend
```

## Troubleshooting

### High Memory Usage

```bash
# Monitor memory
watch -n 1 free -h

# Reduce workers
export INFERENCE_WORKERS=2
```

### Connection Issues

```bash
# Check port availability
netstat -tulpn | grep 8000

# Test WebSocket
wscat -c ws://localhost:8000/ws/predict?token=YOUR_KEY
```
