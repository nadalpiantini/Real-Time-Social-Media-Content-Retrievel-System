# Deployment Guide

## Overview

This guide covers deployment options for the Real-Time LinkedIn Content Retrieval System, from local development to production environments.

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Docker (for Qdrant vector database)
- Git

### Setup Steps

```bash
# 1. Clone repository
git clone <repository-url>
cd Real-Time-Social-Media-Content-Retrievel-System

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Qdrant database
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# 5. Run application
streamlit run app.py
```

## Production Deployment

### Option 1: Streamlit Cloud

#### Advantages
- ✅ Free hosting
- ✅ Automatic deployments from GitHub
- ✅ Built-in SSL/HTTPS
- ✅ No server management

#### Setup Steps

1. **Prepare Repository**
   ```bash
   # Ensure requirements.txt is optimized
   cp requirements.txt requirements_streamlit_cloud.txt
   
   # Add secrets configuration
   mkdir .streamlit/secrets.toml
   ```

2. **Configure Secrets** (`.streamlit/secrets.toml`)
   ```toml
   [supabase]
   url = "your-supabase-url"
   key = "your-supabase-key"
   
   [qdrant]
   url = "your-qdrant-cloud-url"
   api_key = "your-qdrant-api-key"
   ```

3. **Deploy to Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Connect GitHub repository
   - Configure deployment settings
   - Add secrets via UI

#### Limitations
- Limited to 1GB RAM
- No persistent storage (use cloud databases)
- Limited CPU time
- No custom Docker containers

### Option 2: Docker Deployment

#### Multi-Service Setup

**docker-compose.yml**
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant
    volumes:
      - ./data:/app/data
      - ./models_cache:/app/models_cache

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
```

**Dockerfile Optimization**
```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Performance optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false
ENV TF_CPP_MIN_LOG_LEVEL=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Option 3: Cloud Platform Deployment

#### AWS Deployment

**Using AWS Fargate**
```bash
# 1. Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker build -t linkedin-content-retrieval .
docker tag linkedin-content-retrieval:latest <account>.dkr.ecr.us-east-1.amazonaws.com/linkedin-content-retrieval:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/linkedin-content-retrieval:latest

# 2. Create ECS task definition
# 3. Deploy with Fargate service
```

**Environment Variables**
```bash
QDRANT_URL=https://your-qdrant-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

#### Google Cloud Platform

**Using Cloud Run**
```bash
# 1. Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/linkedin-content-retrieval
gcloud run deploy --image gcr.io/PROJECT_ID/linkedin-content-retrieval --platform managed
```

#### Heroku Deployment

**Procfile**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**heroku.yml**
```yaml
build:
  docker:
    web: Dockerfile
```

## Database Configuration

### Qdrant Vector Database

#### Local Development
```bash
# In-memory mode (no persistence)
# Automatic fallback in code

# Docker persistent mode
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

#### Production Options

1. **Qdrant Cloud** (Recommended)
   - Managed service
   - Global distribution
   - Automatic scaling
   - Built-in monitoring

2. **Self-hosted Qdrant**
   ```bash
   # Production Docker setup
   docker run -d \
     --name qdrant-prod \
     -p 6333:6333 \
     -v /opt/qdrant/storage:/qdrant/storage \
     -e QDRANT__SERVICE__HTTP_PORT=6333 \
     -e QDRANT__SERVICE__GRPC_PORT=6334 \
     qdrant/qdrant
   ```

### Supabase (Optional)

#### Setup
1. Create Supabase project
2. Run schema: `supabase_schema.sql`
3. Configure environment variables
4. Test connection

## Performance Optimization

### Production Checklist

- [ ] Enable Streamlit production mode
- [ ] Configure proper logging levels
- [ ] Set up monitoring and alerting
- [ ] Implement health checks
- [ ] Configure auto-scaling
- [ ] Set up backup procedures
- [ ] Enable HTTPS/SSL
- [ ] Configure CDN if needed

### Resource Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB (2GB for ML models + 2GB for application)
- **Storage**: 10GB (5GB for models + 5GB for data)
- **Network**: 1 Gbps for model downloads

#### Recommended Production
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Load Balancer**: For high availability
- **Auto-scaling**: Based on CPU/memory usage

## Security Considerations

### Application Security
- Environment variables for secrets
- Input validation and sanitization
- Rate limiting for API endpoints
- HTTPS enforcement
- CORS configuration

### Data Protection
- Encrypt data at rest
- Secure vector database access
- Regular security updates
- Access logging and monitoring

### Configuration Example

**.env.production**
```bash
# Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-secure-api-key

# Optional: Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key

# Performance
EMBEDDING_MODEL_DEVICE=cpu
USE_MODEL_CACHE=true
LOG_LEVEL=WARNING

# Security
ALLOWED_ORIGINS=["https://yourdomain.com"]
```

## Monitoring and Maintenance

### Health Monitoring
```bash
# Check application health
curl http://localhost:8501/_stcore/health

# Check Qdrant health
curl http://localhost:6333/health

# Monitor resource usage
docker stats
```

### Regular Maintenance
```bash
# Weekly maintenance
python maintenance.py --full

# Log cleanup
python maintenance.py --logs-only

# Check system status
python maintenance.py --check-only
```

### Backup Procedures

1. **Vector Database Backup**
   ```bash
   # Qdrant backup
   curl -X POST "http://localhost:6333/collections/posts/snapshots"
   ```

2. **Data Files Backup**
   ```bash
   # Backup data directory
   tar -czf backup_$(date +%Y%m%d).tar.gz data/
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch sizes
   - Enable model quantization
   - Add swap space

2. **Slow Performance**
   - Check CPU usage
   - Monitor network latency
   - Verify model caching

3. **Connection Issues**
   - Verify network connectivity
   - Check firewall settings
   - Validate environment variables

### Debugging Commands
```bash
# Check logs
docker logs container_name

# Monitor resources
htop
iotop

# Test connections
telnet qdrant-host 6333
curl -I https://api.endpoint
```

## Scaling Strategies

### Horizontal Scaling
- Multiple app instances behind load balancer
- Distributed vector database clusters
- CDN for static assets

### Vertical Scaling
- Increase CPU/RAM allocation
- Optimize model loading
- Implement caching layers

### Cost Optimization
- Use spot instances for non-critical workloads
- Implement intelligent caching
- Monitor and optimize resource usage
- Consider serverless options for low traffic