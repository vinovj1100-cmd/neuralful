# 🚀 Neural Fulfillment Platform v3.0 — Deployment Guide

Complete guide to deploy the **Neural Fulfillment Platform** (YESI v3.0) on Streamlit Cloud, local servers, or Docker containers.

---

## 📁 Project Structure

```
neural-fulfillment/
├── app_enhanced.py              # Main Streamlit application (1,791 lines)
├── db.py                        # SQLite database module
├── memory.py                    # Settings & alias management
├── sync.py                      # Offline sync queue handler
├── requirements.txt             # Python dependencies
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── ozone_wms_guardian/
│   ├── __init__.py
│   ├── guardian.py              # System health monitoring
│   └── admin/
│       ├── __init__.py
│       └── dashboard.py         # Guardian dashboard renderer
└── README.md
```

---

## 🛠️ Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **OS**: Linux, macOS, or Windows (WSL2 recommended for Windows)
- **Memory**: 4GB RAM minimum (8GB+ recommended for neural vision)
- **Storage**: 2GB free space

### System Dependencies (OS-Level)

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev libzbar0 poppler-utils libgl1-mesa-glx
```

#### macOS
```bash
brew install tesseract zbar poppler
```

#### Windows
1. Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
2. Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases
3. Add both to your system PATH

---

## 📦 Installation

### 1. Create Project Directory
```bash
mkdir neural-fulfillment
cd neural-fulfillment
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Tesseract Installation
```bash
tesseract --version
# Should output version info (e.g., tesseract 5.x.x)
```

---

## ⚙️ Configuration

### Streamlit Config (`.streamlit/config.toml`)
Create the `.streamlit` directory and config file:

```toml
[theme]
base = "dark"
primaryColor = "#64ffda"
backgroundColor = "#0a192f"
secondaryBackgroundColor = "#112240"
textColor = "#ccd6f6"
font = "monospace"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 50

[browser]
gatherUsageStats = false
```

### Environment Variables (Optional)
Create a `.env` file for production:

```bash
# Database path override
DB_PATH=warehouse_neural.db

# Tesseract path (if not in PATH)
TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Sync endpoint (for future cloud integration)
SYNC_API_URL=https://your-api.com/sync
SYNC_API_KEY=your_api_key_here
```

---

## 🚀 Running the Application

### Local Development
```bash
streamlit run app_enhanced.py
```

Access at: `http://localhost:8501`

### Production Mode
```bash
streamlit run app_enhanced.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
```

---

## ☁️ Streamlit Cloud Deployment

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial Neural Fulfillment Platform deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/neural-fulfillment.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repository
4. Set **Main file path**: `app_enhanced.py`
5. Click **Deploy**

### ⚠️ Streamlit Cloud Limitations
- **Tesseract OCR**: May not be pre-installed. Add a `packages.txt` file:
  ```
  tesseract-ocr
  libtesseract-dev
  libzbar0
  poppler-utils
  libgl1
  ```
- **File uploads**: Max 200MB per file (Streamlit Cloud limit)
- **Session state**: Resets on app restart — use database for persistence

---

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libzbar0 \
    poppler-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
ENTRYPOINT ["streamlit", "run", "app_enhanced.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build & Run
```bash
# Build image
docker build -t neural-fulfillment .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data neural-fulfillment

# With environment variables
docker run -p 8501:8501 --env-file .env neural-fulfillment
```

### Docker Compose
```yaml
version: '3.8'

services:
  neural-fulfillment:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./warehouse_neural.db:/app/warehouse_neural.db
    environment:
      - DB_PATH=/app/warehouse_neural.db
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker-compose up -d
```

---

## 🔐 Default Credentials

| Username | Password  | Role     |
|----------|-----------|----------|
| admin    | admin123  | Admin    |
| operator | operator123 | Operator |

> **⚠️ SECURITY WARNING**: Change default passwords immediately after first login!

---

## 📋 Feature Checklist

### Core Features (All Tabs)
- [x] **Dashboard** — Real-time metrics, anomaly alerts, predictive forecast charts
- [x] **Inventory** — CRUD operations with alias/template suggestions
- [x] **Orders** — Order creation, status updates, SKU tracking
- [x] **Auditor** — Discrepancy analysis with color-coded results
- [x] **Bulk Convert** — Translation + template matching for product titles
- [x] **PDF Sequencer** — Smart Sort, Strict Rearrange, WB Phone+Code Matcher
- [x] **Templates** — Raw-to-standard title mapping database
- [x] **Memory** — Preferences, aliases, recent activity

### Neural Advanced Features
- [x] **Neural Ops** — AI vision inspection (damage/hazmat detection)
- [x] **Holo-Deck** — 3D warehouse digital twin (Three.js)
- [x] **Quantum Routes** — TSP pick path optimization with SVG heatmaps
- [x] **Command Center** — Natural language command interface + gamification
- [x] **Eco-Logistics** — Carbon footprint tracking & sustainability scoring
- [x] **Admin Panel** — User management, SIM database, Guardian diagnostics

---

## 🧪 Testing

### Run Syntax Check
```bash
python -m py_compile app_enhanced.py
```

### Verify Database
```bash
python -c "from db import init_db; init_db(); print('Database initialized successfully')"
```

### Test Guardian Module
```bash
python -c "from ozone_wms_guardian import Guardian, GuardianConfig; g = Guardian(GuardianConfig()); g.start(); print('Guardian initialized')"
```

---

## 🔄 Updating the Application

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart Streamlit
pkill -f "streamlit run"
streamlit run app_enhanced.py
```

---

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| `tesseract not found` | Install Tesseract OCR system package |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `cv2.error` | Install `libgl1-mesa-glx` (Linux) or update OpenCV |
| Database locked | Close other instances; SQLite allows one writer |
| 3D Holo-Deck not rendering | Enable JavaScript; check browser console |
| PDF processing slow | Reduce DPI in `convert_from_bytes(dpi=150)` |
| Memory errors | Reduce `maxlen` in deque configurations |

---

## 📄 License

MIT License — Free for commercial and personal use.

---

**Built with ❤️ using Streamlit, OpenCV, and Neural Intelligence**
