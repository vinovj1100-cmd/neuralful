# 🧠 Neural Fulfillment Platform v3.0

> **YESI v3.0** — Autonomous Warehouse Intelligence with Predictive Analytics, Neural Vision, and Quantum Routing

![Platform](https://img.shields.io/badge/Platform-Streamlit-FF4B4B?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## ✨ Features

### Core Operations
- 📦 **Inventory Management** with alias/template suggestions
- 📋 **Order Processing** with status workflows
- 🔍 **Discrepancy Auditor** with color-coded analysis
- 🔄 **Bulk Title Converter** (translation + template matching)
- 📄 **PDF Label Sequencer** (Smart Sort / Strict / WB Phone+Code)

### Neural Advanced Systems
- 🧠 **AI Vision Inspection** — Damage detection, hazmat classification
- 🗺️ **3D Warehouse Holo-Deck** — Interactive digital twin
- ⚡ **Quantum Route Optimizer** — TSP pick path with 2-opt
- 🎮 **Neural Command Center** — Natural language operations + gamification
- 🌱 **Eco-Logistics Tracker** — Carbon footprint & sustainability
- 🛡️ **Anomaly Sentinel** — Statistical process control
- 🔮 **Oracle Predictive Engine** — Double exponential smoothing forecasts

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/neural-fulfillment.git
cd neural-fulfillment

# 2. Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr libtesseract-dev libzbar0 poppler-utils libgl1-mesa-glx

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Run the application
streamlit run app_enhanced.py
```

Access at: `http://localhost:8501`

## 🔐 Default Login
- **Admin**: `admin` / `admin123`
- **Operator**: `operator` / `operator123`

> ⚠️ Change default passwords after first login!

## 📖 Documentation

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for:
- Docker deployment
- Streamlit Cloud setup
- Configuration options
- Troubleshooting

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Streamlit Frontend (app_enhanced.py)  │
├─────────────────────────────────────────┤
│  Neural Vision │ Quantum Routes │ Eco  │
│  Command AI    │ Oracle Forecast│ Holo │
├─────────────────────────────────────────┤
│  db.py │ memory.py │ sync.py           │
├─────────────────────────────────────────┤
│  SQLite (warehouse_neural.db)          │
└─────────────────────────────────────────┘
```

## 📦 File Structure

| File | Purpose |
|------|---------|
| `app_enhanced.py` | Main application (1,791 lines) |
| `db.py` | SQLite database operations |
| `memory.py` | Settings & alias management |
| `sync.py` | Offline sync queue |
| `ozone_wms_guardian/` | System health monitoring |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System dependencies (Streamlit Cloud) |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License — Free for commercial and personal use.
