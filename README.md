# SegFL Local System

A privacy-preserving behavioural segmentation system with a Federated Learning backend and a modern React dashboard.

## 🚀 How to Run (Recommended)

### 1. Streamlit Dashboard (Unified)
```bash
streamlit run streamlit_app.py
```
This is the modern, unified application containing both the ML logic and the dashboard.

## 🛠️ Alternative Setup (FastAPI + React)

### 1. Backend (FastAPI)
```bash
cd backend
python main.py
```

### 2. Frontend (React + Vite)
```bash
cd frontend
npm run dev
```
The frontend will run on `http://localhost:3000` (proxied to backend).

## 📊 Features
- **CSV Data Ingestion**: Supports flexible CSV structures with high-end behavioral heuristics.
- **Federated Training**: Local training with TAL Adapter and Global Bottleneck Autoencoder.
- **Differential Privacy**: DP-SGD noise addition for privacy budgets.
- **Interactive Dashboard**: Real-time training logs, Silhouette/DBI scores, and Cluster Semantic Profiles.
