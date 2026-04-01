# 🔬 ImageClassifier App

AI-powered image classification using **ResNet-50** pretrained on **ImageNet** (1000 classes).
Built with **FastAPI** (backend) + **React.js** (frontend).
Framework: **Keras / TensorFlow**

---

## Features

- 🌙 **Dark mode** by default with polished UI
- 📁 **Drag & Drop** or browse image upload (PNG, JPG, WEBP, GIF)
- 🤖 **Top-5 predictions** with animated confidence bars
- 🍌 **Demo mode** — try with a banana image in one click
- 📄 **PDF Report** download — styled report with image thumbnail + predictions table
- 🔗 **Share** button — Web Share API with clipboard fallback
- ⚡ **Fast inference** — < 1 second with ResNet-50 via Keras

---

## Project Structure

```
image-classifier/
├── backend/
│   ├── main.py              # FastAPI app — /classify & /report endpoints
│   └── requirements.txt     # Python dependencies (TensorFlow, FastAPI …)
└── frontend/
    ├── public/index.html
    ├── src/
    │   ├── index.js
    │   ├── App.jsx          # Full React UI
    │   └── App.css          # Dark mode design system
    └── package.json
```

---

## Quick Start

### 1. Backend (FastAPI + Keras)

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

API → **http://localhost:8000**
Interactive docs → **http://localhost:8000/docs**

> **Note:** On first run, Keras downloads the ResNet-50 ImageNet weights (~100 MB) automatically.

---

### 2. Frontend (React)

```bash
cd frontend
npm install
npm start
```

Opens at **http://localhost:3000** and proxies API calls to `localhost:8000`.

---

## API Reference

| Method | Endpoint    | Description                                |
|--------|-------------|--------------------------------------------|
| GET    | `/`         | Health check + framework/model info        |
| POST   | `/classify` | Upload image → JSON with top-5 predictions |
| POST   | `/report`   | Upload image → download styled PDF report  |

### Example `/classify` response

```json
{
  "filename": "banana.jpg",
  "framework": "TensorFlow / Keras",
  "model": "ResNet-50",
  "timestamp": "2024-12-01T14:32:10.123456",
  "predictions": [
    { "rank": 1, "label": "Banana",      "confidence": 97.43, "class_id": "n07753592" },
    { "rank": 2, "label": "Lemon",       "confidence": 1.12,  "class_id": "n07749582" },
    { "rank": 3, "label": "Pineapple",   "confidence": 0.61,  "class_id": "n07753275" },
    { "rank": 4, "label": "Orange",      "confidence": 0.44,  "class_id": "n07747607" },
    { "rank": 5, "label": "Strawberry",  "confidence": 0.18,  "class_id": "n07745940" }
  ]
}
```

---

## Tech Stack

| Layer     | Technology                       |
|-----------|----------------------------------|
| Model     | ResNet-50 (`tf.keras.applications`) |
| Backend   | FastAPI + Uvicorn                |
| PDF Gen   | ReportLab                        |
| Frontend  | React 18 + Axios                 |
| Styling   | Custom CSS (dark mode)           |
| Dataset   | ImageNet (1000 classes)          |

---

## Confidence Score Legend

| Range  | Colour   | Label  |
|--------|----------|--------|
| ≥ 60%  | 🟢 Green | High   |
| 30–59% | 🟡 Amber | Medium |
| < 30%  | 🔴 Red   | Low    |
