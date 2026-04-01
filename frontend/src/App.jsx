import React, { useState, useCallback, useRef } from "react";
import axios from "axios";

// ─── Config ──────────────────────────────────────────────────────────────────
const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Banana demo placeholder (nano banana reference)
const DEMO_IMAGE_URL =
  "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=600&q=80";

// ─── Utilities ────────────────────────────────────────────────────────────────
const getConfidenceColor = (conf) => {
  if (conf >= 60) return "var(--green)";
  if (conf >= 30) return "var(--amber)";
  return "var(--red)";
};

const getConfidenceLabel = (conf) => {
  if (conf >= 60) return "High";
  if (conf >= 30) return "Medium";
  return "Low";
};

const formatTime = (iso) => {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
};

// ─── Header ───────────────────────────────────────────────────────────────────
function Header({ onShare, onDownload, hasPredictions, shareLabel }) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="brand">
          <div className="brand-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
              <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
              <line x1="12" y1="22.08" x2="12" y2="12" />
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-name">ImageClassifier</span>
            <span className="brand-tag">Finetuned Model · Keras / TensorFlow</span>
          </div>
        </div>

        <div className="header-actions">
          <button className="btn btn-ghost" onClick={onShare}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
              <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
            </svg>
            <span>{shareLabel}</span>
          </button>

          {hasPredictions && (
            <button className="btn btn-primary" onClick={onDownload}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              <span>Download Report</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
 
// ─── Upload Zone ──────────────────────────────────────────────────────────────
function UploadZone({ onFile, isDragging, setIsDragging, image, loading }) {
  const inputRef = useRef(null);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onFile(file);
    },
    [onFile, setIsDragging]
  );

  return (
    <div
      className={`drop-zone ${isDragging ? "dragging" : ""} ${image ? "has-image" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => !image && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />

      {image ? (
        <div className="image-preview-wrap">
          <img src={image} alt="Preview" className="preview-img" />
          {loading && (
            <div className="preview-overlay">
              <div className="spinner-large" />
              <span>Analyzing…</span>
            </div>
          )}
          {!loading && (
            <div className="preview-hover-overlay">
              <button
                className="btn btn-ghost small"
                onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                Change Image
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="drop-empty">
          <div className="drop-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
          </div>
          <p className="drop-title">Drop your image here</p>
          <p className="drop-sub">Any of: Sunflower, Rose, Tulip, Daisy, Daffodil</p>
          <p className="drop-sub">PNG, JPG, WEBP, GIF supported</p>
          <div className="drop-divider"><span>or</span></div>
          <button
            className="btn btn-primary"
            onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
          >
            Browse Files
          </button>
        </div>
      )}
    </div>
  );
}


// ─── Prediction Bar ───────────────────────────────────────────────────────────
function PredictionBar({ pred, index, animate }) {
  // Parse confidence value (remove '%' if present and convert to number)
  const pred_str = pred.confidence.replace("%", "");
  // console.log(typeof pred_str === 'string');

  const confValue = typeof pred_str === 'string' 
    ? parseFloat(pred_str) 
    : pred.confidence;
  
  const color = getConfidenceColor(confValue);
  const label = getConfidenceLabel(confValue);

  return (
    <div
      className={`pred-item ${index === 0 ? "pred-top" : ""} ${animate ? "pred-animate" : ""}`}
      style={{ animationDelay: `${index * 80}ms` }}
    >
      <div className="pred-rank" style={{ color }}>
        {index === 0 ? "🥇" : `#${pred.rank}`}
      </div>
      <div className="pred-body">
        <div className="pred-meta">
          <span className="pred-label">{pred.label}</span>
          <div className="pred-right">
            <span className="pred-badge" style={{ background: color + "22", color }}>
              {label}
            </span>
            <span className="pred-conf" style={{ color }}>{pred.confidence}</span>
          </div>
        </div>
        <div className="bar-track">
          <div
            className="bar-fill"
            style={{
              width: animate ? `${confValue}%` : "0%",
              background: `linear-gradient(90deg, ${color}cc, ${color})`,
              transition: `width 0.8s cubic-bezier(0.4,0,0.2,1) ${index * 80}ms`,
            }}
          />
        </div>
      </div>
    </div>
  );
}

// ─── Results Panel ────────────────────────────────────────────────────────────
function ResultsPanel({ predictions, loading, animate, onDownload, onShare, shareLabel }) {
  if (loading) {
    return (
      <div className="panel results-panel">
        <div className="state-center">
          <div className="pulse-ring"><div className="spinner-large" /></div>
          <p className="state-title">Analyzing image…</p>
          <p className="state-sub">Finetuned Model (Keras) is running inference</p>
        </div>
      </div>
    );
  }

  if (!predictions) {
    return (
      <div className="panel results-panel">
        <div className="state-center">
          <div className="empty-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </div>
          <p className="state-title">No predictions yet</p>
          <p className="state-sub">Upload an image and click <strong>Classify</strong></p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel results-panel">
      <div className="results-header">
        <div>
          <h2 className="results-title">Top 5 Predictions</h2>
          <p className="results-sub">
            Model: <span className="chip">Finetuned Model</span>
            &nbsp;Framework: <span className="chip">Keras / TF</span>
            &nbsp;· <span className="chip">{formatTime(predictions.timestamp)}</span>
          </p>
        </div>
      </div>

      <div className="pred-list">
        {predictions.predictions.map((pred, i) => (
          <PredictionBar key={pred.rank} pred={pred} index={i} animate={animate} />
        ))}
      </div>

      <div className="results-footer">
        <button className="btn btn-primary w-full" onClick={onDownload}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Download PDF Report
        </button>
        <button className="btn btn-secondary w-full" onClick={onShare}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
          </svg>
          {shareLabel}
        </button>
      </div>
    </div>
  );
}

// ─── Toast ────────────────────────────────────────────────────────────────────
function Toast({ message, visible }) {
  return (
    <div className={`toast ${visible ? "toast-visible" : ""}`}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
        <polyline points="20 6 9 17 4 12" />
      </svg>
      {message}
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [imageFile, setImageFile]     = useState(null);
  const [imageURL, setImageURL]       = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [isDragging, setIsDragging]   = useState(false);
  const [animate, setAnimate]         = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [toast, setToast]             = useState({ visible: false, message: "" });

  const showToast = (message) => {
    setToast({ visible: true, message });
    setTimeout(() => setToast({ visible: false, message: "" }), 2500);
  };

  const handleFile = useCallback((file) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload a valid image file (PNG, JPG, WEBP …)");
      return;
    }
    if (imageURL) URL.revokeObjectURL(imageURL);
    setImageFile(file);
    setImageURL(URL.createObjectURL(file));
    setPredictions(null);
    setAnimate(false);
    setError(null);
  }, [imageURL]);

  const loadDemo = async () => {
    setError(null);
    try {
      const resp = await fetch(DEMO_IMAGE_URL);
      const blob = await resp.blob();
      handleFile(new File([blob], "banana_demo.jpg", { type: "image/jpeg" }));
      showToast("Demo image loaded 🍌");
    } catch {
      setError("Could not load demo image. Check your connection.");
    }
  };

  const classify = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setPredictions(null);
    setAnimate(false);

    const fd = new FormData();
    fd.append("file", imageFile);

    try {
      const { data } = await axios.post(`${API}/classify_v1`, fd);
      setPredictions(data);
      setTimeout(() => setAnimate(true), 50);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Classification failed. Make sure the FastAPI backend is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!imageFile) return;
    setReportLoading(true);
    const fd = new FormData();
    fd.append("file", imageFile);
    try {
      const { data, headers } = await axios.post(`${API}/report`, fd, {
        responseType: "blob",
      });
      const cdHeader  = headers["content-disposition"] || "";
      const nameMatch = cdHeader.match(/filename="?([^"]+)"?/);
      const filename  = nameMatch ? nameMatch[1] : "classification_report.pdf";
      const url = URL.createObjectURL(new Blob([data], { type: "application/pdf" }));
      const a   = document.createElement("a");
      a.href = url; a.download = filename;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
      showToast("Report downloaded ✓");
    } catch {
      setError("Failed to generate report. Ensure the backend is running.");
    } finally {
      setReportLoading(false);
    }
  };

  const share = async () => {
    let text = "Check out this Image Classifier App powered by Finetuned Model (Keras/TensorFlow)!";
    if (predictions?.predictions?.length) {
      const top = predictions.predictions[0];
      text = `🔬 Image Classifier: "${top.label}" — ${top.confidence}% confidence\n(Finetuned Model / Keras / ImageNet)`;
    }
    if (navigator.share) {
      try { await navigator.share({ title: "ImageClassifier", text }); return; } catch {}
    }
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied to clipboard ✓");
    } catch {
      showToast("Could not copy to clipboard");
    }
  };

  const shareLabel = toast.visible && toast.message.includes("Copied") ? "Copied!" : "Share";

  return (
    <div className="app">
      <Toast message={toast.message} visible={toast.visible} />

      <Header
        onShare={share}
        onDownload={downloadReport}
        hasPredictions={!!predictions}
        shareLabel={shareLabel}
      />

      <main className="main">
        {/* Hero */}
        <section className="hero">
          <div className="hero-badge">
            <span className="dot" />
            Keras · TensorFlow · Finetuned Model · ImageNet
          </div>
          <h1 className="hero-title">
            Classify Any Image
            <br />
            <span className="hero-accent">in Seconds</span>
          </h1>
          <p className="hero-sub">
            Upload an Image of any (Sunflower, Rose, Tulip, Daisy, Daffodil) and get the top 5 predicted classes with confidence scores,
            powered by finetuned Model using Keras and TensorFlow.
          </p>
        </section>

        {/* Grid */}
        <div className="content-grid">
          {/* Upload */}
          <div className="panel upload-panel">
            <div className="panel-head">
              <h2 className="panel-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                Upload Image
              </h2>
              {!imageFile && (
                <button className="btn btn-ghost small" onClick={loadDemo}>
                  Try Demo 🍌
                </button>
              )}
            </div>

            <UploadZone
              onFile={handleFile}
              isDragging={isDragging}
              setIsDragging={setIsDragging}
              image={imageURL}
              loading={loading}
            />

            {error && (
              <div className="error-banner">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                {error}
              </div>
            )}

            {imageFile && (
              <div className="file-info">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
                <span className="file-name">{imageFile.name}</span>
                <span className="file-size">{(imageFile.size / 1024).toFixed(0)} KB</span>
              </div>
            )}

            <button
              className={`btn btn-primary classify-btn ${loading ? "loading" : ""}`}
              onClick={classify}
              disabled={!imageFile || loading}
            >
              {loading ? (
                <><div className="spinner-sm" />Analyzing…</>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                  Classify Image
                </>
              )}
            </button>
          </div>

          {/* Results */}
          <ResultsPanel
            predictions={predictions}
            loading={loading}
            animate={animate}
            onDownload={downloadReport}
            onShare={share}
            shareLabel={reportLoading ? "Generating…" : shareLabel}
          />
        </div>

        {/* Stats */}
        <div className="stats-row">
          {[
            { icon: "🧠", label: "Model",     value: "Finetuned Model"     },
            { icon: "⚙️",  label: "Framework", value: "Keras / TF"    },
            { icon: "📦", label: "Classes",   value: "1,000"         },
            { icon: "⚡", label: "Inference", value: "< 1s"          },
          ].map((s) => (
            <div key={s.label} className="stat-card">
              <span className="stat-icon">{s.icon}</span>
              <div>
                <p className="stat-value">{s.value}</p>
                <p className="stat-label">{s.label}</p>
              </div>
            </div>
          ))}
        </div>
      </main>

      <footer className="footer">
        <p>
          Built with <span className="accent">FastAPI</span> &amp;{" "}
          <span className="accent">React</span> · Powered by{" "}
          <span className="accent">Finetuned Model (Keras / TensorFlow)</span>
        </p>
      </footer>
    </div>
  );
}
