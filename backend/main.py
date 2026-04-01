"""
Image Classification API - FastAPI Backend
Model: ResNet-50 (ImageNet pre-trained, 1000 classes)
Framework: TensorFlow / Keras
"""

import io
from datetime import datetime

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Image Classifier API",
    description="ResNet-50 (Keras/TensorFlow) image classification — top-5 ImageNet predictions",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Model (loaded once at startup)
# ─────────────────────────────────────────────────────────────
print("⏳  Loading ResNet-50 (Keras) …")
_model = tf.keras.applications.ResNet50(weights="imagenet")
_model.trainable = False
print("✅  ResNet-50 ready")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _preprocess(img: Image.Image) -> np.ndarray:
    """Resize to 224×224 and apply ResNet-50 ImageNet preprocessing."""
    img = img.resize((224, 224), Image.LANCZOS)
    arr = tf.keras.preprocessing.image.img_to_array(img)          # (224,224,3)
    arr = np.expand_dims(arr, axis=0)                              # (1,224,224,3)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)     # channel-wise norm
    return arr


def _predict(img: Image.Image) -> list[dict]:
    """Run ResNet-50 and return top-5 dicts via Keras decode_predictions."""
    arr = _preprocess(img)
    preds = _model.predict(arr, verbose=0)                         # (1,1000)
    # decode_predictions returns [[( class_id, label, prob ), …]]
    top5 = tf.keras.applications.resnet50.decode_predictions(preds, top=5)[0]
    return [
        {
            "rank": idx + 1,
            "label": label.replace("_", " ").title(),
            "confidence": round(float(prob) * 100, 2),
            "class_id": class_id,
        }
        for idx, (class_id, label, prob) in enumerate(top5)
    ]


def _build_pdf(img: Image.Image, preds: list[dict], filename: str) -> io.BytesIO:
    """Create a styled PDF report with ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    PURPLE     = colors.HexColor("#6C63FF")
    DARK       = colors.HexColor("#1A1D27")
    LIGHT_ROW  = colors.HexColor("#F3F4F6")
    WHITE      = colors.white

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch,
    )
    styles = getSampleStyleSheet()
    story  = []

    # ── Title ────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "T", parent=styles["Title"],
        fontSize=26, textColor=PURPLE,
        spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold",
    )
    sub_style = ParagraphStyle(
        "S", parent=styles["Normal"],
        fontSize=10, textColor=colors.gray,
        spaceAfter=18, alignment=TA_CENTER,
    )
    story.append(Paragraph("🔬 Image Classification Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}"
        f"  &nbsp;|&nbsp;  Model: ResNet-50 (Keras / TensorFlow)",
        sub_style,
    ))

    hr = Table([[""]], colWidths=[7*inch], rowHeights=[2])
    hr.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), PURPLE)]))
    story.append(hr)
    story.append(Spacer(1, 0.2*inch))

    # ── Image thumbnail ──────────────────────────────────────
    img_buf = io.BytesIO()
    thumb = img.copy()
    thumb.thumbnail((320, 320), Image.LANCZOS)
    max_side = max(thumb.size)
    padded = Image.new("RGB", (max_side, max_side), (26, 29, 39))
    padded.paste(thumb, ((max_side - thumb.size[0])//2, (max_side - thumb.size[1])//2))
    padded.save(img_buf, format="PNG")
    img_buf.seek(0)

    rl_img = RLImage(img_buf, width=2.8*inch, height=2.8*inch)
    img_tbl = Table([[rl_img]], colWidths=[7*inch])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",          (0,0), (-1,-1), "CENTER"),
        ("BACKGROUND",     (0,0), (-1,-1), DARK),
        ("BOX",            (0,0), (-1,-1), 1, PURPLE),
        ("TOPPADDING",     (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 10),
    ]))
    story.append(img_tbl)

    fn_style = ParagraphStyle(
        "FN", parent=styles["Normal"],
        fontSize=9, textColor=colors.gray,
        spaceBefore=6, spaceAfter=18, alignment=TA_CENTER,
    )
    story.append(Paragraph(f"📁  {filename or 'uploaded_image'}", fn_style))

    # ── Section heading ──────────────────────────────────────
    sec_style = ParagraphStyle(
        "SEC", parent=styles["Heading2"],
        fontSize=14, textColor=PURPLE, spaceAfter=10, fontName="Helvetica-Bold",
    )
    story.append(Paragraph("Top 5 Predictions", sec_style))

    # ── Predictions table ────────────────────────────────────
    RANK_COLORS = ["#6C63FF","#7B74FF","#938DFF","#A8A3FF","#C2BFFF"]
    CONF_COLORS = ["#10B981","#10B981","#F59E0B","#F59E0B","#EF4444"]

    rows = [["#", "Class Label", "Confidence", "Confidence Bar"]]
    for p in preds:
        bar = "█" * int(p["confidence"] / 100 * 20) + "░" * (20 - int(p["confidence"] / 100 * 20))
        rows.append([str(p["rank"]), p["label"], f"{p['confidence']}%", bar])

    pred_tbl = Table(rows, colWidths=[0.45*inch, 2.8*inch, 1.1*inch, 2.65*inch])
    tbl_style = [
        ("BACKGROUND",  (0,0), (-1,0), PURPLE),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ALIGN",       (0,0), (-1,0), "CENTER"),
        ("ALIGN",       (0,1), (0,-1), "CENTER"),
        ("ALIGN",       (2,1), (2,-1), "CENTER"),
        ("FONTNAME",    (3,1), (3,-1), "Courier"),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#DEE2E6")),
        ("TOPPADDING",  (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]
    for i in range(1, len(rows)):
        tbl_style.append(("BACKGROUND", (0,i), (-1,i), LIGHT_ROW if i%2==0 else WHITE))
    for i, c in enumerate(RANK_COLORS, 1):
        tbl_style += [("TEXTCOLOR",(0,i),(0,i),colors.HexColor(c)),
                      ("FONTNAME", (0,i),(0,i),"Helvetica-Bold")]
    for i, c in enumerate(CONF_COLORS, 1):
        tbl_style += [("TEXTCOLOR",(2,i),(2,i),colors.HexColor(c)),
                      ("FONTNAME", (2,i),(2,i),"Helvetica-Bold")]

    pred_tbl.setStyle(TableStyle(tbl_style))
    story.append(pred_tbl)
    story.append(Spacer(1, 0.3*inch))

    # ── Footer ───────────────────────────────────────────────
    footer_hr = Table([[""]], colWidths=[7*inch], rowHeights=[1])
    footer_hr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#363950"))]))
    story.append(footer_hr)
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "Generated by <b>ImageClassifier App</b>  •  "
        "Powered by ResNet-50 (Keras/TensorFlow) &amp; ImageNet  •  1000 classes",
        ParagraphStyle("FT", parent=styles["Normal"],
                       fontSize=8, textColor=colors.gray, alignment=TA_CENTER),
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "framework": "TensorFlow / Keras",
        "model": "ResNet-50",
        "num_classes": 1000,
        "endpoints": ["/classify", "/report"],
    }


@app.post("/classify", tags=["Classification"])
async def classify(file: UploadFile = File(...)):
    """
    Upload an image → receive top-5 class predictions with confidence scores.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp …)")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    predictions = _predict(img)

    return {
        "filename": file.filename,
        "framework": "TensorFlow / Keras",
        "model": "ResNet-50",
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
    }


@app.post("/report", tags=["Report"])
async def report(file: UploadFile = File(...)):
    """
    Upload an image → download a styled PDF classification report.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    predictions = _predict(img)
    pdf_buf = _build_pdf(img, predictions, file.filename or "image")

    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="classification_report_'
                f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
            )
        },
    )
