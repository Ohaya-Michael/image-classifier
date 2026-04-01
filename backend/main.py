"""
Image Classification API - FastAPI Backend
Model: ResNet-50 (ImageNet pre-trained, 1000 classes)
Framework: TensorFlow / Keras
"""

import io
from datetime import datetime
import json

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
#from helper_functions.process import _predict
from helper_functions.pdf_builder import _build_pdf
from helper_functions.predict import predict_image
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


# @app.post("/classify", tags=["Classification"])
# async def classify(file: UploadFile = File(...)):
#     """
#     Upload an image → receive top-5 class predictions with confidence scores.
#     """
#     if not file.content_type or not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp …)")

#     data = await file.read()
#     try:
#         img = Image.open(io.BytesIO(data)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=422, detail="Could not decode image.")

#     predictions = _predict(img)

#     return {
#         "filename": file.filename,
#         "framework": "TensorFlow / Keras",
#         "model": "ResNet-50",
#         "timestamp": datetime.now().isoformat(),
#         "predictions": predictions,
#     }


@app.post("/classify_v1", tags=["Classification_v1"])
async def classify_v1(file: UploadFile = File(...)):
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

    # Read file content into memory
    #content = await file.read()
    
    # save the uploaded file to a temporary location
    # temp_file_path = f"uploads/temp_{file.filename}"
    # with open(temp_file_path, "wb") as temp_file:
    #     temp_file.write(content)

    # Predict image using the prediction function
    predictions_dict = predict_image("image_prediction/models/trained_model_finetuned.keras", img)
    
    # Convert dictionary to array format for frontend/PDF
    # Parse confidence string (e.g., "85.50%") to numeric value
    predictions = [
        {
            "rank": rank + 1,
            "label": label,
            "confidence": float(confidence.rstrip("%"))
        }
        for rank, (label, confidence) in enumerate(predictions_dict.items())
    ]

    return {
        "filename": file.filename,
        "framework": "TensorFlow / Keras",
        "model": "Custom Fine-tuned",
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

    predictions_dict = predict_image("image_prediction/models/trained_model_finetuned.keras", img)
    
    # Convert to array format with numeric confidence for PDF builder
    predictions = [
        {
            "rank": rank + 1,
            "label": label,
            "confidence": float(confidence.rstrip("%"))
        }
        for rank, (label, confidence) in enumerate(predictions_dict.items())
    ]
    
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
