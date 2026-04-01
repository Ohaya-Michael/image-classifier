
import io
import numpy as np
from PIL import Image
from datetime import datetime


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
        leftMargin=0.75*inch, 
        rightMargin=0.75*inch,
        topMargin=0.75*inch,  
        bottomMargin=0.75*inch,
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
        conf_numeric = p["confidence"]
        bar = "█" * int(conf_numeric / 100 * 20) + "░" * (20 - int(conf_numeric / 100 * 20))
        rows.append([str(p["rank"]), p["label"], f"{conf_numeric:.2f}%", bar])

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