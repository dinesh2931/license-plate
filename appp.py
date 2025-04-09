import cv2
import numpy as np
import os
import re
from ultralytics import YOLOv10
from paddleocr import PaddleOCR
import streamlit as st

# Avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model and OCR
model = YOLOv10("best.pt")  # Make sure it's YOLOv8 .pt model
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Streamlit UI
st.title("License Plate Recognition")

def paddle_ocr(frame, x1, y1, x2, y2):
    cropped = frame[y1:y2, x1:x2]
    result = ocr.ocr(cropped, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
    text = re.sub(r'\W+', '', text)
    text = text.replace("???", "").replace("O", "0").replace("粤", "")
    return str(text)

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run detection
    results = model(frame)  # ✅ FIXED: direct inference

    license_plates = set()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = paddle_ocr(frame, x1, y1, x2, y2)
            if label:
                license_plates.add(label)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    st.image(frame, channels="BGR", caption="Detected License Plates", use_column_width=True)
    st.success(", ".join(license_plates) if license_plates else "No license plates found.")
