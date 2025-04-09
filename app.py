import cv2
import numpy as np
import os
import re
from ultralytics import YOLOv10
from paddleocr import PaddleOCR
import streamlit as st
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model and OCR
model = YOLOv10("best.pt")  # Replace with your model path
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Streamlit app title
st.title("License Plate Recognition - Image")

# OCR helper function
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
    # Clean up
    text = re.sub(r'\W+', '', text)
    text = text.replace("???", "").replace("O", "0").replace("粤", "")
    return str(text)

# File uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Run model
    results = model.predict(frame, conf=0.45)

    license_plates = set()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = paddle_ocr(frame, x1, y1, x2, y2)
            if label:
                license_plates.add(label)
            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0], y1 - textSize[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    st.image(frame, channels="BGR", caption="Processed Image", use_column_width=True)
    st.write("Detected License Plates:")
    st.success(", ".join(license_plates) if license_plates else "No license plates found.")
