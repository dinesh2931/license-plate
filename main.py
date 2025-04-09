import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from paddleocr import PaddleOCR
import tempfile
import os

# Initialize OCR model
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

# Set Streamlit page config
st.set_page_config(page_title="License Plate Detection", layout="centered")
st.title("🚗 Vehicle License Plate Detection")
st.markdown("Upload an image, and the system will detect and read the license plate.")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Update path to your trained YOLO model

model = load_model()

# Function to read plate text
def recognize_text(image_np):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        temp_path = temp.name
        cv2.imwrite(temp_path, image_np)

    result = ocr_model.ocr(temp_path, cls=True)
    os.remove(temp_path)

    text = ""
    if result:
        for line in result:
            for word in line:
                text += word[1][0] + " "
    return text.strip()

# Function to process image
def process_image(image_np):
    results = model.predict(image_np)[0]

    boxes = results.boxes.data.cpu().numpy().astype(int)
    annotated_img = image_np.copy()
    final_text = ""

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        cropped = annotated_img[y1:y2, x1:x2]
        plate_text = recognize_text(cropped)
        final_text = plate_text

        # Draw rectangle and text
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return annotated_img, final_text

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_np is not None:
        with st.spinner("Processing..."):
            result_image, recognized_text = process_image(image_np)

        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)

        if recognized_text:
            st.success(f"Recognized Plate Text: **{recognized_text}**")
        else:
            st.warning("No text could be recognized from the license plate.")
    else:
        st.error("Could not read the uploaded image. Try a different file.")
