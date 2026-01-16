import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="LICENSE Detection AI", page_icon="⛑️", layout="centered")

@st.cache_resource
def load_yolo_model():
    # Ensure this path matches your file location
    model_path = r"C:\Users\kk\OneDrive\Desktop\streamlit\best (1).pt" 
    return YOLO(model_path)

model = load_yolo_model()

# 2. UI Layout
st.title("LICENSE Detection System")
st.write("Upload an image to detect if a LICENSE NO PLATE is being worn.")

# 3. File Uploader (Added jpeg and JPG to the allowed types)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "JPG", "JPEG"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Original Image")
        st.image(image, use_column_width=True)

    # 4. Predict Button
    if st.button('Run LICENSE Detection'):
        with st.spinner('Analyzing image...'):
            # Run the YOLO model
            results = model.predict(image)
            
            # Plot results and convert BGR to RGB
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.success("Detection Result")
                st.image(res_rgb, caption='Processed Image', use_column_width=True)
            
            # 5. Status Logic: Check if "Helmet" is in the detected classes
            detections = results[0].boxes
            if len(detections) > 0:
                # Get list of all detected labels
                detected_names = [model.names[int(cls)] for cls in detections.cls]
                
                # Check if 'helmet' appears in our list of detected names
                # (Lower() ensures it matches even if your label is 'Helmet' or 'HELMET')
                if any("NO PLATE" in name.lower() for name in detected_names):
                    st.success("### ✅ Prediction: Helmet Detected")
                    st.balloons() # Visual celebration for safety!
                else:
                    st.error("### ❌ Prediction: No Helmet Detected")
                    st.warning("Only other objects were found in the frame.")
            else:
                st.error("### ❌ Prediction: No Helmet Detected")
                st.info("The model could not identify any objects in this image.")

            # Show count for context
            st.metric(label="Total Objects Found", value=len(detections))