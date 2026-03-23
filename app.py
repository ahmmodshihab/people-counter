import streamlit as st
import cv2
import tempfile
import numpy as np
from detector import detect_people
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

st.set_page_config(
    page_title="People Counter",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Real-Time People Counter")
st.markdown("**YOLOv8 + OpenCV** দিয়ে তৈরি — মানুষ detect ও count করে")

st.sidebar.title("⚙️ Options")
source = st.sidebar.radio(
    "Input Source বেছে নাও",
    ["Webcam", "Video File", "Image"]
)

if source == "Image":
    uploaded = st.file_uploader("Image upload করো", type=["jpg","jpeg","png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, count = detect_people(frame)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1:
            st.image(frame[:,:,::-1], caption="Original", use_column_width=True)
        with col2:
            st.image(annotated_rgb, caption=f"Detected: {count} people", use_column_width=True)
        st.metric("Total People Detected", count)

elif source == "Video File":
    uploaded = st.file_uploader("Video upload করো", type=["mp4","avi","mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        st.info("Processing হচ্ছে...")
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        count_display = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, count = detect_people(frame)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, use_column_width=True)
            count_display.metric("People Count", count)
        cap.release()
        st.success("Done!")

elif source == "Webcam":
    st.warning("Webcam mode Streamlit-এ directly কাজ করে না — detector.py দিয়ে run করো")
    st.code("python detector.py")