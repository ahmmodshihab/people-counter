import streamlit as st
import cv2
import tempfile
import os
from detector import detect_people, detect_from_image, detect_from_video, save_video

st.set_page_config(page_title="People Counter", page_icon="👥", layout="wide")
st.title("👥 People Detection System")

# Sidebar
st.sidebar.header("Settings")
source = st.sidebar.radio("Source", ["Image", "Video", "Webcam"])

# ─── IMAGE ───────────────────────────────────────────
if source == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        annotated, count = detect_from_image(uploaded.read())
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="Detected People", use_column_width=True)
        st.metric("👥 People Count", count)

# ─── VIDEO ───────────────────────────────────────────
elif source == "Video":
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Processing... Please wait"):
            frames, counts = detect_from_video(tmp_path)

            # Detected video save করো
            output_path = tmp_path.replace('.mp4', '_detected.mp4')
            save_video(frames, output_path, fps=10)

        if frames:
            st.success(f"✅ {len(frames)} frames detected")

            # Max count দেখাও
            st.metric("👥 Max People Detected", max(counts))

            # Smooth video playback
            st.subheader("Detected Video")
            st.video(output_path)

            # Cleanup
            os.remove(tmp_path)
            os.remove(output_path)

# ─── WEBCAM ──────────────────────────────────────────
elif source == "Webcam":
    st.info("Webcam live feed — Click Start ")

    col1, col2 = st.columns(2)
    run = col1.button("▶ Start")
    stop = col2.button("⏹ Stop")
    frame_window = st.image([])
    count_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break

            annotated, count = detect_people(frame)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_window.image(annotated_rgb, use_column_width=True)

            with count_placeholder.container():
                st.metric("👥 People Count", count)

        cap.release()