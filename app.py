import streamlit as st
import tempfile
import cv2
from detector import detect_from_image, detect_from_video

st.set_page_config(
    page_title="People Counter",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Real-Time People Counter")


st.sidebar.title("⚙️ Options")
source = st.sidebar.radio(
    "Select Input Source",
    ["Webcam", "Video File", "Image"]
)

if source == "Image":
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )
    if uploaded:
        image_bytes = uploaded.read()
        annotated, count = detect_from_image(image_bytes)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_bytes, caption="Original")
        with col2:
            st.image(annotated_rgb, caption=f"Detected: {count} people")
        st.metric("Total People", count)

elif source == "Video File":
    uploaded = st.file_uploader(
        "Upload Video",
        type=["mp4","avi","mov"]
    )
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        st.info("Processing...")
        frames, counts = detect_from_video(tfile.name)
        stframe = st.empty()
        count_display = st.empty()
        for frame, count in zip(frames, counts):
            annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, use_column_width=True)
            count_display.metric("People Count", count)
        st.success("Done!")

elif source == "Webcam":
    st.warning("Webcam mode — run detector.py locally ")
    st.code("python detector.py")