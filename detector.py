from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')

def detect_people(frame):
    """যেকোনো frame-এ people detect করবে"""
    results = model(frame, classes=[0], verbose=False)
    annotated_frame = results[0].plot()
    people_count = len(results[0].boxes)
    return annotated_frame, people_count


def detect_from_image(image_bytes):
    """Image bytes থেকে detect করবে"""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return detect_people(frame)


def detect_from_video(video_path):
    """Video file path থেকে frames detect করবে"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    counts = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated, count = detect_people(frame)
        frames.append(annotated)
        counts.append(count)
    cap.release()
    return frames, counts


def run_webcam():
    """Webcam থেকে real-time detect করবে"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    print("Running webcam... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, count = detect_people(frame)
        cv2.putText(
            annotated_frame,
            f'People Count: {count}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 255, 0), 3
        )
        cv2.imshow('People Counter', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()