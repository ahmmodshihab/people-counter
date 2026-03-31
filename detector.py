from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')


def detect_people(frame):
    """যেকোনো frame এ people detect করবে"""
    results = model(frame, classes=[0], verbose=False)
    annotated_frame = results[0].plot()
    people_count = len(results[0].boxes)
    return annotated_frame, people_count


def detect_from_image(image_bytes):
    """Image bytes থেকে detect করবে"""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return detect_people(frame)


def detect_from_video(video_path, frame_skip=3):
    """Video file থেকে frame by frame detect করবে"""
    cap = cv2.VideoCapture(video_path)

    frames = []
    counts = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        # resize frame to speed up detection
        frame = cv2.resize(frame, (1280, 720))

        annotated, count = detect_people(frame)
        frames.append(annotated)
        counts.append(count)

    cap.release()
    return frames, counts


def save_video(frames, output_path, fps=10):
    """Save detected frames as a video file"""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'avc1'),  # browser compatible
        fps,
        (w, h)
    )
    for f in frames:
        writer.write(f)
    writer.release()


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