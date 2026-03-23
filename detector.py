from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

def detect_people(frame):
    results = model(frame, classes=[0], verbose=False)
    annotated_frame = results[0].plot()
    people_count = len(results[0].boxes)
    return annotated_frame, people_count


def run_webcam():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam could not be opened.")
        return

    print("Running... for exit press 'q'")

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
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow('People Counter', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()