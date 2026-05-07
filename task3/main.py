import os
import time
from collections import Counter
from datetime import datetime

import cv2
from ultralytics import YOLO


PHONE_IP = "192.168.100.17"
PHONE_PORT = "8080"
USE_PHONE_CAMERA = True
CAMERA_INDEX = 0

MODEL_PATH = "yolov8s.pt"
CONFIDENCE = 0.25
OUTPUT_PATH = "output.mp4"
WINDOW_NAME = "Object Detection and Tracking"
SHOW_ALL_DETECTIONS = True

CAPTURES_FOLDER = "captures"
CAPTURE_COOLDOWN = 3.0

ALLOWED_CLASSES = [
    "person", "car", "bottle", "cell phone",
    "laptop", "dog", "cat", "book",
]

COUNT_LABELS = {
    "person": "People",
    "cell phone": "Phones",
    "bottle": "Bottles",
    "laptop": "Laptops",
    "dog": "Dogs",
    "cat": "Cats",
    "book": "Books",
    "car": "Cars",
}


class CaptureManager:
    def __init__(self, folder: str, cooldown: float):
        self.folder = folder
        self.cooldown = cooldown
        self._last_saved: dict[str, float] = {}
        self._total_saved = 0
        os.makedirs(folder, exist_ok=True)
        print(f"[INFO] Captures will be saved to: {os.path.abspath(folder)}/")

    def try_save(self, frame, detected_classes: set) -> list[str]:
        now = time.time()
        triggered = []

        for cls in detected_classes:
            last = self._last_saved.get(cls, 0)
            if now - last >= self.cooldown:
                triggered.append(cls)
                self._last_saved[cls] = now

        if triggered:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            label = "_".join(sorted(triggered))
            filename = f"{label}_{timestamp}.jpg"
            filepath = os.path.join(self.folder, filename)
            cv2.imwrite(filepath, frame)
            self._total_saved += 1
            print(f"[CAPTURE] Saved: {filepath}  (total: {self._total_saved})")

        return triggered

    @property
    def total_saved(self):
        return self._total_saved


def open_phone_camera(ip: str, port: str):
    url = f"http://{ip}:{port}/video"
    print(f"[INFO] Connecting to phone camera: {url}")
    print("[INFO] Make sure phone and PC are on the same WiFi network.")

    cap = cv2.VideoCapture(url)

    for attempt in range(10):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0 and frame.mean() > 1:
            print(f"[INFO] Phone camera connected. Resolution: {frame.shape[1]}x{frame.shape[0]}")
            return cap, frame
        print(f"[INFO] Waiting for phone stream... attempt {attempt + 1}/10")
        time.sleep(0.5)

    cap.release()
    raise RuntimeError(
        f"Could not connect to phone camera at {url}\n"
        "Make sure the camera app is running on your phone.\n"
        "Make sure phone and PC are on the same WiFi network.\n"
        "Update PHONE_IP at the top of this file.\n"
        "Test by opening the URL in your browser."
    )


def open_laptop_camera(camera_index: int):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        for _ in range(30):
            cap.read()
            time.sleep(0.05)

        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0 and frame.mean() > 1:
                print(
                    f"[INFO] Laptop camera opened: index={camera_index}, "
                    f"backend={backend}, resolution={frame.shape[1]}x{frame.shape[0]}"
                )
                return cap, frame
            time.sleep(0.1)

        cap.release()

    raise RuntimeError(
        f"Could not read from camera index {camera_index}.\n"
        "Close other apps using the camera.\n"
        "Check camera permissions in system settings.\n"
        "Try CAMERA_INDEX = 1 or 2."
    )


def draw_box(frame, x1, y1, x2, y2, label: str, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    label_y = max(y1, text_h + 10)
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - baseline - 4),
        (x1 + text_w + 4, label_y),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame, label,
        (x1 + 2, label_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
    )


def draw_capture_flash(frame, frame_h: int, frame_w: int, triggered: list[str]):
    cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), (0, 255, 0), 6)
    msg = f"CAPTURED: {', '.join(triggered)}"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cx = (frame_w - tw) // 2
    cy = frame_h // 2
    cv2.rectangle(frame, (cx - 8, cy - 30), (cx + tw + 8, cy + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def draw_hud(frame, fps: float, total_boxes: int, counts: Counter,
             frame_h: int, frame_w: int, source_label: str, total_captures: int):
    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    (tw, _), _ = cv2.getTextSize(source_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, source_label,
                (frame_w - tw - 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    cv2.putText(frame, f"YOLO boxes: {total_boxes}",
                (20, frame_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cap_label = f"Captures saved: {total_captures}"
    (tw2, _), _ = cv2.getTextSize(cap_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, cap_label,
                (frame_w - tw2 - 15, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)

    if not counts:
        cv2.putText(frame, "No allowed objects detected",
                    (20, frame_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    y = 80
    for class_name, label in COUNT_LABELS.items():
        cv2.putText(frame, f"{label}: {counts[class_name]}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        y += 35


def main():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    if USE_PHONE_CAMERA:
        cap, first_frame = open_phone_camera(PHONE_IP, PHONE_PORT)
        source_label = f"IP Webcam  {PHONE_IP}"
    else:
        cap, first_frame = open_laptop_camera(CAMERA_INDEX)
        source_label = f"Webcam [{CAMERA_INDEX}]"

    frame_h, frame_w = first_frame.shape[:2]
    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = camera_fps if camera_fps and camera_fps > 0 else 20

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (frame_w, frame_h),
    )
    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for '{OUTPUT_PATH}'.")

    capture_mgr = CaptureManager(CAPTURES_FOLDER, CAPTURE_COOLDOWN)

    print("[INFO] Ready. Show objects: person, bottle, phone, laptop, book, car, dog, cat.")
    print("[INFO] Press Q to quit.")

    prev_time = time.time()
    next_frame = first_frame
    flash_frames_left = 0
    last_triggered = []

    while True:
        if next_frame is not None:
            frame = next_frame
            next_frame = None
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Stream stopped returning frames.")
                break

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now

        results = model.predict(frame, conf=CONFIDENCE, imgsz=640, verbose=False)

        annotated = frame.copy()
        counts = Counter()
        boxes = results[0].boxes
        total_boxes = len(boxes) if boxes is not None else 0
        detected_allowed = set()

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])

                if class_name in ALLOWED_CLASSES:
                    counts[class_name] += 1
                    detected_allowed.add(class_name)

                if not SHOW_ALL_DETECTIONS and class_name not in ALLOWED_CLASSES:
                    continue

                color = (0, 220, 0) if class_name in ALLOWED_CLASSES else (0, 140, 255)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_box(annotated, x1, y1, x2, y2, f"{class_name} {confidence:.2f}", color)

        if detected_allowed:
            triggered = capture_mgr.try_save(annotated, detected_allowed)
            if triggered:
                last_triggered = triggered
                flash_frames_left = 8

        draw_hud(annotated, fps, total_boxes, counts,
                 frame_h, frame_w, source_label, capture_mgr.total_saved)

        if flash_frames_left > 0:
            draw_capture_flash(annotated, frame_h, frame_w, last_triggered)
            flash_frames_left -= 1

        out.write(annotated)
        cv2.imshow(WINDOW_NAME, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Video saved to '{OUTPUT_PATH}'.")
    print(f"[INFO] Total captures: {capture_mgr.total_saved} in {os.path.abspath(CAPTURES_FOLDER)}/")


if __name__ == "__main__":
    main()
