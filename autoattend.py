
import cv2
import os
import numpy as np
import csv
from pathlib import Path
from datetime import datetime
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from PIL import Image
from ultralytics import YOLO

# Load the YOLOv8 anti-spoofing model
SPOOF_MODEL_PATH = "E:/STUDIES/Prj/Anti_Spoof/Anti_Spoofing_YoLo/Anti_spoof3/weights/best.pt"  # Change if needed
spoof_model = YOLO(SPOOF_MODEL_PATH)

# === CONFIG ===
CAMERA_INDEX = 1
FRAME_SAVE_ROOT = Path("E:/New folder (2)/camera_capture_frames")
ALIGN_SAVE_ROOT = Path("E:/New folder (2)/Align")
EMBEDDED_DIR = Path("E:/New folder (2)/Embedded")
ATTENDANCE_LOG_PATH = Path("E:/New folder (2)/logs/attendance.csv")
THRESHOLD = 0.5


# === LOAD EXISTING ATTENDANCE LOG ===
def load_logged_names(path):
    logged = set()
    if path.exists():
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 1:
                    logged.add(row[0])
    return logged

def capture_frames(name, user_id):
    user_tag = f"{name}_{user_id}".replace(" ", "_")
    user_frame_dir = FRAME_SAVE_ROOT / user_tag
    user_frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(round(camera_fps / 3)) if camera_fps and camera_fps > 0 else 6

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            frame_filename = f"frame_{saved_count+1:04d}.jpg"
            frame_path = user_frame_dir / frame_filename
            success, encoded_img = cv2.imencode('.jpg', frame)
            if success:
                with open(frame_path, 'wb') as f:
                    f.write(encoded_img.tobytes())
                saved_count += 1

        frame_count += 1
        cv2.imshow("Capturing... Press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Success: Saved {saved_count} frames to: {user_frame_dir}")


def align_faces():
    detector = MTCNN()
    template = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    ALIGN_SAVE_ROOT.mkdir(exist_ok=True)

    for person_folder in FRAME_SAVE_ROOT.iterdir():
        if not person_folder.is_dir():
            continue

        aligned_person_folder = ALIGN_SAVE_ROOT / person_folder.name
        aligned_person_folder.mkdir(exist_ok=True)

        for img_path in person_folder.glob("*.jpg"):
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(img)
            if not detections:
                continue

            keypoints = detections[0]['keypoints']
            src = np.array([
                keypoints['left_eye'],
                keypoints['right_eye'],
                keypoints['nose'],
                keypoints['mouth_left'],
                keypoints['mouth_right']
            ], dtype=np.float32)

            M = cv2.estimateAffinePartial2D(src, template, method=cv2.LMEDS)[0]
            aligned_face = cv2.warpAffine(img, M, (112, 112))

            save_path = aligned_person_folder / img_path.name
            Image.fromarray(aligned_face).save(save_path)

def embed_faces():
    EMBEDDED_DIR.mkdir(exist_ok=True)

    for person_folder in ALIGN_SAVE_ROOT.iterdir():
        if not person_folder.is_dir():
            continue

        face_db = {}
        for file in os.listdir(person_folder):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(person_folder, file)
                try:
                    embedding_obj = DeepFace.represent(
                        img_path=img_path,
                        model_name="ArcFace",
                        enforce_detection=False,
                        detector_backend="skip"
                    )
                    embedding = embedding_obj[0]["embedding"]
                    face_db[f"{person_folder.name}_{file}"] = embedding
                except:
                    continue

        normalized_db = {
            k: normalize([v])[0] for k, v in face_db.items()
        }
        save_path = EMBEDDED_DIR / f"face_database_{person_folder.name}.npz"
        np.savez(save_path, **normalized_db)
def is_real_face(frame, confidence_threshold=0.6):
    """
    Runs YOLOv8 spoof detection on the frame.
    Returns:
        - True if a real face is detected
        - List of spoof bounding boxes (x1, y1, x2, y2)
    """
    results = spoof_model(frame, verbose=False)[0]  # disable print spam
    spoof_boxes = []
    is_real = False

    if results.boxes is not None and len(results.boxes) > 0:
        for i in range(len(results.boxes.cls)):
            cls_id = int(results.boxes.cls[i].item())
            conf = float(results.boxes.conf[i].item())

            x1, y1, x2, y2 = results.boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            print(f"[YOLO] Detected class {cls_id} with conf {conf:.2f}")

            if cls_id == 1 and conf > confidence_threshold:
                is_real = True
            elif cls_id == 0 and conf > confidence_threshold:
                spoof_boxes.append((x1, y1, x2, y2))

    return is_real, spoof_boxes




def run_real_time_recognition():
    detector = MTCNN()
    db = {}

    for npz_path in EMBEDDED_DIR.glob("*.npz"):
        db_name = npz_path.stem.replace("face_database_", "")
        data = np.load(npz_path)
        for name in data.files:
            db[db_name] = normalize([data[name]])[0]

    TEMPLATE = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    logged_names = load_logged_names(ATTENDANCE_LOG_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # --- Spoof Detection ---
        is_real, spoof_boxes = is_real_face(frame)

        # Draw spoof bounding boxes
        for (x1, y1, x2, y2) in spoof_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Spoof", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
            

        if not is_real:
            cv2.putText(frame, "Spoof Detected - Skipping", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Real-Time Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue



        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)

        for face in results:
            x, y, w, h = face['box']
            keypoints = face['keypoints']
            src = np.array([
                keypoints['left_eye'], keypoints['right_eye'],
                keypoints['nose'], keypoints['mouth_left'], keypoints['mouth_right']
            ], dtype=np.float32)

            M, _ = cv2.estimateAffinePartial2D(src, TEMPLATE, method=cv2.LMEDS)
            if M is None:
                continue

            aligned = cv2.warpAffine(rgb, M, (112, 112))
            aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
            cv2.imwrite("temp_face.jpg", aligned_bgr)

            try:
                result = DeepFace.represent(
                    img_path="temp_face.jpg",
                    model_name="ArcFace",
                    detector_backend="skip",
                    enforce_detection=False
                )
                emb = normalize([result[0]["embedding"]])[0]

                best_match = None
                best_dist = float("inf")
                for person_id, stored_emb in db.items():
                    dist = cosine(emb, stored_emb)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = person_id

                similarity = 1 - best_dist
                if best_dist < THRESHOLD:
                    label = f"{best_match} ({similarity:.2f})"
                    color = (0, 255, 0)
                    if best_match not in logged_names:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ATTENDANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                        with open(ATTENDANCE_LOG_PATH, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([best_match, now, "Present"])
                        logged_names.add(best_match)
                else:
                    label = f"Unknown ({similarity:.2f})"
                    color = (0, 165, 255)

            except:
                label = "Recognition Error"
                color = (128, 128, 128)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Real-Time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists("temp_face.jpg"):
        os.remove("temp_face.jpg")




def main():
    while True:
        print("\n=== AutoAttend Menu ===")
        print("1. Capture Frames")
        print("2. Align Faces")
        print("3. Embed Faces")
        print("4. Recognize Faces")
        print("5. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            name = input("Enter name: ")
            user_id = input("Enter ID: ")
            capture_frames(name, user_id)
        elif choice == "2":
            align_faces()
        elif choice == "3":
            embed_faces()
        elif choice == "4":
            run_real_time_recognition()
        elif choice == "5":
            print("Exiting program.")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()

 