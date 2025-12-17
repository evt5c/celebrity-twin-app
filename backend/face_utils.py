import os
import urllib.request
import shutil
import onnxruntime as ort
import cv2
import numpy as np
import mediapipe as mp


def ensure_arcface_model(model_path: str):
    if os.path.exists(model_path):
        return

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("Downloading ArcFace model...")

    download_url = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx"
    tmp_path = model_path + ".download"

    urllib.request.urlretrieve(download_url, tmp_path)
    shutil.move(tmp_path, model_path)

    print("ArcFace model saved as:", model_path)



class FaceProcessor:
    """
    Handles:
    - MediaPipe face detection
    - MediaPipe face mesh landmark extraction
    - ArcFace ONNX embedding extraction
    """

    def __init__(self, onnx_path):
        ensure_arcface_model(onnx_path)

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name

        # MediaPipe models
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    # ---------------------------------------------------------
    # FACE DETECTION (returns bounding box of largest face)
    # ---------------------------------------------------------
    def detect_face(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(img_rgb)

        if not results.detections:
            return None

        h, w, _ = img_bgr.shape

        # Find largest face by bounding box area
        best_box = None
        best_area = 0

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

        return best_box

    # ---------------------------------------------------------
    # FACE CROPPING
    # ---------------------------------------------------------
    def crop_face(self, img_bgr, box, margin=0.25):
        (x1, y1, x2, y2) = box
        h, w = img_bgr.shape[:2]

        # Expand box
        w_box = x2 - x1
        h_box = y2 - y1

        x1 = max(int(x1 - w_box * margin), 0)
        y1 = max(int(y1 - h_box * margin), 0)
        x2 = min(int(x2 + w_box * margin), w)
        y2 = min(int(y2 + h_box * margin), h)

        return img_bgr[y1:y2, x1:x2]

    # ---------------------------------------------------------
    # FACEMESH LANDMARK EXTRACTION (468-landmark vector)
    # ---------------------------------------------------------
    def get_landmarks(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = img_bgr.shape
        points = []

        for lm in face_landmarks.landmark:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append([px, py])

        return np.array(points, dtype=np.float32)

    # ---------------------------------------------------------
    # ARC FACE EMBEDDING (returns 512-d normalized vector)
    # ---------------------------------------------------------
    def preprocess_arcface(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112))
        arr = face_resized.astype(np.float32)
        arr = (arr - 127.5) / 128.0
        # NHWC format
        arr = np.expand_dims(arr, axis=0)   # (1,112,112,3)
        return arr

    def get_embedding(self, face_bgr):
        inp = self.preprocess_arcface(face_bgr)
        out = self.session.run(None, {self.input_name: inp})
        emb = out[0].reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    # ---------------------------------------------------------
    # HIGH-LEVEL PROCESS: returns (embedding, landmarks)
    # ---------------------------------------------------------
    def process(self, img_bgr):
        box = self.detect_face(img_bgr)
        if box is None:
            return None, None, None

        face_crop = self.crop_face(img_bgr, box)

        emb = self.get_embedding(face_crop)
        lm = self.get_landmarks(face_crop)

        return emb, lm, face_crop
