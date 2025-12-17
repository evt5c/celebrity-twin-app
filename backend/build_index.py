import os
import cv2
import numpy as np
from tqdm import tqdm
from face_utils import FaceProcessor
import faiss

# -------------------------------------------------
# Project-relative base directory
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "dataset")
EMB_DIR = os.path.join(BASE_DIR, "..", "embeddings")
ARC_PATH = os.path.join(BASE_DIR, "..", "models", "arcface.onnx")

os.makedirs(EMB_DIR, exist_ok=True)

# ----------------------------------------------
# Convert 468 landmark points → feature vector
# ----------------------------------------------
def landmarks_to_feature_vector(landmarks):
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]

    w = xs.max() - xs.min() + 1e-6
    h = ys.max() - ys.min() + 1e-6

    norm_x = (xs - xs.min()) / w
    norm_y = (ys - ys.min()) / h

    norm_landmarks = np.stack([norm_x, norm_y], axis=1)
    return norm_landmarks.flatten()

# ----------------------------------------------
# MAIN
# ----------------------------------------------
def build_index():
    fp = FaceProcessor(ARC_PATH)

    all_embeddings = []
    all_landmarks = []
    all_ids = []

    for celeb_id in sorted(os.listdir(DATA_DIR)):
        celeb_dir = os.path.join(DATA_DIR, celeb_id)
        if not os.path.isdir(celeb_dir):
            continue

        print(f"\nProcessing celebrity: {celeb_id}")

        for fname in tqdm(os.listdir(celeb_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = cv2.imread(os.path.join(celeb_dir, fname))
            if img is None:
                continue

            emb, lm, _ = fp.process(img)

            if emb is None or lm is None:
                continue

            all_embeddings.append(emb)
            all_landmarks.append(landmarks_to_feature_vector(lm))
            all_ids.append(celeb_id)

    X = np.vstack(all_embeddings)
    L = np.vstack(all_landmarks)
    ids = np.array(all_ids)

    np.save(os.path.join(EMB_DIR, "X_embeddings.npy"), X)
    np.save(os.path.join(EMB_DIR, "landmarks.npy"), L)
    np.save(os.path.join(EMB_DIR, "celeb_ids.npy"), ids)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X.astype(np.float32))
    faiss.write_index(index, os.path.join(EMB_DIR, "index.faiss"))

    print("\nDONE")

if __name__ == "__main__":
    build_index()
