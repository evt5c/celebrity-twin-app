import os
import numpy as np
import faiss
from backend.face_utils import FaceProcessor
import cv2

# -------------------------------------------------
# Project-relative base directory
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMB_DIR = os.path.join(BASE_DIR, "embeddings")
ARC_PATH = os.path.join(BASE_DIR, "models", "arcface.onnx")


class CelebrityPredictor:
    def __init__(self, top_k=5):
        self.top_k = top_k

        # Load FAISS index
        self.index = faiss.read_index(os.path.join(EMB_DIR, "index.faiss"))

        # Load arrays
        self.X = np.load(os.path.join(EMB_DIR, "X_embeddings.npy"))
        self.L = np.load(os.path.join(EMB_DIR, "landmarks.npy"))
        self.ids = np.load(os.path.join(EMB_DIR, "celeb_ids.npy"))

        # Face processor
        self.fp = FaceProcessor(ARC_PATH)
        self.last_crop = None

    # -------------------------------------------------
    # Cosine similarity → percentage
    # -------------------------------------------------
    def cosine_to_percentage(self, cos_sim):
        return float(max(0, min(1, (cos_sim + 1) / 2))) * 100

    # -------------------------------------------------
    # Per-feature similarity
    # -------------------------------------------------
    def compute_feature_similarity(self, inp_landmarks, celeb_landmarks):
        N = len(celeb_landmarks) // 2
        celeb_landmarks = celeb_landmarks.reshape(N, 2)

        def norm(pts):
            xs, ys = pts[:, 0], pts[:, 1]
            w = xs.max() - xs.min() + 1e-6
            h = ys.max() - ys.min() + 1e-6
            return np.stack(
                [(xs - xs.min()) / w, (ys - ys.min()) / h],
                axis=1
            )

        A = norm(inp_landmarks)
        B = norm(celeb_landmarks)

        L = A.shape[0]

        def region_sim(idxs):
            diff = A[list(idxs)] - B[list(idxs)]
            dist = np.mean(np.linalg.norm(diff, axis=1))
            return max(0, min(1, 1 - dist)) * 100

        return {
            "eyes": region_sim(range(int(L * 0.07), int(L * 0.28))),
            "nose": region_sim(range(int(L * 0.00), int(L * 0.10))),
            "mouth": region_sim(range(int(L * 0.16), int(L * 0.33))),
            "jawline": region_sim(range(int(L * 0.33), int(L * 0.45))),
            "face_shape": region_sim(range(0, L)),
        }

    # -------------------------------------------------
    # MAIN PREDICTION (UNIQUE CELEBRITY RANKING)
    # -------------------------------------------------
    def predict(self, img_bgr):
        emb, landmarks, crop = self.fp.process(img_bgr)
        self.last_crop = crop

        if emb is None:
            return {"error": "no_face"}
        if landmarks is None:
            return {"error": "no_landmarks"}

        # Search more than top_k to allow deduplication
        search_k = self.top_k * 5
        q = emb.reshape(1, -1).astype(np.float32)
        D, I = self.index.search(q, search_k)

        # ---------------------------------------------
        # Deduplicate celebrities (keep best score)
        # ---------------------------------------------
        best_per_celebrity = {}

        for score, idx in zip(D[0], I[0]):
            celeb_id = self.ids[idx]
            sim = self.cosine_to_percentage(score)

            if celeb_id not in best_per_celebrity:
                best_per_celebrity[celeb_id] = {
                    "celebrity_id": celeb_id,
                    "similarity": sim,
                    "index": idx
                }
            else:
                if sim > best_per_celebrity[celeb_id]["similarity"]:
                    best_per_celebrity[celeb_id]["similarity"] = sim
                    best_per_celebrity[celeb_id]["index"] = idx

        # Sort by similarity
        unique_results = sorted(
            best_per_celebrity.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )

        # Take top K unique celebrities
        top_results = unique_results[:self.top_k]

        # Feature similarity from top match
        top_idx = top_results[0]["index"]
        feature_sim = self.compute_feature_similarity(
            landmarks,
            self.L[top_idx]
        )

        # Remove internal index before returning
        for r in top_results:
            r.pop("index", None)

        return {
            "top_match": top_results[0],
            "top_5": top_results,
            "feature_similarity": feature_sim
        }
