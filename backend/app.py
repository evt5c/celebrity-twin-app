import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.predict import CelebrityPredictor
import numpy as np
import cv2
import base64

app = FastAPI()
predictor = CelebrityPredictor()

# Allow your phone, PC, or any device on LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    content = await file.read()
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = predictor.predict(img)

    # Face preview (your face)
    if predictor.last_crop is None:
        face_preview = None
    else:
        _, buffer = cv2.imencode(".jpg", predictor.last_crop)
        face_preview = base64.b64encode(buffer).decode("utf-8")

    # --- NEW: Celebrity thumbnail ---
    celeb_id = result["top_match"]["celebrity_id"]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    celeb_dir = os.path.abspath(os.path.join(base_dir, "..", "dataset", celeb_id))
    
    print("LOOKING FOR CELEB DIR:", celeb_dir)


    celeb_thumbnail_b64 = None
    if os.path.exists(celeb_dir):
        # pick first image in folder
        for f in os.listdir(celeb_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(celeb_dir, f)
                celeb_img = cv2.imread(img_path)

                if celeb_img is not None:
                    celeb_img = cv2.resize(celeb_img, (300, 300))
                    _, buffer = cv2.imencode(".jpg", celeb_img)
                    celeb_thumbnail_b64 = base64.b64encode(buffer).decode("utf-8")
                break

    return {
        "top_match": result["top_match"],
        "top_5": result["top_5"],
        "feature_similarity": result["feature_similarity"],
        "face_preview": face_preview,
        "celebrity_preview": celeb_thumbnail_b64  # <-- new
    }
