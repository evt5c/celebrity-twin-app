import cv2
import os
from face_utils import FaceProcessor

if __name__ == "__main__":
    img_path = "C:/Users/stell/celebtwin/test.jpg"  # put a face image here
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Failed to load image")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(base_dir, "..", "models", "arcface.onnx")

    fp = FaceProcessor(onnx_path)
    embedding, landmarks, face_crop = fp.process(img)

    print("Embedding shape:", embedding.shape)
    print("Landmarks shape:", landmarks.shape)

    cv2.imshow("Face Crop", face_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
