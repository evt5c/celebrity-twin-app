import cv2
from predict import CelebrityPredictor

if __name__ == "__main__":
    img_path = "C:/Users/stell/celebtwin/test.jpg"  # put a face image here
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Failed to load image")

    predictor = CelebrityPredictor()
    result = predictor.predict(img)

    print("Prediction result:")
    print(result)
