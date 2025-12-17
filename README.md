🎭 Celebrity Twin Recognition System

**Face Similarity Analysis Using Deep Learning**

## 1. Introduction

The *Celebrity Twin Recognition System* is a computer vision application that identifies which public figure a user most closely resembles based on facial features.
This project applies modern face recognition techniques using deep learning and similarity search algorithms to compare a user’s face against a curated celebrity dataset.

The system is designed as an **educational AI project** to demonstrate how face embeddings, feature similarity, and image processing can be integrated into a complete application.

---

## 2. Objectives

The objectives of this project are:

* To implement a facial similarity system using deep learning models.
* To extract meaningful facial embeddings from images.
* To compare facial features using vector similarity search.
* To present similarity results in an intuitive and understandable way.
* To demonstrate real-world applications of Artificial Intelligence in image analysis.

---

## 3. System Overview

The application consists of three main components:

1. **Frontend**

   * Web-based interface for uploading images.
   * Displays top similarity matches and facial feature comparison.

2. **Backend**

   * Built using **FastAPI**.
   * Handles image upload, face detection, feature extraction, and prediction logic.

3. **AI Processing Pipeline**

   * Face detection using **MediaPipe**.
   * Face embedding extraction using **ArcFace (ONNX)**.
   * Similarity search using **FAISS** (cosine similarity).
   * Per-feature facial similarity analysis (eyes, nose, mouth, jawline, face shape).

---

## 4. Technologies Used

| Category             | Technology                 |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Backend Framework    | FastAPI                    |
| Face Detection       | MediaPipe                  |
| Face Embedding Model | ArcFace (ONNX)             |
| Similarity Search    | FAISS                      |
| Image Processing     | OpenCV                     |
| Frontend             | HTML, CSS, JavaScript      |
| Environment          | Python Virtual Environment |

---

## 5. Dataset

The dataset consists of multiple images per celebrity, organized in folders by name.
Each image is processed to extract:

* Facial embeddings (512-dimensional vectors)
* Facial landmarks for feature-level similarity analysis

**Note:**
The dataset and generated embeddings are excluded from this repository to maintain reasonable repository size and to respect data handling practices.

---

## 6. How the System Works

1. The user uploads a facial image.
2. The system detects the face and extracts facial landmarks.
3. A face embedding vector is generated using ArcFace.
4. The embedding is compared against stored celebrity embeddings using FAISS.
5. The system returns:

   * Top 1 most similar celebrity
   * Top 5 ranked similarities (unique celebrities)
   * Facial feature similarity breakdown

---

## 7. Output Example

The application outputs:

* Celebrity name
* Similarity percentage
* Top 5 closest matches
* Facial feature comparison (eyes, nose, mouth, jawline, face shape)
* Face preview of the user and the matched celebrity

<img width="2559" height="1297" alt="Screenshot 2025-12-16 222900" src="https://github.com/user-attachments/assets/3b8ba5c1-7d90-4d02-99c2-872fc126374e" />

<img width="2557" height="1291" alt="Screenshot 2025-12-16 222917" src="https://github.com/user-attachments/assets/e29fc808-bfbc-475d-b143-e715008cf2fc" />

<img width="1041" height="1154" alt="Screenshot 2025-12-16 225213" src="https://github.com/user-attachments/assets/3a2797f7-a842-404a-8de9-b0529f2df142" />

<img width="976" height="651" alt="Screenshot 2025-12-16 225221" src="https://github.com/user-attachments/assets/133f6437-33be-42c3-acd9-adeaddf8defb" />


---
## 8. Installation & Running the Application

### 8.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 8.2 Rebuild Embeddings (After Adding Dataset)

```bash
python build_index.py
```

### 8.3 Run the Backend Server

```bash
uvicorn backend.app:app --reload
```

### 8.4 Open Frontend

Open `frontend/index.html` in a browser.

---

## 9. Limitations

* Accuracy depends on dataset quality and lighting conditions.
* Facial similarity does not imply identity or exact resemblance.
* The system is intended for educational and demonstration purposes only.

---

## 10. Future Improvements

* Expand dataset with more diverse celebrities.
* Improve facial feature weighting for similarity scoring.
* Add real-time webcam input.
* Deploy the system as a standalone desktop application.

---

## 11. Conclusion

This project demonstrates how Artificial Intelligence can be applied to facial analysis and similarity matching.
By combining deep learning, computer vision, and efficient similarity search, the system provides an interactive and informative experience while showcasing practical AI implementation.

The Celebrity Twin Recognition System serves as a strong example of how theoretical AI concepts can be transformed into a functional and engaging application.

---

## 12. Author

**Name:** *evt5c*
**Field of Study:** Artificial Intelligence / Computer Science
**Purpose:** Academic Project Submission

---
