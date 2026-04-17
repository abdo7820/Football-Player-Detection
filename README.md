# ⚽ Football Player Detection using YOLOv8

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8-blue">
  <img src="https://img.shields.io/badge/Task-Object%20Detection-green">
</p>

---

## 📖 Overview

This project implements a **deep learning-based football player detection system** using the YOLOv8 architecture.
It aims to provide accurate and efficient detection of players in both images and videos, enabling applications in **sports analytics, tracking, and real-time vision systems**.

---

## 🎯 Key Features

* ⚡ Real-time player detection
* 🎥 Video processing with bounding boxes
* 🖼️ Image inference support
* 📊 Model evaluation metrics (mAP, Precision, Recall)
* 🔍 Optimized for performance and accuracy

---

## 🧠 Model Details

| Parameter  | Value     |
| ---------- | --------- |
| Model      | YOLOv8s   |
| Image Size | 640       |
| Epochs     | 80        |
| Task       | Detection |

---

## 📂 Dataset

The model is trained on a custom football dataset provided via Roboflow.

### Structure:

```id="y74g9o"
train/
valid/
test/
```

---

## ⚙️ Installation

```bash id="1t6e4r"
pip install ultralytics opencv-python
```

---

## 🏋️ Training

```python id="u9k4cu"
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640,
    batch=32
)
```

---

## 🖼️ Image Inference

```python id="l0j64r"
model.predict(source="image.jpg", conf=0.5, show=True)
```

---

## 🎥 Video Inference

```python id="6n9i9y"
model.predict(source="video.mp4", save=True)
```

---

## 🔴 Real-Time Detection (Local Machine)

```python id="0c2m0p"
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5)
    annotated = results[0].plot()

    cv2.imshow("Football Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 Results

* **mAP@50:** ~0.80+
* Strong detection performance in football scenes
* Robust against different camera angles

---

## 🚀 Future Work

* 🎯 Multi-object tracking (player IDs)
* 🧠 Action recognition (passes, shots)
* 📊 Advanced match analytics
* 🧍‍♂️ Detect referees and ball

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit pull requests.

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgments

* Ultralytics YOLOv8
* Roboflow Dataset Platform

---
