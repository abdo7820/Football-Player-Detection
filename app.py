import cv2
from ultralytics import YOLO

# تحميل الموديل
model = YOLO(r'D:\instant\CV\Projects\Football Player Detection\best.pt')   # حط الموديل عندك

# فتح الكاميرا
cap = cv2.VideoCapture(r'D:\instant\CV\Projects\Football Player Detection\video.mp4\video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # prediction
    results = model.predict(frame, imgsz=640, conf=0.5)

    # رسم النتائج
    annotated_frame = results[0].plot()

    # عرض الفيديو
    cv2.imshow("YOLO Real-Time", annotated_frame)

    # خروج عند الضغط على q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()