# 🚀 Realtime Objectdetectie met je eigen YOLOv8-model

from ultralytics import YOLO
import cv2

# 🔹 Pad naar je eigen getrainde model (pas dit aan!)
if 1:
    MODEL_PATH = "runs/detect/yolo8_tensorboard_run/weights/best.pt"
else:
    MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)

# 🎥 Open de webcam (0 = standaardcamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kan de webcam niet openen.")
    exit()

print("✅ Webcam gestart — druk op 'q' om te stoppen.")

# 🔁 Realtime detectie
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Geen frame ontvangen.")
        break

    # 🔍 Objectdetectie uitvoeren met jouw model
    results = model(frame)

    # 🎯 Bounding boxes + labels tekenen
    annotated_frame = results[0].plot()

    # 🖼️ Toon het beeld
    cv2.imshow("YOLOv8 Live - Eigen model", annotated_frame)

    # ⏹️ Stoppen met 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Opruimen
cap.release()
cv2.destroyAllWindows()
