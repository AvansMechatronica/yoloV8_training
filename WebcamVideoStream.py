# 🚀 Realtime Objectdetectie met je eigen YOLOv8-model
# ----------------------------------------------------
# Dit script opent je webcam en voert objectdetectie uit met een YOLOv8-model.
# Het toont de bounding boxes + objectnamen in een venster en print de resultaten
# ook live in de terminal, inclusief coördinaten en betrouwbaarheid (confidence).

# 📦 Importeren van benodigde modules
from ultralytics import YOLO   # YOLOv8 voor objectdetectie
import cv2                     # OpenCV voor beeldverwerking en webcamtoegang
import time                    # Tijdmodule om FPS (frames per seconde) te berekenen

# 🔹 MODELKEUZE
# Zet de waarde van 'if 0' naar 'if 1' als je je eigen getrainde model wilt gebruiken.
# YOLOv8n.pt is een klein, voorgedefinieerd model dat standaardobjecten herkent (zoals mensen, auto's, etc.)
if 0:
    MODEL_PATH = "runs/detect/yolo8_tensorboard_run/weights/best.pt"  # <-- Jouw eigen model
else:
    MODEL_PATH = "yolov8n.pt"  # <-- Standaard YOLOv8 Nano model

# 🧠 Laad het YOLOv8-model
model = YOLO(MODEL_PATH)

# 🎥 Open de webcam
# cv2.VideoCapture(0) gebruikt de standaardcamera van je systeem.
# Gebruik een ander getal (1, 2, ...) voor een tweede of externe camera.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kan de webcam niet openen.")
    exit()

print("✅ Webcam gestart — druk op 'q' om te stoppen.")

# ⏱️ Starttijd bijhouden (voor FPS-berekening)
start = time.time()

# 🔁 Oneindige lus voor realtime detectie
while True:
    # 🎞️ Lees één frame van de webcam
    ret, frame = cap.read()
    if not ret:
        print("❌ Geen frame ontvangen van webcam.")
        break

    # 🔍 Objectdetectie uitvoeren met het model
    # De functie retourneert een lijst met detectieresultaten (meestal 1 per frame)
    results = model(frame)

    # 🧾 Loop door alle detectieresultaten
    for result in results:
        boxes = result.boxes  # Bevat alle bounding boxes in het frame
        for box in boxes:
            cls = int(box.cls[0])           # Klasse-index (bijv. 0 = persoon)
            label = result.names[cls]       # Klassenaam (tekstlabel)
            conf = float(box.conf[0])       # Vertrouwen van het model (0.0–1.0)
            xyxy = box.xyxy[0].tolist()     # Coördinaten [x1, y1, x2, y2]

            # 📟 Print detectieresultaten naar de terminal
            print(f"{label} ({conf:.2f}) op {xyxy}")

    # 🎯 Teken bounding boxes, labels, en confidence op het beeld
    annotated_frame = results[0].plot()

    # ⚡ Bereken en toon FPS (frames per seconde)
    fps = 1 / (time.time() - start)
    start = time.time()  # Reset timer voor volgende frame
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # 🖼️ Toon het beeld in een OpenCV-venster
    cv2.imshow("YOLOv8 Live - Eigen model", annotated_frame)

    # ⏹️ Stop het programma als de gebruiker op 'q' drukt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Netjes afsluiten en resources vrijgeven
cap.release()
cv2.destroyAllWindows()
