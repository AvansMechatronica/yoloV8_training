# 📦 Importeer de YOLO-module uit Ultralytics
from ultralytics import YOLO

# 🔹 Laad een voorgetraind YOLOv8-model
#   Mogelijke modellen:
#   - 'yolov8n.pt' → Nano (zeer snel, minder nauwkeurig)
#   - 'yolov8s.pt' → Small (balans tussen snelheid en precisie)
#   - 'yolov8m.pt' → Medium (beter voor complexere datasets)
#   - 'yolov8l.pt' → Large (langzamer, hogere nauwkeurigheid)
#   - 'yolov8x.pt' → Extra Large (zeer nauwkeurig, vraagt veel GPU)
model = YOLO('yolov8n.pt')

# 🚀 Start de training
results = model.train(
    data='SimpleFruits-1/data.yaml',  # YAML-bestand dat paden en klassen bevat
    epochs=50,                        # Aantal keer dat het model de dataset doorloopt
    imgsz=640,                        # Grootte waarop de afbeeldingen worden geschaald
    batch=16,                         # Aantal afbeeldingen per batch (meer = sneller, maar meer RAM)
    name='my_yolo8_model',            # Naam van de map waarin de resultaten worden opgeslagen
)
