from ultralytics import YOLO

# Laad een voorgetraind model (je kunt ook 'yolov8m.pt', 'yolov8l.pt', enz. kiezen)
model = YOLO('yolov8n.pt')

# Train het model
results = model.train(
    data='SimpleFruits-1/data.yaml',  # pad naar YAML-bestand
    epochs=50,                 # aantal epochs
    imgsz=640,                 # afbeeldingsgrootte
    batch=16,                  # batchgrootte
    name='my_yolo8_model',     # naam van de runs-folder
)
