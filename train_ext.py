"""
train_yolov8.py
---------------
Volledig script om een YOLOv8-model te trainen met je eigen dataset.

Vereisten:
    pip install ultralytics

Datasetstructuur:
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml
"""

import os
from ultralytics import YOLO
import torch

# ========= CONFIG =========
DATA_YAML = "SimpleFruits-1/data.yaml"     # Pad naar je YAML-bestand
MODEL_NAME = "yolov8n.pt"           # Basismodel (n = klein, m = medium, l = groot)
EPOCHS = 50                         # Aantal epochs
IMG_SIZE = 640                      # Beeldresolutie
BATCH_SIZE = 16                     # Batchgrootte
RUN_NAME = "my_yolov8_training"     # Naam voor de run (output map)
# ===========================


def check_dataset_structure():
    """Controleer of datasetmap en YAML-bestand bestaan."""
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"❌ data.yaml niet gevonden op pad: {DATA_YAML}")

    base_dir = os.path.dirname(DATA_YAML)
    train_images = os.path.join(base_dir, "train/images")
    val_images = os.path.join(base_dir, "val/images")

    if not os.path.exists(train_images):
        raise FileNotFoundError(f"❌ Train images map niet gevonden: {train_images}")
    if not os.path.exists(val_images):
        raise FileNotFoundError(f"❌ Validation images map niet gevonden: {val_images}")

    print("✅ Datasetstructuur correct.")


def train_yolov8():
    """Train een YOLOv8-model met opgegeven instellingen."""
    # Controleer GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Apparaat: {device}")

    # Controleer dataset
    check_dataset_structure()

    # Laad model
    print(f"📦 Model laden: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # Train model
    print(f"🚀 Training starten voor {EPOCHS} epochs...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        device=device,
    )

    print("✅ Training voltooid!")
    print(f"📁 Resultaten opgeslagen in: runs/detect/{RUN_NAME}/")


def test_trained_model(image_path):
    """Test het getrainde model op een voorbeeldafbeelding."""
    model_path = f"runs/detect/{RUN_NAME}/weights/best.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Getraind model niet gevonden: {model_path}")

    model = YOLO(model_path)
    print(f"🔍 Testen met afbeelding: {image_path}")
    results = model.predict(image_path, save=True)

    print("✅ Voorspelling voltooid. Resultaten opgeslagen in de map 'runs/predict'.")


if __name__ == "__main__":
    # 1️⃣ Train het model
    train_yolov8()

    # 2️⃣ Test het model (optioneel)
    # test_trained_model("test_images/example.jpg")
