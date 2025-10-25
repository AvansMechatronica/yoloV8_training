"""
train_yolov8.py
---------------
Volledig script om een YOLOv8-model te trainen met je eigen dataset.

📘 Vereisten:
    pip install ultralytics torch

📁 Datasetstructuur:
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml

💡 Tip:
    - Pas de variabelen in de CONFIG-sectie aan voor je eigen project.
    - Resultaten en modelgewichten worden automatisch opgeslagen in: runs/detect/<RUN_NAME>/
"""

import os
from ultralytics import YOLO
import torch

# ========= CONFIG =========
DATA_YAML = "SimpleFruits-1/data.yaml"  # Pad naar YAML-bestand van je dataset
MODEL_NAME = "yolov8n.pt"               # Basismodel: n (nano), s (small), m (medium), l (large), x (extra large)
EPOCHS = 50                             # Hoe vaak het model over de dataset traint
IMG_SIZE = 640                          # Beeldresolutie (meer = nauwkeuriger, maar trager)
BATCH_SIZE = 16                         # Aantal afbeeldingen dat tegelijk verwerkt wordt
RUN_NAME = "my_yolov8_training"         # Naam voor deze trainingssessie
# ===========================


def check_dataset_structure():
    """
    Controleer of de datasetstructuur correct is en data.yaml aanwezig is.
    Dit voorkomt fouten tijdens training.
    """
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"❌ data.yaml niet gevonden op pad: {DATA_YAML}")

    base_dir = os.path.dirname(DATA_YAML)
    train_images = os.path.join(base_dir, "train/images")
    val_images = os.path.join(base_dir, "val/images")

    if not os.path.exists(train_images):
        raise FileNotFoundError(f"❌ Train images-map niet gevonden: {train_images}")
    if not os.path.exists(val_images):
        raise FileNotFoundError(f"❌ Validation images-map niet gevonden: {val_images}")

    print("✅ Datasetstructuur correct gevonden.\n")


def train_yolov8():
    """
    Train een YOLOv8-model met opgegeven instellingen.
    """
    # 🔍 Controleer of GPU beschikbaar is
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Apparaat: {device}\n")

    # ✅ Controleer dataset voordat training start
    check_dataset_structure()

    # 📦 Model laden
    print(f"📦 YOLO-model laden: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # 🚀 Start training
    print(f"🚀 Training starten ({EPOCHS} epochs, batch={BATCH_SIZE}, imgsz={IMG_SIZE})...\n")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        device=device,
    )

    print("\n✅ Training voltooid!")
    print(f"📁 Resultaten opgeslagen in: runs/detect/{RUN_NAME}/")
    print(f"🏋️‍♂️ Beste modelgewicht: runs/detect/{RUN_NAME}/weights/best.pt\n")


def test_trained_model(image_path):
    """
    Test het getrainde model op een voorbeeldafbeelding.
    Resultaten worden opgeslagen in 'runs/predict/'.
    """
    model_path = f"runs/detect/{RUN_NAME}/weights/best.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Getraind model niet gevonden: {model_path}")

    print(f"📂 Getraind model geladen: {model_path}")
    model = YOLO(model_path)

    print(f"🔍 Testen met afbeelding: {image_path}")
    results = model.predict(image_path, save=True)

    print("✅ Voorspelling voltooid!")
    print("📸 Resultaten opgeslagen in de map: runs/predict/\n")


# ================================
#  MAIN EXECUTIE
# ================================
if __name__ == "__main__":
    # 1️⃣ Train het model
    train_yolov8()

    # 2️⃣ Test het model (optioneel)
    # test_trained_model("test_images/example.jpg")
