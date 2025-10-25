"""
train_yolov8_tensorboard_fixed.py
---------------------------------
Volledig script om YOLOv8 te trainen met automatische TensorBoard-integratie en validatie.

Vereisten:
    pip install ultralytics torch tensorboard

Gebruik:
    python train_yolov8_tensorboard_fixed.py
"""

import os
import glob
import subprocess
import time
import torch
from ultralytics import YOLO, settings


# ========= CONFIG =========
DATA_YAML = "SimpleFruits-1/data.yaml"      # Pad naar je data.yaml
MODEL_NAME = "yolov8n.pt"                   # Basismodel
EPOCHS = 30                                 # Aantal epochs
IMG_SIZE = 640                              # Beeldresolutie
BATCH_SIZE = 16                             # Batchgrootte
RUN_NAME = "yolo8_tensorboard_fixed"        # Naam voor training
PROJECT_DIR = "runs/detect"                 # Locatie waar resultaten worden opgeslagen
START_TENSORBOARD = True                    # Automatisch TensorBoard starten
# ===========================


def check_dataset_structure():
    """Controleer of dataset correct is opgezet."""
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"❌ YAML-bestand niet gevonden: {DATA_YAML}")

    base_dir = os.path.dirname(DATA_YAML)
    for folder in ["train/images", "valid/images"]:
        if not os.path.exists(os.path.join(base_dir, folder)):
            raise FileNotFoundError(f"❌ Map ontbreekt: {os.path.join(base_dir, folder)}")

    print("✅ Datasetstructuur correct.\n")


def start_tensorboard(logdir="runs"):
    """Start TensorBoard (in achtergrond)."""
    print("📊 TensorBoard starten...")
    subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", "6006"])
    time.sleep(3)
    print("✅ TensorBoard draait op: http://localhost:6006\n")


def check_tfevents(logdir="runs"):
    """Controleer of TensorBoard logs bestaan."""
    log_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents*"), recursive=True)
    if not log_files:
        print("⚠️  Geen TensorBoard logs gevonden. Controleer of training correct liep.")
    else:
        print(f"✅ {len(log_files)} TensorBoard logbestand(en) gevonden.\n")


def train_yolov8():
    """Train YOLOv8 met TensorBoard logging."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Apparaat: {device}")

    # Dataset check
    check_dataset_structure()

    # TensorBoard logging forceren
    settings.update({'tensorboard': True})

    # Model laden
    print(f"📦 Model laden: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # TensorBoard starten
    if START_TENSORBOARD:
        start_tensorboard("runs")

    print("🚀 Training starten...\n")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        project=PROJECT_DIR,
        device=device,
        verbose=True,
    )

    print("\n✅ Training voltooid!")
    run_dir = os.path.join(PROJECT_DIR, RUN_NAME)
    print(f"📁 Resultaten opgeslagen in: {run_dir}")

    # TensorBoard logs controleren
    check_tfevents(run_dir)

    print("📈 Bekijk live resultaten op: http://localhost:6006\n")


if __name__ == "__main__":
    train_yolov8()
