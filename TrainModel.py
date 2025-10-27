"""
train_yolov8_tensorboard_fixed.py
---------------------------------
Volledig script om YOLOv8 te trainen met automatische TensorBoard-integratie en validatie.

Vereisten:
    pip install ultralytics torch tensorboard

Gebruik:
    python train_yolov8_tensorboard_fixed.py
"""

# ==========================================================
# 🧩 Imports
# ----------------------------------------------------------
# os, glob, subprocess en time zijn standaard Python-modules.
# torch wordt gebruikt om te detecteren of een GPU beschikbaar is.
# ultralytics bevat de YOLOv8-functionaliteit en instellingen.
# ==========================================================

import os
import glob
import subprocess
import time
import torch
from ultralytics import YOLO, settings


# ==========================================================
# ⚙️ CONFIGURATIE-INSTELLINGEN
# ----------------------------------------------------------
# Hier stel je de dataset, modelparameters en trainingsopties in.
# Pas deze waarden aan je project aan.
# ==========================================================

DATA_YAML = "SimpleFruits-1/data.yaml"      # Pad naar je data.yaml-bestand
MODEL_NAME = "yolov8n.pt"                   # Basismodel (YOLOv8 Nano)
EPOCHS = 30                                 # Aantal trainingsrondes (epochs)
IMG_SIZE = 640                              # Beeldresolutie (px)
BATCH_SIZE = 16                             # Hoeveel afbeeldingen tegelijk
RUN_NAME = "yolo8_tensorboard_fixed"        # Naam van de trainingsrun
PROJECT_DIR = "runs/detect"                 # Locatie voor resultaten
START_TENSORBOARD = True                    # Automatisch TensorBoard starten


# ==========================================================
# 🧰 Functies
# ----------------------------------------------------------
# Hieronder staan hulpfuncties voor datasetcontrole, TensorBoard
# opstarten en het trainen van het YOLO-model.
# ==========================================================

def check_dataset_structure():
    """
    Controleer of de dataset correct is opgezet.
    Verwacht structuur:
        dataset/
        ├── train/images/
        ├── train/labels/
        ├── valid/images/
        ├── valid/labels/
        └── data.yaml
    """
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"❌ YAML-bestand niet gevonden: {DATA_YAML}")

    base_dir = os.path.dirname(DATA_YAML)
    for folder in ["train/images", "valid/images"]:
        if not os.path.exists(os.path.join(base_dir, folder)):
            raise FileNotFoundError(f"❌ Map ontbreekt: {os.path.join(base_dir, folder)}")

    print("✅ Datasetstructuur correct.\n")


def start_tensorboard(logdir="runs"):
    """
    Start TensorBoard in de achtergrond.
    Hiermee kun je de trainingsgrafieken live bekijken op http://localhost:6006
    """
    print("📊 TensorBoard starten...")
    subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", "6006"])
    time.sleep(3)
    print("✅ TensorBoard draait op: http://localhost:6006\n")


def check_tfevents(logdir="runs"):
    """
    Controleer of er TensorBoard-logbestanden (events.out.tfevents) zijn aangemaakt.
    Zo weet je zeker dat de training correct gelogd is.
    """
    log_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents*"), recursive=True)
    if not log_files:
        print("⚠️  Geen TensorBoard logs gevonden. Controleer of training correct liep.")
    else:
        print(f"✅ {len(log_files)} TensorBoard logbestand(en) gevonden.\n")


def train_yolov8():
    """
    Hoofdfunctie: voert de volledige YOLOv8-training uit.
    Inclusief datasetcontrole, modelinrichting, TensorBoard-integratie en validatie.
    """
    # 🔍 Kies automatisch GPU (cuda) of CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Apparaat: {device}")

    # 📂 Controleer datasetstructuur
    check_dataset_structure()

    # 📈 Zorg dat YOLO logs naar TensorBoard schrijft
    settings.update({'tensorboard': True})

    # 🧠 Model laden
    print(f"📦 Model laden: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # ▶️ TensorBoard starten (optioneel)
    if START_TENSORBOARD:
        start_tensorboard("runs")

    # 🚀 Start de training
    print("🚀 Training starten...\n")
    results = model.train(
        data=DATA_YAML,          # YAML-bestand met klassen en paden
        epochs=EPOCHS,           # Aantal trainingsrondes
        imgsz=IMG_SIZE,          # Afbeeldingsgrootte
        batch=BATCH_SIZE,        # Batchgrootte
        name=RUN_NAME,           # Naam van de run
        project=PROJECT_DIR,     # Opslaglocatie
        device=device,           # GPU of CPU
        verbose=True,            # Toon details in terminal
    )

    # ✅ Samenvatting na training
    print("\n✅ Training voltooid!")
    run_dir = os.path.join(PROJECT_DIR, RUN_NAME)
    print(f"📁 Resultaten opgeslagen in: {run_dir}")

    # 📊 TensorBoard-logbestanden controleren
    check_tfevents(run_dir)

    print("📈 Bekijk live resultaten op: http://localhost:6006\n")


# ==========================================================
# 🚦 Hoofduitvoer
# ----------------------------------------------------------
# Dit zorgt dat de training start wanneer het script direct
# wordt uitgevoerd vanaf de terminal of IDE.
# ==========================================================

if __name__ == "__main__":
    train_yolov8()
