import easygui
import os

# Huidige map waar het script draait
current_path = os.getcwd()

# Open dialoog in die map
folder_path = easygui.diropenbox(
    msg="Selecteer een map:",
    title="Map selecteren",
    default=current_path
)

if folder_path:
    print("Geselecteerde map:", folder_path)
else:
    print("Geen map geselecteerd.")
