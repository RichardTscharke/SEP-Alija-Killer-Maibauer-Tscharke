import sys
import os

# Modules that are needed
from src.models.ResNetLight2 import ResNetLightCNN2
from src.train import main as train

def ask_user(question):
    """Ask a yes/no question via input() and return their answer as True/False."""
    valid = {"yes": True, "y": True, "ye": True, "ja": True, "j": True,
             "no": False, "n": False}
    
    while True:
        choice = input(f"{question} [y/n]: ").lower()
        if choice in valid:
            return valid[choice]
        else:
            print("Bitte antworte mit 'y' oder 'n'.\n")

def main():
    print("==========================================")
    print("   EMOTION RECOGNITION - HAUPTMENÜ      ")
    print("==========================================")

    # 1. Training
    if ask_user("Do you want to retrain the model on your own data?"):
        print("\n--- Starting Training ---")
       
        try:
            train() 
        except Exception as e:
            print(f"Fehler beim Training: {e}")