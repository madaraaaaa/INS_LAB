import torch
import os
from model import EMG_CNN             # Import your architecture
from preprocess import process_file_for_model # Import your cleaning tool

# --- CONFIGURATION ---
WEIGHTS_FILE = "emg_cnn_weights_finals.pth"   # The file you downloaded from Colab
CLASS_NAMES = {
    0: 'Stone',
    1: 'Paper',
    2: 'Scissors',
    3: 'Rock',
    4: 'Pointing',
    5: 'Rest'
}

def load_trained_model(weights_path, device):
    """Loads the architecture and weights."""
    model = EMG_CNN().to(device)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Error: Cannot find {weights_path}. Did you download it?")
        
    # Load weights (map_location ensures it works on CPU even if trained on GPU)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Set to evaluation mode (Important!)
    print(f"✅ Model loaded from {weights_path}")
    return model

def predict_gesture(file_path):
    """
    Main function: Loads file -> Cleans it -> Predicts Gesture.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # 2. Load Model
    try:
        model = load_trained_model(WEIGHTS_FILE, device)
    except Exception as e:
        print(e)
        return

    # 3. Preprocess Data
    print(f"Processing file: {file_path}...")
    try:
        # This function handles loading .mat, filtering, and windowing
        input_tensor = process_file_for_model(file_path).to(device)
    except Exception as e:
        print(f"❌ Preprocessing Error: {e}")
        return

    # 4. Run Prediction
    with torch.no_grad():
        output = model(input_tensor)
        # Get the highest probability
        _, predicted_idx = torch.max(output, 1)
        
    # 5. Show Result
    idx = predicted_idx.item()
    gesture_name = CLASS_NAMES[idx]
    
    print("\n" + "="*30)
    print(f"🎉 PREDICTION: {gesture_name.upper()} (Class {idx})")
    print("="*30 + "\n")

# --- EXECUTION SECTION ---
if __name__ == "__main__":
    # CHANGE THIS to the path of a file you want to test
    test_file = "test2.otb+.mat" 
    
    # Check if the test file exists before running
    if os.path.exists(test_file):
        predict_gesture(test_file)
    else:
        print(f"⚠️ Warning: Could not find '{test_file}'.")
        print("Please put a .mat file in this folder and update 'test_file' variable.")