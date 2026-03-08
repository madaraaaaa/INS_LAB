import os
import torch
import gc  # <--- [NEW] Import Garbage Collector
from flask import Flask, request, render_template, redirect, url_for
from model import EMG_CNN              # Import your Blueprint
from preprocess import process_file_for_model  # Import your Cleaner

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
WEIGHTS_FILE = "emg_cnn_weights_finals.pth"
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

# Class Mapping
CLASS_NAMES = {
    0: 'Stone', 1: 'Paper', 2: 'Scissors', 
    3: 'Rock', 4: 'Pointing', 5: 'Rest'
}

# --- LOAD MODEL (Once at startup) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMG_CNN().to(device)

if os.path.exists(WEIGHTS_FILE):
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    model.eval()
    print("✅ Model Loaded Successfully!")
else:
    print(f"❌ Warning: {WEIGHTS_FILE} not found. Please put it in this folder.")

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def upload_file():
    # --- [MEMORY FIX] CLEAN BEFORE STARTING ---
    gc.collect() 
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # ------------------------------------------

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        input_tensor = None # Initialize variable for safety

        try:
            # 2. Preprocess
            input_tensor = process_file_for_model(file_path).to(device)
            
            # 3. Predict
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
            
            gesture_name = CLASS_NAMES[predicted_idx.item()]
            
            # 4. Cleanup File
            if os.path.exists(file_path):
                os.remove(file_path)

            # --- [MEMORY FIX] AGGRESSIVE CLEANUP ---
            # Delete heavy variables immediately
            del input_tensor
            del output
            gc.collect() # Force Python to free RAM
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # ---------------------------------------

            return render_template('index.html', result=gesture_name)

        except Exception as e:
            # Clean up even if there is an error
            if 'input_tensor' in locals(): del input_tensor
            gc.collect()
            return render_template('index.html', error=f"Memory/Processing Error: {str(e)}")

if __name__ == '__main__':
    # Disable debug mode to save a little memory in Docker
    app.run(host='0.0.0.0', debug=False, port=5000)