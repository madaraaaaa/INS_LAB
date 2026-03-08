import numpy as np
import scipy.io
import scipy.signal
import pandas as pd
import torch

# --- CONFIGURATION ---
FS = 2048
WINDOW_SAMPLES = 512  # The exact size your model expects (250ms)

# --- 1. FILTERING FUNCTION (The Cleaning) ---
def apply_filters(data, fs=FS):
    # Bandpass 20-500Hz
    # We use fs=fs so we can use Hz directly (no need to divide by fs/2)
    b, a = scipy.signal.butter(4, [20, 500], btype='bandpass', fs=fs)
    data = scipy.signal.filtfilt(b, a, data, axis=1)

    # Notch 50Hz
    b_notch, a_notch = scipy.signal.iirnotch(50.0, 30.0, fs=fs)
    data = scipy.signal.filtfilt(b_notch, a_notch, data, axis=1)
    
    return data

# --- 2. SEGMENTATION FUNCTION (The Smart Cut) ---
def extract_active_segment(data, fs=FS):
    """
    Uses your 'Percentile' logic to remove silence.
    Calculates the envelope and keeps only data above the 50th percentile.
    """
    # 1. Calculate Envelope
    rectified = np.abs(data)
    avg_envelope = np.mean(rectified, axis=0) 

    # 2. Smooth (0.25s window)
    window_size = int(0.25 * fs)
    smooth_envelope = pd.Series(avg_envelope).rolling(window=window_size, center=True).mean().fillna(0).values

    # 3. Calculate Threshold (Median / 50th Percentile)
    threshold = np.percentile(smooth_envelope, 50)

    # 4. Create Mask & Cut
    mask = smooth_envelope > threshold
    cleaned_data = data[:, mask]
    
    # Safety Check: If we cut too much, warn the user
    if cleaned_data.shape[1] < WINDOW_SAMPLES:
        print("Warning: Signal is too short after trimming! Returning original.")
        return data 
        
    return cleaned_data

# --- 3. MAIN PREPROCESSING FUNCTION ---
def process_file_for_model(file_path_or_data):
    """
    Master function: Takes a file path (or raw array), cleans it, 
    cuts it, and formats it into the exact Tensor the model needs.
    """
    
    # A. Load Data
    if isinstance(file_path_or_data, str):
        # If it's a file path, load the .mat file
        mat = scipy.io.loadmat(file_path_or_data)
        # Find the variable name automatically (skipping __header__, etc.)
        var_name = [k for k in mat if not k.startswith('__')][0]
        raw_data = mat[var_name]
    else:
        # If it's already a numpy array
        raw_data = file_path_or_data

    # B. Apply Filters
    filtered_data = apply_filters(raw_data)

    # C. "Smart Cut" (Segmentation)
    # Note: If this is a 'Rest' file, we technically shouldn't cut it. 
    # But for a generic predictor, we usually assume the user is doing a gesture.
    # If you want to strictly skip cutting for rest, you'd need to know the label beforehand.
    segment = extract_active_segment(filtered_data)

    # D. Windowing (Get the Center Slice)
    # The model expects exactly 512 samples. We take the middle of the active segment.
    center = segment.shape[1] // 2
    start = center - (WINDOW_SAMPLES // 2)
    end = center + (WINDOW_SAMPLES // 2)
    
    window = segment[:, start:end] # Shape: (128, 512)

    # E. Convert to PyTorch Tensor
    # Shape becomes: (1, 1, 128, 512) -> (Batch, Channel, Height, Width)
    tensor_input = torch.tensor(window, dtype=torch.float32)
    tensor_input = tensor_input.unsqueeze(0).unsqueeze(0) 

    return tensor_input



