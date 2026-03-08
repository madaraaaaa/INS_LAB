# INS_LAB
# EMG Gesture Recognition Pipeline

This project implements a complete pipeline for **High-Density Surface EMG (sEMG) gesture recognition**, including signal preprocessing, deep learning model training, a web interface for inference, and containerized deployment using Docker.

---

# Project Overview

The system processes **128-channel High-Density EMG signals** to classify **6 different hand gestures** using a Convolutional Neural Network (CNN).

The pipeline includes:

- Signal preprocessing and segmentation
- CNN-based gesture classification
- Web interface for gesture prediction
- Docker-based deployment for reproducibility

---

# Phase 1: Data Preprocessing & Segmentation

### Filtering

The raw **128-channel sEMG signals** were cleaned using signal processing techniques:

- **50 Hz Notch Filter**  
  Removes electrical power-line interference.

- **20–500 Hz Bandpass Filter**  
  Isolates the frequency range corresponding to muscle activity.

### Silence Removal

To remove inactive signal segments:

1. Signals were **rectified** and **averaged**.
2. A **512-sample smoothing window** was applied to bridge zero crossings.
3. This produced a stable **signal envelope**.
4. A **dynamic threshold** was used to remove silent periods.

Only segments containing **active gestures** were retained.

### Windowing

The continuous signal was divided into smaller windows:

- **Window size:** 250 ms  
- **Samples per window:** 512  

These windows form the **input tensors** for the neural network.

---

# Phase 2: Deep Learning Model Training

### Data Splitting

To evaluate real-world generalization:

- **Training / Validation:** Subjects **P1** and **P2**
- **Testing:** Subject **P3** (completely unseen during training)

This setup evaluates **cross-subject generalization**.

### Model Architecture

A **Convolutional Neural Network (CNN)** was designed with:

- 3 Convolutional blocks
  - 16 filters
  - 32 filters
  - 64 filters
- **MaxPooling layers**
- **Dropout** for regularization
- **Dense classification layer**

The model predicts **6 hand gestures** from the **128-electrode EMG grid**.

### Results

| Metric | Result |
|------|------|
| Training Accuracy | ~92% |
| Cross-Subject Test Accuracy | ~52% |

These results demonstrate that deep learning can automatically learn **spatial muscle activation patterns** from high-density EMG signals.

---

# Phase 3: Application Development

### Inference Pipeline

An inference script loads the **trained CNN model** and processes new EMG recordings.

The pipeline:

1. Load trained model
2. Preprocess uploaded signal
3. Generate gesture prediction

### Web Interface

A lightweight web application allows users to interact with the model.

Features:

- Upload raw **`.mat` EMG files**
- Run model inference
- Display predicted gesture

Server runs on:

```
http://localhost:5000
```

---

# Phase 4: Containerization & Deployment (Docker)

### Docker Environment

A **Docker container** packages the entire system including:

- Linux environment
- Python runtime
- PyTorch
- Preprocessing pipeline
- Web application
- Trained model

This guarantees **reproducibility across machines**.

### Run the Application

Pull and run the Docker container:

```bash
docker run -p 5000:5000 madaraa/emg-app:v1
```

Then open:

```
http://localhost:5000
```

---

# Model Weights

Due to file size limitations on GitHub, the trained model weights are hosted on Google Drive.

Download the model weights here:

https://drive.google.com/file/d/1mIHyHU0lBXujammZAFpyHZyBqBQdGoEA/view?usp=drive_link

After downloading, place the weights file in the following directory:

```
models/
```

Example structure:

```
project/
│
├── models/
│   └── model.pth
├── app/
├── preprocessing/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Tech Stack

- Python
- PyTorch
- NumPy
- SciPy
- Flask (Web Interface)
- Docker

---

# Summary

This project demonstrates a full **end-to-end machine learning system**:

1. **Signal Processing** for EMG data cleaning and segmentation
2. **Deep Learning** for gesture classification
3. **Web Application** for interactive predictions
4. **Docker Deployment** for portable and reproducible execution

---
