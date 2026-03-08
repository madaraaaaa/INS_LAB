# INS_LAB
# EMG Gesture Recognition Pipeline

This project implements a complete pipeline for **High-Density sEMG gesture recognition**, including data preprocessing, deep learning training, a web interface for inference, and containerized deployment using Docker.

---

# Phase 1: Data Preprocessing & Segmentation

### Filtering
The raw **High-Density sEMG data (128 channels)** was cleaned using signal processing techniques:

- **50 Hz Notch Filter** – removes electrical power-line interference.
- **20–500 Hz Bandpass Filter** – isolates the frequency range corresponding to muscle activity.

### Silence Removal
To remove inactive segments:

1. Signals were **rectified** and **averaged**.
2. A **512-sample smoothing window** was applied to bridge signal zero-crossings.
3. This produced a stable **signal envelope**.
4. A **dynamic threshold** was applied to detect and remove silent segments.

Only **active gesture signals** were kept for further processing.

### Windowing
The continuous active signal was divided into smaller segments:

- **Window size:** 250 ms  
- **Samples per window:** 512

This created properly shaped **input tensors** for the neural network.

---

# Phase 2: Deep Learning (CNN) Training

### Data Splitting
To evaluate real-world generalization:

- **Training & Validation:** Subjects **P1** and **P2**
- **Testing:** Subject **P3** (completely unseen during training)

This setup tests the model's ability to generalize across different individuals.

### Model Architecture
A **Convolutional Neural Network (CNN)** was designed with:

- **3 Convolutional Blocks**
  - 16 filters
  - 32 filters
  - 64 filters
- **Max Pooling** layers
- **Dropout** for regularization
- **Dense classification head**

The model predicts **6 different hand gestures** from the 128-channel electrode grid.

### Results
- **Training accuracy:** ~92%
- **Cross-subject test accuracy:** ~52%

These results demonstrate that deep learning can **automatically learn spatial muscle activation patterns** from high-density EMG signals.

---

# Phase 3: Application Development

### Inference Pipeline
A dedicated inference script was implemented to:

- Load the **trained CNN model**
- Process **new unseen EMG data**
- Generate gesture predictions

### Web Interface
A lightweight **web application** was developed:

- Runs on **port 5000**
- Allows users to **upload `.mat` EMG files**
- Displays the **predicted gesture instantly**

This enables easy interaction with the trained model directly from a browser.

---

# Phase 4: Containerization & Deployment (Docker)

### Dockerfile
A **Dockerfile** was created to define the application environment.  
It bundles:

- Linux OS
- Python environment
- PyTorch
- Preprocessing scripts
- Web application
- Trained CNN model weights

### Building & Tagging
The Docker image was built locally and tagged as:

```
madaraa/emg-app:v1
```

### Global Deployment
The image (~4GB) was pushed to **Docker Hub**, enabling anyone to run the entire system without installing dependencies.

Run the application using:

```bash
docker run -p 5000:5000 madaraa/emg-app:v1
```

This command downloads the image and launches the **complete deep learning inference pipeline**.

---

# Summary

This project demonstrates a full **end-to-end machine learning pipeline**:

1. **Signal Processing** for EMG data cleaning and segmentation
2. **Deep Learning** for gesture classification
3. **Web Application** for user interaction
4. **Docker Deployment** for reproducible and portable execution
