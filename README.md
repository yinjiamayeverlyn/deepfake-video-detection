# Deepfake Detection Web App
## Project Overview
This Deepfake Detection Web App was developed as a final year project to address the growing threat of deepfake videos. It enables users to upload videos and detect manipulations using deep learning models.

The system extracts faces from videos and predicts whether they are Real or Fake using an ensemble of Convolutional Neural Networks (CNNs). A downloadable report provides predictions, confidence scores, and technical analysis.

## Our Mission
To build an accessible, user-friendly platform that helps people detect and analyze deepfake videos using state-of-the-art AI technologies — promoting media authenticity and digital trust.

## Technology Stack
- Frontend: Streamlit (Python Web Framework)
- Backend: Python
- Deep Learning: Ensemble of CNNs for deepfake detection
- Deployment: Streamlit Cloud
- Computer Vision: OpenCV, MTCNN for face detection
- Data Visualization: Plotly

## Datasets
This project leverages publicly available deepfake datasets:
- FaceForensics++ – Provides manipulated videos for research.
- Celeb-DF – High-quality deepfake video dataset for evaluation.
- DFDC (Deepfake Detection Challenge) – Free, large-scale deepfake dataset for training and testing.
  
All datasets are freely available online for research purposes.

## Getting Started
This web app allows users to detect deepfake videos. It is currently public for anyone to try, though primarily developed for educational purposes.

### Step 1: Upload a Video
1. Go to the Upload Video tab.
2. Choose your video input method:
  - Upload from device: Select a .mp4, .mov, or .avi video file.
  - Provide video URL: Enter a shareable link from Google Drive or Dropbox.
3. Wait for the preview to appear on the screen.

### Step 2: Submit for Detection
1. Click Submit for Detection after uploading.
2. The system analyzes the video using the deep learning ensemble model.
3. You will receive:
  - Prediction: Real or Fake
  - Confidence score (probability)
  - Technical analysis (faces detected, timestamps)

### Step 3: Download the Report
1. Click Download Detection Report to save results as a PDF.
2. The report includes:
  - Prediction
  - Confidence score
  - Video frame timestamps

## Results & Evaluation
The model performance is evaluated using a subset of validation data from the datasets. Metrics include:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
The ensemble model uses a best-threshold approach to improve classification of real and fake faces.

## Deployment
- The app is deployed on Streamlit Cloud.
- Users can access it publicly via the provided link.
- The app integrates TFLite models to reduce inference time for large videos.

## Acknowledgements
- Dr. Rasha Ragheb Attaallah – Supervisor for guidance and support.
- Multimedia Department – Resources and feedback.
- Dataset providers: FaceForensics++, Celeb-DF, DFDC.
