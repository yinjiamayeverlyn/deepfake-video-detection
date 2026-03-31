import streamlit as st

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🎬",
    layout="wide"
)

# ======================
# AUTH
# ======================
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    st.title("Deepfake Detection - Access Required")

    with st.form("login_form"):
        password = st.text_input("Enter Password:", type="password")
        login_clicked = st.form_submit_button("Login")

        if login_clicked:
            if password == st.secrets["APP_PASSWORD"]:
                st.session_state["auth"] = True
                st.success("Access granted")
            else:
                st.error("Wrong password")

    st.stop()

# ======================
# IMPORTS
# ======================
import os
import cv2
import torch
import tempfile
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

# ======================
# DEVICE
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    model = torch.load("best_deepfake_model.pth", map_location=DEVICE)
    model.eval()
    return model

model = load_model()

# ======================
# LOAD FACE DETECTOR
# ======================
@st.cache_resource
def load_mtcnn():
    return MTCNN(keep_all=False, device=DEVICE)

mtcnn = load_mtcnn()

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ======================
# FACE CROP (FIXED VERSION)
# ======================
def crop_face(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is None:
        return None

    x1, y1, x2, y2 = boxes[0]

    w = x2 - x1
    h = y2 - y1

    # make square
    size = max(w, h)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # margin 0.3
    size = size * 1.3

    new_x1 = int(cx - size / 2)
    new_y1 = int(cy - size / 2)
    new_x2 = int(cx + size / 2)
    new_y2 = int(cy + size / 2)

    # FIX: if out of bound → fallback to tight box
    if new_x1 < 0 or new_y1 < 0 or new_x2 > frame.shape[1] or new_y2 > frame.shape[0]:
        new_x1 = int(max(0, x1))
        new_y1 = int(max(0, y1))
        new_x2 = int(min(frame.shape[1], x2))
        new_y2 = int(min(frame.shape[0], y2))

    face = frame[new_y1:new_y2, new_x1:new_x2]

    if face.size == 0:
        return None

    return face

# ======================
# PREDICT
# ======================
def predict(frames):

    preds = []

    for frame in frames:
        img = transform(frame).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img)
            prob = torch.sigmoid(output).item()

        preds.append(prob)

    avg = np.mean(preds)

    label = "Fake" if avg > 0.5 else "Real"
    confidence = abs(avg - 0.5) * 200

    return label, confidence

# ======================
# UI
# ======================
st.title("🎬 Deepfake Video Detection")

video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if video_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("Detect"):

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps)

        frames = []
        count = 0

        with st.spinner("Processing video..."):

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % interval == 0:
                    face = crop_face(frame)

                    if face is not None:
                        frames.append(face)

                count += 1

        cap.release()

        if len(frames) == 0:
            st.error("No face detected")
        else:
            st.success(f"{len(frames)} faces extracted")

            label, confidence = predict(frames)

            st.subheader(f"Prediction: {label}")
            st.subheader(f"Confidence: {confidence:.2f}%")
