import streamlit as st
import os
import cv2
import torch
import tempfile
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms, models
import torch.nn as nn
import plotly.graph_objects as go

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
# SESSION STATE (TOGGLE)
# ======================
if "show_all_faces" not in st.session_state:
    st.session_state.show_all_faces = False

# ======================
# DEVICE
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# MODEL ARCHITECTURE
# ======================
def create_model():
    efficientnet = models.efficientnet_b0(weights=None)
    efficientnet.classifier = nn.Sequential(
        nn.Linear(efficientnet.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1)
    )
    return efficientnet.to(DEVICE)

MODEL_PATH = "best_deepfake_model.pth"

@st.cache_resource
def load_model():
    model = create_model()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# ======================
# FACE DETECTOR
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
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# FACE CROP
# ======================
def crop_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is None:
        return None

    x1, y1, x2, y2 = boxes[0]
    w, h = x2 - x1, y2 - y1
    size = max(w, h) * 1.3
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    new_x1 = int(max(0, cx - size / 2))
    new_y1 = int(max(0, cy - size / 2))
    new_x2 = int(min(frame.shape[1], cx + size / 2))
    new_y2 = int(min(frame.shape[0], cy + size / 2))

    face = frame[new_y1:new_y2, new_x1:new_x2]

    if face.size == 0:
        return None

    return face

# ======================
# PREDICT (FAKE PROBABILITY)
# ======================
def predict_video(frames):
    if len(frames) == 0:
        return 0

    frames_tensor = torch.stack([transform(f) for f in frames]).to(DEVICE)

    with torch.no_grad():
        outputs = model(frames_tensor)
        outputs = outputs.view(-1)

        outputs_max = outputs.max()
        outputs_mean = outputs.mean()

        final_score = 0.7 * outputs_max + 0.3 * outputs_mean

        fake_prob = torch.sigmoid(final_score).item() * 100

    return fake_prob

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
        interval = int(fps) if fps > 0 else 1

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
                
            # ======================
            # SHOW FACES (EXPANDER STYLE)
            # ======================
            st.subheader("Extracted Faces")
            
            total_faces = len(frames)
            
            # --- Show first 15 faces ---
            preview_faces = frames[:15]
            cols = st.columns(5)
            
            for i, face in enumerate(preview_faces):
                col = cols[i % 5]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                col.image(
                    face_rgb,
                    caption=f"Face {i+1}",
                    use_container_width=True
                )
            
            # --- Show remaining faces in expander ---
            if total_faces > 15:
                st.caption(f"Showing 15 of {total_faces} extracted faces")
            
                with st.expander(f"View remaining {total_faces - 15} faces"):
                    cols_all = st.columns(5)
            
                    for i, face in enumerate(frames[15:], start=16):
                        col = cols_all[(i - 16) % 5]
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        col.image(
                            face_rgb,
                            caption=f"Face {i}",
                            use_container_width=True
                        )
            
            # ======================
            # PREDICTION
            # ======================
            fake_prob = predict_video(frames)

            # ======================
            # RESULT + GAUGE
            # ======================
            st.subheader(f"Fake Probability: {fake_prob:.2f}%")
            
            # --- Determine label & color ---
            if fake_prob > 70:
                status = "Highly likely FAKE"
                color = "red"
                st.error(status)
            elif fake_prob > 40:
                status = "Suspicious (uncertain)"
                color = "yellow"
                st.warning(status)
            else:
                status = "Likely REAL"
                color = "green"
                st.success(status)
            
            # --- Create smooth gradient steps manually ---
            gradient_steps = []
            for i in range(0, 101, 5):
                if i < 50:
                    r = 182 + int((255 - 182) * (i / 50))
                    g = 239 + int((233 - 239) * (i / 50))
                    b = 162 + int((169 - 162) * (i / 50))
                else:
                    r = 255
                    g = 233 - int((233 - 182) * ((i - 50) / 50))
                    b = 169 - int((169 - 166) * ((i - 50) / 50))
            
                hex_color = f'#{r:02X}{g:02X}{b:02X}'
                gradient_steps.append({'range': [i, i + 5], 'color': hex_color})
            
            # --- Gauge size ---
            gauge_font = 52
            
            # --- Create figure ---
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fake_prob,
            
                number={
                    'font': {'size': gauge_font},
                    'valueformat': '.2f',
                    'suffix': '%'
                },
            
                domain={'x': [0, 1], 'y': [0, 1]},  # full domain for easier positioning
            
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': "lightgray",
                        'tickfont': {'size': 18}
                    },
                    'bar': {'color': color, 'thickness': 0.25},
                    'bgcolor': 'white',
                    'steps': gradient_steps,
                    'borderwidth': 1,
                    'bordercolor': '#ddd',
                    'threshold': {
                        'line': {'color': color, 'width': 4},
                        'thickness': 1.0,
                        'value': fake_prob
                    }
                }
            ))
            
            # --- Add status text below the number ---
            fig.update_layout(
                height=420,
                margin=dict(t=40, b=40, l=40, r=40),
                annotations=[
                    dict(
                        x=0.5,               # center horizontally
                        y=0.22,              # lower than number (adjust as needed)
                        text=f"<b>{status}</b>",
                        showarrow=False,
                        font=dict(size=32, color=color)
                    )
                ]
            )
            # --- Show in Streamlit ---
            st.plotly_chart(fig, use_container_width=True)
