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
import urllib.request
import datetime
from zoneinfo import ZoneInfo

st.title("🎬 Deepfake Video Detection")

st.markdown("<h2 style='text-align:center;'>Upload Video for Deepfake Detection</h2>", unsafe_allow_html=True)

# ======================
# INPUT METHOD
# ======================
upload_option = st.radio(
    "Select video input method:",
    ["Upload from device", "Provide video URL"]
)

video_path = None
video_filename = None
valid_video = False

# ======================
# DEVICE UPLOAD
# ======================
if upload_option == "Upload from device":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            video_path = tfile.name
            video_filename = uploaded_file.name
            valid_video = True
        except:
            st.error("Failed to process uploaded file.")

# ======================
# URL INPUT
# ======================
elif upload_option == "Provide video URL":
    video_url = st.text_input("Paste video URL (Google Drive / Dropbox)")

    if video_url:
        try:
            if "dropbox.com" in video_url:
                video_url = video_url.replace("dl=0", "dl=1")
                video_url = video_url.replace("www.dropbox.com", "dl.dropboxusercontent.com")

            elif "drive.google.com" in video_url:
                if "/d/" in video_url:
                    file_id = video_url.split("/d/")[1].split("/")[0]
                elif "id=" in video_url:
                    file_id = video_url.split("id=")[1].split("&")[0]
                else:
                    raise ValueError()

                video_url = f"https://drive.google.com/uc?export=download&id={file_id}"

            else:
                st.error("Only Google Drive and Dropbox links are supported.")
                st.stop()

            video_filename = os.path.basename(video_url.split("?")[0])
            video_path = os.path.join(tempfile.gettempdir(), video_filename)

            urllib.request.urlretrieve(video_url, video_path)
            valid_video = True

        except:
            st.error("Unable to download video. Ensure link is public.")

# ======================
# PROCESS VIDEO
# ======================
if valid_video and video_path and os.path.exists(video_path):

    st.video(video_path)
    st.markdown(f"**Video source:** `{video_filename}`")

    malaysia_time = datetime.datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
    upload_time = malaysia_time.strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"**Upload time:** `{upload_time}`")

    # --- duration check ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps else 0
    cap.release()

    if "is_detecting" not in st.session_state:
        st.session_state.is_detecting = False

    if st.button("Submit for Detection"):

        if st.session_state.is_detecting:
            st.warning("Already processing. Please wait.")
            st.stop()

        st.session_state.is_detecting = True

        try:
            if duration < 4:
                st.error("Video too short (<4 seconds).")
                st.session_state.is_detecting = False
                st.stop()

            frames = []
            cap = cv2.VideoCapture(video_path)
            interval = int(fps) if fps > 0 else 1
            count = 0

            with st.spinner("Processing video..."):

                faces_dir = tempfile.mkdtemp(prefix="faces_")

                if "temp_dirs" not in st.session_state:
                    st.session_state.temp_dirs = []

                st.session_state.temp_dirs.append(faces_dir)

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

            # ======================
            # SHOW FACES (EXPANDER BACK 🔥)
            # ======================
            if len(frames) == 0:
                st.error("No face detected.")
            else:
                st.success(f"{len(frames)} faces extracted")

                st.subheader("Extracted Faces")

                total_faces = len(frames)

                preview_faces = frames[:15]
                cols = st.columns(5)

                for i, face in enumerate(preview_faces):
                    col = cols[i % 5]
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    col.image(face_rgb, caption=f"Face {i+1}", use_container_width=True)

                if total_faces > 15:
                    st.caption(f"Showing 15 of {total_faces} faces")

                    with st.expander(f"View remaining {total_faces - 15} faces"):
                        cols_all = st.columns(5)

                        for i, face in enumerate(frames[15:], start=16):
                            col = cols_all[(i - 16) % 5]
                            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            col.image(face_rgb, caption=f"Face {i}", use_container_width=True)

                # ======================
                # PREDICTION
                # ======================
                fake_prob = predict_video(frames)

                st.subheader(f"Fake Probability: {fake_prob:.2f}%")

                # ======================
                # GAUGE BACK 🔥
                # ======================
                if fake_prob > 70:
                    status = "Highly likely FAKE"
                    color = "red"
                elif fake_prob > 40:
                    status = "Suspicious (uncertain)"
                    color = "orange"
                else:
                    status = "Likely REAL"
                    color = "green"

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

                    gradient_steps.append({
                        'range': [i, i + 5],
                        'color': f'#{r:02X}{g:02X}{b:02X}'
                    })

                font_family = "Arial Black"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fake_prob,

                    number={
                        'font': {
                            'size': 48,
                            'color': color,
                            'family': font_family
                        },
                        'valueformat': '.2f',
                        'suffix': '%'
                    },

                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': gradient_steps,
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'value': fake_prob
                        }
                    }
                ))

                fig.update_layout(
                    height=420,
                    font=dict(family=font_family),
                    annotations=[dict(
                        x=0.5,
                        y=0.3,
                        text=f"<b>{status}</b>",
                        showarrow=False,
                        font=dict(size=26, color=color)
                    )]
                )

                st.plotly_chart(fig, use_container_width=True)

        except:
            st.error("Error during processing.")

        finally:
            st.session_state.is_detecting = False

# ======================
# FOOTER
# ======================
st.markdown("""
<hr style="margin-top: 50px; margin-bottom: 10px;">
<div style='text-align: center; font-size: 14px; color: gray;'>
    © 2025 Deepfake Video Detection Web App | Developed for University Final Year Project 22004860
</div>
""", unsafe_allow_html=True)
