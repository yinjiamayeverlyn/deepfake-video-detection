import streamlit as st
import os
import cv2
import torch
import tempfile
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms, models
import torch.nn as nn
import plotly.graph_objects as go
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from streamlit_js_eval import streamlit_js_eval
import urllib.request
import datetime
from zoneinfo import ZoneInfo
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

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
# DEVICE
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# DETECT SCREEN WIDTH
# ======================
width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH')
is_mobile = width < 768 if width else False

# ======================
# SESSION STATE
# ======================
if "show_all_faces" not in st.session_state:
    st.session_state.show_all_faces = False
if "is_detecting" not in st.session_state:
    st.session_state.is_detecting = False

# ======================
# LOAD EMBEDDER
# ======================
@st.cache_resource
def load_embedder():
    return InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
embedder = load_embedder()

# ======================
# LOAD FACE DETECTOR
# ======================
@st.cache_resource
def load_mtcnn():
    return MTCNN(keep_all=True, device=DEVICE)
mtcnn = load_mtcnn()

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
# CROP FACES
# ======================
def crop_faces(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)
    faces = []
    if boxes is None:
        return faces
    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        size = max(w, h) * 1.3
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_x1 = int(max(0, cx - size / 2))
        new_y1 = int(max(0, cy - size / 2))
        new_x2 = int(min(frame.shape[1], cx + size / 2))
        new_y2 = int(min(frame.shape[0], cy + size / 2))
        face = frame[new_y1:new_y2, new_x1:new_x2]
        if face.size != 0:
            faces.append(face)
    return faces

# ======================
# PREDICT FAKE PROBABILITY
# ======================
def predict_video(frames):
    if len(frames) == 0:
        return 0
    frames_tensor = torch.stack([transform(f) for f in frames]).to(DEVICE)
    with torch.no_grad():
        outputs = model(frames_tensor).view(-1)
        outputs_max = outputs.max()
        outputs_mean = outputs.mean()
        final_score = 0.7 * outputs_max + 0.3 * outputs_mean
        fake_prob = torch.sigmoid(final_score).item() * 100
    return fake_prob

# ======================
# CLUSTER FACES
# ======================
def cluster_faces(faces, eps=0.6, min_samples=2):
    if len(faces) == 0:
        return {}
    embeddings = []
    for f in faces:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = embedder(img_tensor).cpu().numpy()
        embeddings.append(emb[0])
    embeddings = normalize(np.array(embeddings))
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    labels = clustering.labels_
    clustered_faces = {}
    for face, label in zip(faces, labels):
        if label == -1:
            label = max(labels) + 1
        clustered_faces.setdefault(label, []).append(face)
    return clustered_faces

# ======================
# UI HEADER
# ======================
st.image("images/home_mobile.png" if is_mobile else "images/home_banner.png", use_container_width=True)
st.markdown("<h2 style='text-align:center;'>Upload Video for Deepfake Detection</h2>", unsafe_allow_html=True)

upload_option = st.radio(
    "Select video input method:",
    ["Upload from device", "Provide video URL"]
)

video_path = None
video_filename = None
valid_video = False

# ======================
# UPLOAD VIDEO
# ======================
if upload_option == "Upload from device":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        video_filename = uploaded_file.name
        valid_video = True

elif upload_option == "Provide video URL":
    video_url = st.text_input("Paste video URL (Google Drive / Dropbox)")
    if video_url:
        try:
            if "dropbox.com" in video_url:
                video_url = video_url.replace("dl=0", "dl=1").replace("www.dropbox.com", "dl.dropboxusercontent.com")
            elif "drive.google.com" in video_url:
                if "/d/" in video_url:
                    file_id = video_url.split("/d/")[1].split("/")[0]
                elif "id=" in video_url:
                    file_id = video_url.split("id=")[1].split("&")[0]
                else:
                    raise ValueError()
                video_url = f"https://drive.google.com/uc?export=download&id={file_id}"
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

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps else 0
    cap.release()

    if st.button("Submit for Detection"):
        if st.session_state.is_detecting:
            st.error("Processing already in progress. Please wait.")
            st.session_state.is_detecting = False
            st.stop()
        st.session_state.is_detecting = True

        try:
            if duration < 4:
                st.error("Video too short (<4 seconds). Please upload a longer video.")
                st.session_state.is_detecting = False
                st.stop()

            cap = cv2.VideoCapture(video_path)
            frames = []
            interval = max(1, int(fps // 3)) if duration < 10 else int(fps)
            count = 0
            prev_gray = None
            DIFF_THRESHOLD = 15

            with st.spinner("Processing video..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % interval == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if prev_gray is not None:
                            diff = np.mean(cv2.absdiff(gray, prev_gray))
                            if diff < DIFF_THRESHOLD:
                                count += 1
                                continue
                        prev_gray = gray
                        faces_in_frame = crop_faces(frame)
                        frames.extend(faces_in_frame)
                    count += 1
            cap.release()

            # ======================
            # CLUSTER FACES
            # ======================
            clustered_faces = cluster_faces(frames)
            num_people = len(clustered_faces)

            if num_people == 0:
                st.error("No face detected.")
            else:
                st.success(f"{len(frames)} faces detected, clustered into {num_people} individual(s)")

                for person_id, person_faces in clustered_faces.items():
                    preview_limit = 3 if is_mobile else 15
                    total_faces_person = len(person_faces)
                    preview_faces_person = person_faces[:preview_limit]
                    num_cols = 3 if is_mobile else 5

                    st.subheader(f"Person {person_id + 1} - {total_faces_person} faces")

                    # --- preview faces ---
                    for i in range(0, len(preview_faces_person), num_cols):
                        row_faces = preview_faces_person[i:i+num_cols]
                        cols = st.columns(num_cols)
                        for j, face in enumerate(row_faces):
                            cols[j].image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
                                          caption=f"Face {i+j+1}", use_container_width=True)

                    # --- expander for remaining faces ---
                    if total_faces_person > preview_limit:
                        with st.expander(f"View remaining {total_faces_person - preview_limit} faces of Person {person_id + 1}"):
                            remaining_faces = person_faces[preview_limit:]
                            for i in range(0, len(remaining_faces), num_cols):
                                row_faces = remaining_faces[i:i+num_cols]
                                cols = st.columns(num_cols)
                                for j, face in enumerate(row_faces):
                                    cols[j].image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
                                                  caption=f"Face {preview_limit + i + j +1}", use_container_width=True)

                    # --- prediction & gauge ---
                    fake_prob_person = predict_video(person_faces)
                    st.subheader(f"Person {person_id + 1} Fake Probability: {fake_prob_person:.2f}%")
                    if fake_prob_person > 70:
                        status = "Highly likely FAKE"
                        color = "red"
                        st.error(status)
                    elif fake_prob_person > 40:
                        status = "Suspicious (uncertain)"
                        color = "orange"
                        st.warning(status)
                    else:
                        status = "Likely REAL"
                        color = "green"
                        st.success(status)

                    # --- gauge ---
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
                        gradient_steps.append({'range':[i,i+5],'color':f'#{r:02X}{g:02X}{b:02X}'})
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fake_prob_person,
                        number={'font': {'size': 38 if is_mobile else 46,'color': color,'family': "Arial Black"},
                                'valueformat': '.2f','suffix':'%'},
                        gauge={'axis':{'range':[0,100]},'bar':{'color':color},'steps':gradient_steps,
                               'threshold':{'line':{'color':'black','width':3},'value':fake_prob_person}}
                    ))
                    fig.update_layout(height=440 if is_mobile else 380,
                                      margin=dict(l=20 if is_mobile else 40,r=20 if is_mobile else 40,t=40,b=40),
                                      annotations=[dict(x=0.5,y=0.43 if is_mobile else 0.2,
                                                        text=f"<b>{status}</b>",showarrow=False,
                                                        font=dict(size=20 if is_mobile else 24,color=color))])
                    st.plotly_chart(fig, use_container_width=True)

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
