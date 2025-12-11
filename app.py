import streamlit as st

# --- First and only set_page_config ---
st.set_page_config(
    page_title="Deepfake Video Detection",
    page_icon="🎬",
    layout="wide"
)

# ---- Password Protection ----
def login():
    st.title(" Deepfake Video Detection - Access Required")

    password = st.text_input("Enter Access Password:", type="password")

    if password == st.secrets["APP_PASSWORD"]:
        st.session_state["auth"] = True
        st.experimental_rerun()
    elif password:
        st.error("Incorrect password. Please try again.")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login()
    st.stop()

# --- Standard Library Imports ---
import os
import io
import shutil
import tempfile
import datetime
import urllib.request
from zoneinfo import ZoneInfo

# --- Image & Video Processing ---
import cv2
import numpy as np

# --- PDF & Reporting ---
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics

# --- Data Visualization ---
import plotly.graph_objects as go

# --- Deep Learning ---
import tensorflow as tf

# --- Implementing Device Layout ---
from streamlit_js_eval import streamlit_js_eval

try:
    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()
except Exception as e:
    st.error(
        "MTCNN failed to load. The app will reload automatically. "
        "Face detection will not work until this succeeds."
    )    
    # --- Reload the app safely ---
    st.experimental_rerun() 

# Get browser width safely
raw_width = streamlit_js_eval(js_expressions='window.innerWidth', want_output=True)

# Normalize output
if isinstance(raw_width, list) and len(raw_width) > 0:
    width = raw_width[0]
elif isinstance(raw_width, (int, float)):
    width = raw_width
else:
    width = None

# Use width
if width is not None:
    is_mobile = width < 768
else:
    st.write("Waiting for browser size...")

detector = MTCNN()

# --- Load TF Lite model once ---
@st.cache_resource(show_spinner=True)
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

tflite_path = "ensemble_model.tflite"
interpreter, input_details, output_details = load_tflite_model(tflite_path)

# --- Top Navigation Bar using Tabs ---
tabs = st.tabs(["Home", "Upload Video", "Tutorial", "About Us"])

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Home Page ---
with tabs[0]:
    
    st.image("images/home_banner.png", use_container_width=True)  # full width banner

    st.markdown("<h2 style='text-align:center;'>Welcome to the Deepfake Detection Web App</h2>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 1 - Image left, text right
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/img1.png", use_container_width=True)
    with col2:
        st.markdown(
        """
        <div style='display: flex; align-items: center; height: 100%; min-height: 300px;'>
            <div>
                <h3 style='margin-top: 0;'>About the Project</h3>
                <p>
                    In recent years, deepfake technology has rapidly advanced, making it increasingly difficult to distinguish between real 
                    and manipulated video content. This raises concerns about misinformation, identity fraud, and public trust. 
                    Our web-based Deepfake Detection system aims to address this issue by providing users with a fast and accessible 
                    tool to analyze videos for signs of manipulation using artificial intelligence.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2 - Text left, image right
    col3, col4 = st.columns([2, 1])
    with col3:
        st.markdown(
        """
        <div style='display: flex; align-items: center; height: 100%; min-height: 300px;'>
            <div>
                <h3 style='margin-top: 0;'>Purpose of the Application</h3>
                <p>
                    The purpose of this project is to create a user-friendly platform for detecting deepfake videos using advanced machine learning models. 
                    Designed for educational use within Universiti Malaya, this tool allows students, researchers, and faculty members to experiment 
                    with and understand how AI can detect fake media.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True)
    with col4:
        st.image("images/img2.png", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3 - Image left, text right
    col5, col6 = st.columns([1, 2])
    with col5:
        st.image("images/img3.png", use_container_width=True)
    with col6:
        st.markdown(
        """
        <div style='display: flex; align-items: center; height: 100%; min-height: 300px;'>
            <div>
                <h3 style='margin-top: 0;'>How It Works</h3>
                <p>
                    Users simply upload a video file to the system, which is then analyzed by an ensemble of Convolutional Neural Networks (CNNs). 
                    These CNN models work together to examine spatial inconsistencies in individual frames, such as texture artifacts, blending errors, 
                    and illumination mismatches. By combining the strengths of multiple CNN architectures, the system produces a more robust and accurate prediction, 
                    effectively detecting deepfake manipulations in the uploaded video.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

     # Section 4 - Text left, image right
    col7, col8 = st.columns([2, 1])
    with col7:
        st.markdown(
        """
        <div style='display: flex; align-items: center; height: 100%; min-height: 300px;'>
            <div>
                <h3 style='margin-top: 0;'>Key Features</h3>
                <ul>
                    <li>Simple, web-based interface—no technical skills required</li>
                    <li>Support for MP4, MOV, and AVI video formats</li>
                    <li>Real-time detection feedback with confidence scores</li>
                    <li>Downloadable detection reports for academic or reference use</li>
                    <li>Designed with a focus on privacy, accuracy, and accessibility</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True)
    with col8:
        st.image("images/img4.png", use_container_width=True)

    # Section 5 - Image left, text right
    col9, col10 = st.columns([1, 2])
    with col9:
        st.image("images/img5.png", use_container_width=True)
    with col10:
        st.markdown(
        """
        <div style='display: flex; align-items: center; height: 100%; min-height: 300px;'>
            <div>
                <h3 style='margin-top: 0;'>Why It Matters</h3>
                <p>
                    As synthetic media becomes more realistic and widespread, developing tools for automatic detection is crucial. 
                    This application supports digital literacy and helps build awareness of AI-generated misinformation. 
                    By using this platform, users can better understand how to identify and respond to deepfakes in today’s digital environment.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True)
    
    st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style='text-align: center; font-size: 16px; color: gray;'>
        © 2025 Deepfake Video Detection Web App | Developed for University Final Year Project 22004860
    </div>
    """, unsafe_allow_html=True)

# --- Upload Video Page ---
with tabs[1]:
    # --- Safe cleanup for previous temp folders ---
    def cleanup_temp_dirs():
        """Safely remove previously created temporary directories."""
        if "temp_dirs" in st.session_state:
            for d in list(st.session_state.temp_dirs):  # use list() copy for safety
                if d and os.path.exists(d):
                    try:
                        shutil.rmtree(d)
                    except Exception as e:
                        st.warning(f"Skipped removing {d}: {e}")
            st.session_state.temp_dirs.clear()

    # Initialize session variable
    if "temp_dirs" not in st.session_state:
        st.session_state.temp_dirs = []

    # Run cleanup on refresh (safe to call even if empty)
    cleanup_temp_dirs()

    st.image("images/upload_banner.png", use_container_width=True)  # full width banner

    st.markdown("<h2 style='text-align:center;'>Upload Video for Deepfake Detection</h2>", unsafe_allow_html=True)

    upload_option = st.radio("Select video input method:", ["Upload from device", "Provide video URL"])

    video_path = None  # Will hold the final path or URL
    video_filename = None

    if upload_option == "Upload from device":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            video_filename = uploaded_file.name

    elif upload_option == "Provide video URL":
        video_url = st.text_input("Paste video URL here (Dropbox or Google Drive links supported)")
        if video_url:
            try:
                # --- Handle Dropbox links ---
                if "dropbox.com" in video_url:
                    if "dl=0" in video_url:
                        video_url = video_url.replace("dl=0", "dl=1")
                    video_url = video_url.replace("www.dropbox.com", "dl.dropboxusercontent.com")

                # --- Handle Google Drive links ---
                elif "drive.google.com" in video_url:
                    if "/d/" in video_url:  # format: /d/<file_id>/
                        file_id = video_url.split("/d/")[1].split("/")[0]
                        video_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    elif "id=" in video_url:  # format: ?id=<file_id>
                        file_id = video_url.split("id=")[1].split("&")[0]
                        video_url = f"https://drive.google.com/uc?export=download&id={file_id}"

                # --- Save video locally ---
                video_filename = os.path.basename(video_url.split("?")[0])
                video_path = os.path.join(tempfile.gettempdir(), video_filename)
                urllib.request.urlretrieve(video_url, video_path)

            except Exception as e:
                st.error(f"Unable to download video from the link. Please check the URL. ({e})")

    # Initialize Upload time
    upload_time = ''

    # If video is successfully uploaded
    # --- After user submits video ---
    if video_path:
        st.video(video_path)
        st.markdown(f"**Video source:** `{video_filename}`")
        if upload_time == '':
            malaysia_time = datetime.datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
            upload_time = malaysia_time.strftime('%Y-%m-%d %H:%M:%S')
            
        st.markdown(f"**Upload time:** `{upload_time}`")

        # Validate video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps else 0
        cap.release()

        # --- Initialize the detection flag ---
        if "is_detecting" not in st.session_state:
            st.session_state.is_detecting = False

        submit_clicked = st.button("Submit for Detection")

        if submit_clicked:
            if st.session_state.is_detecting:
                st.error("Please be patient, you only need to click the 'Submit for Detection' button **once**. Resubmit now if needed.")
                st.session_state.is_detecting = False
            else:
                # Set flag to indicate detection is in progress
                st.session_state.is_detecting = True
                try:
                    if duration < 4:
                        st.error("Video is too short (less than 4 seconds). Please upload a longer video.")
                    else:
                        st.success("Video submitted! Extracting faces...")
                        
                        # --- Create temp directory ---
                        faces_dir = tempfile.mkdtemp(prefix="faces_")
                        if "temp_dirs" not in st.session_state:
                            st.session_state.temp_dirs = []
                        st.session_state.temp_dirs.append(faces_dir)

                        # --- Face extraction loop ---
                        cap = cv2.VideoCapture(video_path)
                        frame_interval = int(fps * 1)
                        frames = []
                        count = 0

                        with st.spinner("Extracting faces, please wait..."):
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                if count % frame_interval == 0:
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    results = detector.detect_faces(rgb_frame)

                                    for i, face in enumerate(results):
                                        x, y, w, h = face['box']
                                        x, y = max(0, x), max(0, y)
                                        margin = 0.2
                                        x1 = max(0, int(x - w * margin))
                                        y1 = max(0, int(y - h * margin))
                                        x2 = min(frame.shape[1], int(x + w * (1 + margin)))
                                        y2 = min(frame.shape[0], int(y + h * (1 + margin)))

                                        face_crop = frame[y1:y2, x1:x2]
                                        if face_crop.size > 0:
                                            face_resized = cv2.resize(face_crop, (224, 224))
                                            face_path = os.path.join(faces_dir, f"face_{count}_{i}.jpg")
                                            cv2.imwrite(face_path, face_resized)
                                            frames.append(face_resized)                                
                                count += 1
                        cap.release()

                        if not frames:
                            st.error("No face detected in this video. Please upload another video.")
                            st.session_state.is_detecting = False 
                            st.stop()
                        else:
                            st.header("Detected Faces (Sampled Frames)")
                            total_faces = len(frames)
                            st.success(f"{total_faces} faces extracted successfully!")
        
                            # --- Default: show first 12 in grid ---
                            cols = st.columns(4)
                            preview_faces = frames[:12]
                            for i, face in enumerate(preview_faces):
                                col = cols[i % 4]
                                col.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption=f"Face {i+1}", use_container_width=True)

                            # --- If more than 12, show “Show All” toggle ---
                            if total_faces > 12:
                                with st.expander(f"Show remaining {total_faces - 12} faces"):
                                    cols_all = st.columns(4)
                                    for i, face in enumerate(frames[12:], start=13):  # start numbering from 13
                                        col = cols_all[(i - 13) % 4]  # reset column cycle
                                        col.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
                                                caption=f"Face {i}", use_container_width=True)

                            # --- Predict using CNN TF Lite model ---
                            st.subheader("Running Deepfake Detection...")

                            # Function to run predictions on faces
                            def predict_faces_tflite(frames, threshold=0.45):
                                faces_array = np.array(frames, dtype=np.float32) / 255.0
                                preds = []

                                # TF Lite requires input shape (batch, H, W, C)
                                for face in faces_array:
                                    face_batch = np.expand_dims(face, axis=0).astype(np.float32)
                                    interpreter.set_tensor(input_details[0]['index'], face_batch)
                                    interpreter.invoke()
                                    output_data = interpreter.get_tensor(output_details[0]['index'])
                                    preds.append(output_data)

                                preds = np.vstack(preds)

                                # Handle ensemble output
                                if preds.ndim == 2 and preds.shape[1] == 2:
                                    probs = preds[:, 1]  # class 1 = fake
                                else:
                                    probs = preds.reshape(-1)

                                avg_score = float(np.mean(probs))
                                is_fake = avg_score >= threshold
                                confidence = round(abs(avg_score - 0.5) * 200, 2)

                                # Adjust low-confidence results
                                if confidence < 50:
                                    label = "Real Video"
                                    confidence = 100 - confidence
                                else:
                                    label = "Fake Video Detected" if is_fake else "Real Video"

                                return label, confidence, avg_score, probs

                            # --- Call the function to get predictions ---
                            label, confidence, avg_score, probs = predict_faces_tflite(frames, threshold=0.45)

                            # --- Gauge bar color ---
                            color = "#FF0000" if label == "Fake Video Detected" else "#006400" 

                            # --- Create smooth gradient steps manually ---
                            gradient_steps = []
                            for i in range(0, 101, 5):  # every 5%
                                if i < 50:
                                    # green → yellow transition
                                    r = 182 + int((255 - 182) * (i / 50))
                                    g = 239 + int((233 - 239) * (i / 50))
                                    b = 162 + int((169 - 162) * (i / 50))
                                else:
                                    # yellow → coral transition
                                    r = 255
                                    g = 233 - int((233 - 182) * ((i - 50) / 50))
                                    b = 169 - int((169 - 166) * ((i - 50) / 50))
                                hex_color = f'#{r:02X}{g:02X}{b:02X}'
                                gradient_steps.append({'range': [i, i + 5], 'color': hex_color})

                            if is_mobile:
                                gauge_font = 24
                            else: 
                                gauge_font = 48

                            # --- Gauge chart ---
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=confidence,
                                title={'text': ""},
                                number={
                                    'font': {'size': gauge_font, 'color': '#333', 'family': 'Arial Black'},
                                    'valueformat': '.2f',
                                    'suffix': '%'
                                },
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "lightgray"},
                                    'bar': {'color': color, 'thickness': 0.25},
                                    'bgcolor': 'white',
                                    'steps': gradient_steps,
                                    'borderwidth': 1,
                                    'bordercolor': '#ddd',
                                    'threshold': {
                                        'line': {'color': "black", 'width': 3},
                                        'thickness': 1.0,
                                        'value': confidence
                                    }
                                }
                            ))

                            if is_mobile:
                                label_font = 22
                                y_field= 0.40
                            else: 
                                label_font = 36
                                y_field= 0.25


                            # --- Add centered label below the confidence number ---
                            fig.add_annotation(
                                text=f"<b>{label}</b>",
                                x=0.5, y=y_field,
                                xref="paper", yref="paper",
                                showarrow=False,
                                font=dict(size=label_font, color="#333"),
                            )

                            fig.update_layout(
                                margin=dict(t=40, b=20, l=20, r=20),
                                height=350,
                                paper_bgcolor='white',
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                config={
                                    "displayModeBar": False
                                }
                            )

                            # --- Details below gauge ---
                            st.markdown(f"### Confidence: **{confidence:.2f}%**")
                            st.markdown(f"### Prediction: **{label}**")

                            with st.expander("Analysis Details"):

                                # ----- REAL VIDEO ANALYSIS -----
                                if label == "Real Video":
                                    if confidence < 61:
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - The prediction indicates real characteristics are present.  
                                        - Several natural facial features and frame patterns align with real behavior.
                                        """)
                                    
                                    elif 61 <= confidence < 71:
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - The model detected consistent real-video features across the frames. 
                                        - Natural movements and facial textures support a real classification.
                                        """)

                                    elif 71 <= confidence < 81:
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - Strong indicators of real facial behavior were observed.  
                                        - The video shows coherent lighting, textures, and frame-to-frame stability.
                                        """)

                                    elif 81 <= confidence < 91:
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - Clear and consistent real-video characteristics detected.  
                                        - No deepfake-like artifacts or irregularities appeared throughout the frames.
                                        """)

                                    else:  # confidence 91–100
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - Highly consistent real-video patterns detected across the entire clip.  
                                        - The facial features, motion, and textures strongly align with natural video behavior.
                                        """)

                                # ----- FAKE VIDEO ANALYSIS -----
                                else:  # label == "Fake"
                                    if confidence < 61:
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - The analysis detected several patterns associated with manipulated content. 
                                        - Some regions show irregularities commonly found in synthetic or altered frames.
                                        """)

                                    elif 61 <= confidence < 71:
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - Features in the video align with deepfake-like characteristics.
                                        - Subtle inconsistencies in textures and motion contribute to the prediction.
                                        """)

                                    elif 71 <= confidence < 81:
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - Multiple indicators of manipulation were observed. 
                                        - Frame patterns suggest synthetic alterations or generative artifacts.
                                        """)

                                    elif 81 <= confidence < 91:
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - Clear signs of deepfake-related irregularities detected.  
                                        - Texture inconsistencies, blending issues, or motion mismatches support this classification.
                                        """)

                                    else:  # confidence 91–100
                                        st.markdown("""
                                        **Model Used:** Ensemble CNN  
                                        - Strong and consistent evidence of deepfake manipulation detected throughout the video.
                                        - The facial region exhibits distinct synthetic patterns and frame-level anomalies.
                                        """)


                            st.markdown(f"**Important Note:** This model is not 100% perfect. Deepfake methods keep improving, so results should be used as guidance—not absolute proof.")

                        # --- Register font for Unicode ---
                        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

                        # --- Create PDF buffer ---
                        pdf_buffer = io.BytesIO()
                        pdf = SimpleDocTemplate(pdf_buffer, pagesize=A4)
                        styles = getSampleStyleSheet()
                        story = []

                        # --- Report Header ---
                        story.append(Paragraph("<b>Deepfake Detection Report</b>", styles["Title"]))
                        story.append(Spacer(1, 12))

                        summary_text = f"""
                        <b>Source:</b> {video_filename}<br/>
                        <b>Result:</b> {label}<br/>
                        <b>Confidence Score:</b> {confidence:.2f}%<br/>
                        <b>Extracted Faces:</b> {total_faces}<br/>
                        <b>Date:</b> {upload_time}<br/>
                        """
                        story.append(Paragraph(summary_text, styles["Normal"]))
                        story.append(Spacer(1, 12))

                        # --- Analysis Details ---
                        story.append(Paragraph("<b>Analysis Details:</b>", styles["Heading2"]))
                        # ----- REAL VIDEO ANALYSIS -----
                        if label == "Real Video":
                            if confidence < 61:
                                analysis_text = """
                                Model Used: Ensemble CNN <br/>
                                - The prediction indicates real characteristics are present. <br/>
                                - Several natural facial features and frame patterns align with real behavior. <br/>
                                """                
                            
                            elif 61 <= confidence < 71:
                                analysis_text = """
                                Model Used: Ensemble CNN <br/>
                                - The model detected consistent real-video features across the frames. <br/> 
                                - Natural movements and facial textures support a real classification. <br/>
                                """

                            elif 71 <= confidence < 81:
                                analysis_text = """
                                Model Used: Ensemble CNN <br/> 
                                - Strong indicators of real facial behavior were observed. <br/> 
                                - The video shows coherent lighting, textures, and frame-to-frame stability. <br/> 
                                """

                            elif 81 <= confidence < 91:
                                analysis_text = """
                                Model Used: Ensemble CNN <br/> 
                                - Clear and consistent real-video characteristics detected. <br/> 
                                - No deepfake-like artifacts or irregularities appeared throughout the frames. <br/> 
                                """

                            else:  # confidence 91–100
                                analysis_text = """
                                Model Used: Ensemble CNN <br/> 
                                - Highly consistent real-video patterns detected across the entire clip. <br/> 
                                - The facial features, motion, and textures strongly align with natural video behavior. <br/> 
                                """

                        # ----- FAKE VIDEO ANALYSIS -----
                        else:  # label == "Fake"
                            if confidence < 61:
                                analysis_text = """
                                Model Used: Ensemble CNN <br/> 
                                - The analysis detected several patterns associated with manipulated content. <br/> 
                                - Some regions show irregularities commonly found in synthetic or altered frames. <br/> 
                                """

                            elif 61 <= confidence < 71:
                                analysis_text = """
                                Model Used: Ensemble CNN <br/>
                                - Features in the video align with deepfake-like characteristics. <br/>
                                - Subtle inconsistencies in textures and motion contribute to the prediction. <br/>
                                """

                            elif 71 <= confidence < 81:
                                analysis_text = """
                                Model Used: Ensemble CNN <br/>
                                - Multiple indicators of manipulation were observed. <br/>
                                - Frame patterns suggest synthetic alterations or generative artifacts. <br/>
                                """

                            elif 81 <= confidence < 91:
                                analysis_text = """
                                Model Used: Ensemble CNN <br/>
                                - Clear signs of deepfake-related irregularities detected. <br/>
                                - Texture inconsistencies, blending issues, or motion mismatches support this classification. <br/>
                                """

                            else:  # confidence 91–100
                                analysis_text = """
                                Model Used: Ensemble CNN <br/>
                                - Strong and consistent evidence of deepfake manipulation detected throughout the video. <br/>
                                - The facial region exhibits distinct synthetic patterns and frame-level anomalies. <br/>
                                """                    

                        story.append(Paragraph(analysis_text, styles["Normal"]))
                        story.append(Spacer(1, 24))

                        # --- Extracted Face Images Section ---
                        story.append(Paragraph("<b>Extracted Face Images</b>", styles["Heading2"]))
                        story.append(Spacer(1, 12))

                        # Collect all extracted images from your temporary directory
                        image_paths = sorted([
                            os.path.join(faces_dir, f)
                            for f in os.listdir(faces_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        ])

                        # Prepare image grid (5 per row)
                        max_width = 1.1 * inch
                        max_height = 1.1 * inch
                        rows = []
                        row = []

                        for i, img_path in enumerate(image_paths):
                            try:
                                img = Image(img_path, width=max_width, height=max_height)
                            except Exception:
                                continue
                            row.append(img)
                            if (i + 1) % 5 == 0:
                                rows.append(row)
                                row = []

                        if row:  # remaining faces
                            rows.append(row)

                        # Add table layout
                        if rows:
                            table = Table(rows, hAlign='CENTER')
                            table.setStyle(TableStyle([
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
                                ('BOX', (0, 0), (-1, -1), 0.25, colors.grey)
                            ]))
                            story.append(table)

                        # --- Build PDF ---
                        pdf.build(story)
                        pdf_buffer.seek(0)

                        # --- Streamlit Download Button ---
                        st.download_button(
                            label="Download Detection Report",
                            data=pdf_buffer,
                            file_name="detection_report.pdf",
                            mime="application/pdf"
                        )
                finally:
                    # Always reset the flag when done
                    st.session_state.is_detecting = False

    else:
        st.info("Please upload a video file to begin.")

    st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style='text-align: center; font-size: 16px; color: gray;'>
        © 2025 Deepfake Video Detection Web App | Developed for University Final Year Project 22004860
    </div>
    """, unsafe_allow_html=True)
    
# --- Tutorial Page ---
with tabs[2]:
    st.image("images/tutorial_banner.png", use_container_width=True)  # full width banner

    st.markdown("<h2 style='text-align:center;'>Tutorial</h2>", unsafe_allow_html=True)

    st.markdown("### Getting Started")
    st.write("""
        This web app allows users to detect deepfake videos. It is currently public for anyone to try, though primarily developed for educational purposes.
        
        Simply follow the steps below to analyze a video.
        """)

    st.markdown("### Step 1: Upload a Video")
    st.write("""
        - Go to the **Upload Video** tab.
        - Choose your video input method:
            - Upload from device – select a video file in .mp4, .mov, or .avi format.
            - Provide video URL – enter a shareable link from Google Drive or Dropbox only.
        - Wait for the preview to appear on the screen.
        """)

    st.markdown("### Step 2: Submit for Detection")
    st.write("""
        - Click **'Submit for Detection'** after uploading.
        - The system will analyze the video using a deep learning model.
        - You’ll receive a prediction (Real or Fake), a confidence score, and technical analysis.
        """)

    st.markdown("### Step 3: Download the Report")
    st.write("""
        - Click **'Download Detection Report'** to save the results.
        - The report includes the prediction, confidence score, and timestamp.
        """)

    st.markdown("### Tips for Better Detection")
    st.write("""
        - Use videos that clearly show faces.
        - Avoid blurry, dark, or highly edited clips.
        - Short videos (under 20 seconds) process faster.
        - This tool is for educational purposes only.
        """)

    st.markdown("### Frequently Asked Questions")

    with st.expander("## What formats are supported?"):
        st.write("MP4, MOV, AVI")

    with st.expander("## Is it 100% accurate?"):
        st.write("No model is perfect. The results provide strong indications, not absolute proof.")

    with st.expander("## Can I analyze multiple videos?"):
        st.write("Yes, upload one video at a time and repeat the process.")

    st.markdown("### Ready to try it out?")
    st.info("Go to the **Upload Video** tab above to get started!")

    st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style='text-align: center; font-size: 16px; color: gray;'>
        © 2025 Deepfake Video Detection Web App | Developed for University Final Year Project 22004860
    </div>
    """, unsafe_allow_html=True)

# --- About Us Page ---
with tabs[3]:
    st.image("images/about_banner.png", use_container_width=True)  # full width banner

    st.markdown("<h2 style='text-align:center;'>About Us</h2>", unsafe_allow_html=True)

    st.markdown("### Project Overview")
    st.write("""
        This Deepfake Detection Web App was developed as a final year project to address the growing threat of deepfake videos. 
        It enables users to upload videos and detect manipulations using **deep learning** models.
        """)

    st.markdown("### Our Mission")
    st.write("""
        To build an accessible, user-friendly platform that helps people detect and analyze deepfake videos using state-of-the-art AI technologies — 
        promoting media authenticity and digital trust.
        """)

    st.markdown("### Technology Stack")
    st.write("""
        - **Frontend:** Streamlit (Python Web Framework)  
        - **Backend:** Python  
        - **Deep Learning:** Ensemble of Convolutional Neural Networks (CNNs) for deepfake detection 
        - **Deployment:** Streamlit Cloud
        - **Computer Vision:** OpenCV, MTCNN for face detection
        - **Data Visualization:** Plotly
        """)

    st.markdown("### Acknowledgements")
    st.write("""
        I would like to thank to my supervisor **Dr. Rasha Ragheb Attaallah** for her guidance and support throughout the development of this project. 
        Special thanks to **Multimedia Department** for providing resources and feedback.
        """)

    st.markdown("### Contact Us")
    st.write("""
        For inquiries or collaboration, please contact:

        - **Email:** everlynyinjiamay@gmail.com
        - **GitHub:** [https://github.com/yinjiamayeverlyn/deepfake-video-detection](https://github.com/yinjiamayeverlyn/deepfake-video-detection)
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style='text-align: center; font-size: 16px; color: gray;'>
        © 2025 Deepfake Video Detection Web App | Developed for University Final Year Project 22004860
    </div>
    """, unsafe_allow_html=True)








