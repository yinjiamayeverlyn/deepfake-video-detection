import streamlit as st
import datetime
import tempfile

user_email = st.experimental_user.email  # Only works on Streamlit Cloud

if not user_email.endswith("@siswa.um.edu.my"):
    st.error("Access denied. You must use a @siswa.um.edu.my email.")
    st.stop()

st.success(f"Welcome, {user_email}!")
# Continue your app logic here

# Set up the page
st.set_page_config(page_title="Deepfake Detection Web App", layout="wide")

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
                    Users simply upload a video file to the system, which is then processed by a deepfake detection model combining Convolutional Neural Networks (CNN) 
                    and Long Short-Term Memory (LSTM) networks. These models work together to analyze both spatial (image-level) and temporal (frame-sequence) 
                    inconsistencies that are common in deepfake content.
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
    st.image("images/upload_banner.png", use_container_width=True)  # full width banner

    st.markdown("<h2 style='text-align:center;'>Upload Video for Deepfake Detection</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file:
        # Save video temporarily for preview
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.video(tfile.name)
        st.markdown(f"**File name:** `{uploaded_file.name}`")
        st.markdown(f"**Upload time:** `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

        # Submit Button
        if st.button("Submit for Detection"):
            st.success("Video submitted! Detection complete.")

            # Simulated Output Section
            st.header("Detection Result")
            st.markdown("**Prediction:** Fake Video Detected")
            st.markdown("**Confidence Score:** 88.7%")

            # Analysis Output
            with st.expander("Analysis Details"):
                st.markdown("""
                **Model Used:** CNN + LSTM Hybrid  
                - **CNN** detected visual artifacts around the mouth and eyes.  
                - **LSTM** detected temporal inconsistencies (e.g., blinking, motion).  
                - Video closely matches known deepfake manipulation patterns.
                """)

            # Download Report Button
            report = f"""
                Deepfake Detection Report

                Result: Fake Video Detected
                Confidence Score: 88.7%
                Model Used: CNN + LSTM Hybrid

                Key Observations:
                - Visual inconsistencies in facial regions (CNN)
                - Unnatural motion sequences (LSTM)
                - Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
            st.download_button("Download Detection Report", report, file_name="detection_report.txt")

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
        This web app helps users detect deepfake videos. It's intended for use by students and staff of Universiti Malaya.  
        Simply follow the steps below to analyze a video.
        """)

    st.markdown("### Step 1: Upload a Video")
    st.write("""
        - Go to the **Upload Video** tab.
        - Click the **'Browse files'** button.
        - Select a video in `.mp4`, `.mov`, or `.avi` format.
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
        It enables users to upload videos and detect manipulations using machine learning models.
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
        - **Machine Learning:** CNN + LSTM Hybrid Deepfake Model  
        - **Cloud Services:** Firebase, Cloudinary  
        - **Deployment:** Localhost or Streamlit Cloud
        """)

    st.markdown("### Acknowledgements")
    st.write("""
        I would like to thank to my supervisor **Dr. Rasha Ragheb Attaallah** for her guidance and support throughout the development of this project. 
        Special thanks to **Multimedia Department** for providing resources and feedback.
        """)

    st.markdown("### Contact Us")
    st.write("""
        For inquiries or collaboration, please contact:

        - **Email:** <a href="mailto:everlynyinjiamay@gmail.com">everlynyinjiamay@gmail.com</a>  
        - **GitHub:** [https://github.com/your-repo](https://github.com/your-repo)
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style='text-align: center; font-size: 16px; color: gray;'>
        © 2025 Deepfake Video Detection Web App | Developed for University Final Year Project 22004860
    </div>
    """, unsafe_allow_html=True)
