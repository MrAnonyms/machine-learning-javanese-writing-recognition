import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from sklearn.metrics import classification_report, confusion_matrix

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Handwriting Recognition with Aksara Jawa",
    layout="centered",  # Anda bisa coba "wide" juga
    initial_sidebar_state="expanded",
)

# Definisi Palette Warna
COLOR_PALETTE = {
    "periwinkle": "#C6D2F0",  # Warna muda, bisa untuk highlight atau secondary background
    "french_gray": "#BEC5D8",  # Abu-abu kebiruan muda, untuk elemen sekunder
    "white": "#FFFFFF",  # Putih murni, untuk teks atau latar belakang elemen tertentu
    "vista_blue": "#7392DA",  # Biru dominan, cocok untuk judul, tombol, atau highlight
    "french_gray_2": "#AAACB0",  # Abu-abu kebiruan sedikit lebih gelap, untuk border atau teks sekunder
}

# --- CSS Styling Kustom ---
st.markdown(
    f"""
<style>
.stApp {{
    color: {COLOR_PALETTE["french_gray_2"]}; /* Warna teks default */
    font-family: 'Segoe UI', sans-serif;
}}

h1 {{
    text-align: center;
    color: {COLOR_PALETTE["vista_blue"]}; /* Warna judul dengan Vista Blue */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin-bottom: 25px;
}}

h2, h3, h4, h5, h6 {{
    color: {COLOR_PALETTE["vista_blue"]}; /* Subheader dan lainnya */
}}

.stButton>button {{
    background-color: {COLOR_PALETTE["vista_blue"]}; /* Warna tombol dengan Vista Blue */
    color: {COLOR_PALETTE["white"]}; /* Warna teks tombol dengan White */
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    font-weight: bold;
    transition: background-color 0.3s ease;
}}

.stButton>button:hover {{
    background-color: {COLOR_PALETTE["french_gray_2"]}; /* Warna tombol saat hover dengan French Gray 2 */
    color: {COLOR_PALETTE["white"]};
}}

.stTextInput>div>div>input {{
    border: 2px solid {COLOR_PALETTE["french_gray"]}; /* Border input teks dengan French Gray */
    border-radius: 8px;
    padding: 10px;
    background-color: {COLOR_PALETTE["white"]}; /* Latar belakang input dengan White */
    color: {COLOR_PALETTE["french_gray_2"]}; /* Warna teks input dengan French Gray 2 */
}}

.stTextInput>div>div>input:focus {{
    border-color: {COLOR_PALETTE["vista_blue"]}; /* Border input saat fokus dengan Vista Blue */
    outline: none;
    box-shadow: 0 0 0 0.1rem {COLOR_PALETTE["vista_blue"]}50; /* Bayangan ringan saat fokus */
}}

.stSelectbox>div>div {{
    border: 2px solid {COLOR_PALETTE["french_gray"]};
    border-radius: 8px;
    padding: 5px;
    background-color: {COLOR_PALETTE["white"]};
    color: {COLOR_PALETTE["french_gray_2"]};
}}

.stSelectbox>div>div:hover {{
    border-color: {COLOR_PALETTE["vista_blue"]};
}}

.sidebar .sidebar-content {{
    background-color: {COLOR_PALETTE["french_gray"]}; /* Latar belakang sidebar dengan French Gray */
    color: {COLOR_PALETTE["french_gray_2"]}; /* Tetap bisa diatur jika ingin spesifik */
    padding-top: 20px;
}}

.sidebar h2 {{
    color: {COLOR_PALETTE["vista_blue"]}; /* Judul sidebar dengan Vista Blue */
}}

.stRadio div[role="radiogroup"] label {{
    background-color: {COLOR_PALETTE["vista_blue"]}; /* Latar belakang radio button dengan White */
    border: 1px solid {COLOR_PALETTE["vista_blue"]};
    border-radius: 5px;
    padding: 5px 10px;
    margin: 5px 0;
    cursor: pointer;
    transition: background-color 0.2s ease;
    color: {COLOR_PALETTE["french_gray_2"]};
}}

.stRadio div[role="radiogroup"] label:hover {{
    background-color: {COLOR_PALETTE["vista_blue"]}; /* Hover radio button dengan Periwinkle */
}}

.stRadio div[role="radiogroup"] label[data-baseweb="radio"] input[type="radio"]:checked + div {{
    background-color: {COLOR_PALETTE["vista_blue"]}; /* Radio button terpilih dengan Vista Blue */
    color: {COLOR_PALETTE["white"]}; /* Teks radio button terpilih dengan White */
}}

/* Gaya untuk kontainer kustom */
.custom-card {{
    background-color: {COLOR_PALETTE["french_gray"]}; /* Latar belakang card dengan French Gray */
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.01stre);
    margin-top: 20px;
    color: {COLOR_PALETTE["french_gray_2"]}; /* Teks card dengan French Gray 2 */
}}

.custom-card h3 {{
    color: {COLOR_PALETTE["vista_blue"]}; /* Judul card dengan Vista Blue */
    margin-bottom: 15px;
}}

</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
<div style="text-align: center; margin-top: 50px; color: {COLOR_PALETTE["vista_blue"]}; font-size: 0.9em;">
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    "<h1 style='text-align: center; color: #7291DA;'>Handwriting Recognition with Aksara Jawa</h1>",
    unsafe_allow_html=True,
)

st.sidebar.title("Lets Explore!")
page = st.sidebar.radio(
    "What do you want to explore?", ["Page", "Classification", "Recognition", "How it works"]
)

def ekstrak_fitur_hog(img, ukuran=(64, 64), pixels_per_cell=(8, 8)):
    if len(img.shape) == 3:
        img = rgb2gray(img)
    img_resized = resize(img, ukuran)
    
    fitur, _ = hog(
        img_resized,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        orientations=9,
        block_norm="L2-Hys",
        visualize=True,
    )
    return fitur


if page == "Page":
    st.markdown(
        """ ### Welcome to the Javanese Handwriting Recognition App!  
This application uses a Machine Learning model to recognize **Javanese script** from handwritten input.

**Features:**
- Draw characters directly on the canvas
- Upload images of handwritten characters
- Get instant predictions

Start by choosing a menu option from the sidebar 👈  
Enjoy exploring and preserving this cultural heritage through technology!
"""
    )

elif page == "Classification":
    st.subheader("Draw an Aksara Jawa Character")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=True,
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas_top",
    )

    if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
        st.divider()

        image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
        gray = image.convert("L")
        thresholded = gray.point(lambda x: 0 if x < 128 else 255, 'L')
        
        tab1, tab2 = st.tabs(["Low-Resolution Model (8x8)", "High-Resolution Model (4x4)"])

        with tab1:
            col1_low, col2_low = st.columns(2)

            with col1_low:
                st.write("Sample Character:")
                try:
                    background_img = Image.open(r"assets/sample_ra.png")
                    st.image(background_img, width=160)
                except FileNotFoundError:
                    st.warning("Sample image not found.")
            
            with col2_low:
                st.write("Preprocessed Image:")
                size_low = (42, 42)
                resized_low = thresholded.resize(size_low)
                normalized_low = np.array(resized_low) / 255.0
                st.image(
                    normalized_low, 
                    caption=f"Low-Res Preview ({size_low[0]}x{size_low[1]})",
                    use_container_width='auto'
                )

            st.info("Using Model 1: HOG `pixels_per_cell=(8,8)`. Accuracy: 87%")
            
            hog_features = ekstrak_fitur_hog(normalized_low, pixels_per_cell=(8, 8)).reshape(1, -1)
            
            if st.button("Predict with Low-Res Model", key="predict_low_res"):
                model_8x8 = joblib.load("models/svm_model.joblib")
                scaler_8x8 = joblib.load("models/svm_scaler.joblib")
                scaled_input = scaler_8x8.transform(hog_features)
                prediction = model_8x8.predict(scaled_input)[0]
                st.success(f"Low-Res Model Predicted: **{prediction}**")

        with tab2:
            col1_high, col2_high = st.columns(2)

            with col1_high:
                st.write("Sample Character:")
                try:
                    background_img = Image.open(r"assets/sample_ra.png")
                    st.image(background_img, width=160)
                except FileNotFoundError:
                    st.warning("Sample image not found.")

            with col2_high:
                st.write("Preprocessed Image:")
                size_high = (90, 90)
                resized_high = thresholded.resize(size_high)
                normalized_high = np.array(resized_high) / 255.0
                st.image(
                    normalized_high, 
                    caption=f"High-Res Preview ({size_high[0]}x{size_high[1]})",
                    use_container_width='auto'
                )

            st.info("Using Model 2: HOG `pixels_per_cell=(4,4)`. Accuracy: 83%")

            hog_features = ekstrak_fitur_hog(normalized_high, pixels_per_cell=(4, 4)).reshape(1, -1)

            if st.button("Predict with High-Res Model", key="predict_high_res"):
                model_4x4 = joblib.load("models/svm_model_highres.joblib")
                scaler_4x4 = joblib.load("models/svm_scaler_highres.joblib")
                scaled_input = scaler_4x4.transform(hog_features)
                prediction = model_4x4.predict(scaled_input)[0]
                st.success(f"High-Res Model Predicted: **{prediction}**")
    else:
        st.info("Draw a character in the canvas above to get started.")

elif page == "Recognition":
    st.subheader("Upload an image of handwritten Aksara Jawa")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        thresholded = image.point(lambda x: 0 if x < 128 else 255)
        size = (42, 42)
        resized = thresholded.resize(size)
        normalized = np.array(resized) / 255.0
        
        tab1, tab2 = st.tabs(["Low-Resolution Model (8x8)", "High-Resolution Model (4x4)"])
        
        with tab1:
            st.info("Using Model 1: HOG with `pixels_per_cell=(8,8)`. Final Test Accuracy: 87% [cite: 658]")
            
            hog_features = ekstrak_fitur_hog(normalized, pixels_per_cell=(8, 8)).reshape(1, -1)
            model_8x8 = joblib.load("models/svm_model_8x8.joblib")
            scaler_8x8 = joblib.load("models/svm_scaler_8x8.joblib")
            scaled_input = scaler_8x8.transform(hog_features)
            prediction = model_8x8.predict(scaled_input)[0]
            st.success(f"Low-Res Model Predicted: **{prediction}**")

        with tab2:
            st.info("Using Model 2: HOG with `pixels_per_cell=(4,4)`. Final Test Accuracy: 83% [cite: 658]")

            hog_features = ekstrak_fitur_hog(normalized, pixels_per_cell=(4, 4)).reshape(1, -1)
            model_4x4 = joblib.load("models/svm_model_4x4.joblib")
            scaler_4x4 = joblib.load("models/svm_scaler_4x4.joblib")
            scaled_input = scaler_4x4.transform(hog_features)
            prediction = model_4x4.predict(scaled_input)[0]
            st.success(f"High-Res Model Predicted: **{prediction}**")

    else:
        st.info("Please upload an image to begin recognition.")



elif page == "How it works":
    st.markdown(
        """
        This application uses a Machine Learning model to recognize handwritten Javanese script. 
        Here’s a step-by-step breakdown of how we built and evaluated the system.
        """
    )

    st.markdown("### 1. The Dataset and Exploratory Data Analysis")
    st.markdown(
        """
        We used a dataset of Javanese script images sourced from the Universe platform, a common site for computer vision datasets. This dataset was chosen because it provided clearly labeled images, which is essential for training a supervised learning model. The data was already conveniently divided into 'train', 'validation', and 'test' sets.

        Our Exploratory Data Analysis involved:
        - **Checking Class Distribution:** We checked how many images we have for each of the 20 basic *aksara* (from 'ha' to 'nga'). An imbalanced dataset, where some characters have many more samples than others, can bias the model.    
        - **Visualizing the Characters:** We manually inspected samples from each class to check for quality, noise, and variations in handwriting style.
        """
    )

    st.markdown("### 2. Methodology and Model Training")
    st.markdown(
        """
        Our approach can be broken down into four main stages:
        
        **a. Image Preprocessing:**
        Every input image (whether drawn or uploaded) is standardized to ensure consistency:
        1.  **Grayscale Conversion:** The image is converted from color to black and white.
        2.  **Thresholding:** We create a pure black-and-white image to remove any gray shades, making the character's shape clearer.
        3.  **Resizing:** All images are resized to a fixed dimension (42x42 pixels) to be fed into the model.

        **b. Feature Extraction with HOG:**
        Computers don't "see" images like we do. We need to convert images into a numerical format. We used the **Histogram of Oriented Gradients (HOG)** technique. HOG is very effective at capturing the shape and structure of an object (in our case, a character) by analyzing its edge directions.

        **c. Model Selection:**
        We chose a **Support Vector Machine (SVM)**, a powerful and reliable classification algorithm. An SVM's job is to find the optimal boundary that separates the different classes (characters) from each other based on their HOG features.

        **d. Training:**
        The SVM model was trained on thousands of pre-processed images and their corresponding HOG features until it learned the patterns associated with each Javanese character.
        """
    )

    st.markdown("### 3. Performance Metrics")
    st.markdown(
        """
        After training, we evaluated the model's performance on a separate "test set" of images it had never seen before.
        """
    )


    tab1, tab2 = st.tabs(["Model 1: Low-Res (8x8) - WINNER", "Model 2: High-Res (4x4)"])
    with tab1:
        st.markdown("#### Final Performance on the Test Set (HOG 8x8)")
    
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Accuracy")
            st.info("Accuracy: **87%**") 
            st.markdown("Accuracy measures the percentage of characters our model predicted correctly on the test data.")

        with col2:
            st.markdown("##### Confusion Matrix")
            st.image("assets/low_res_test_conf.jpg", caption="Confusion Matrix for Model 1 (8x8)")

        with col3:
            st.markdown("##### Classification Report")
            st.image("assets/low_res_test_class.jpg", caption="Classification Report for Model 1 (8x8)")

    with tab2:
        st.markdown("#### Final Performance on the Test Set (HOG 4x4)")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Accuracy")
            st.info("Accuracy: **83%**") 
            st.markdown("Accuracy measures the percentage of characters our model predicted correctly on the test data.")

        with col2:
            st.markdown("##### Confusion Matrix")
            st.image("assets/high_res_test_conf.jpg", caption="Confusion Matrix for Model 2 (4x4)")
        
        with col3:
            st.markdown("##### Classification Report")
            st.image("assets/high_res_test_class.jpg", caption="Classification Report for Model 2 (4x4)")

