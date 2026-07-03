import streamlit as st
import requests
import numpy as np
import cv2
import os
from PIL import Image
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity

feature_list = pickle.load(open('embedding.pkl', 'rb'))

st.set_page_config(
    page_title="Celebrity Face Matcher",
    page_icon="🎬",
    layout="wide"
)

filenames = pickle.load(open('filenames.pkl', 'rb'))

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def save_uploaded_image(uploaded_image ):
    try :
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except :
        return False

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    # extracting image features

    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result



def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

    return index_pos, similarity[index_pos]



st.markdown("""
<style>

/* ===========================
   BACKGROUND
=========================== */

.stApp{
    background: linear-gradient(135deg,#0F172A,#111827,#1E293B);
    color: white;
}

/* ===========================
   PAGE LAYOUT
=========================== */

.block-container{
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ===========================
   HIDE STREAMLIT DEFAULT UI
=========================== */

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* ===========================
   HEADINGS
=========================== */

h1{
    font-size:58px !important;
    font-weight:800;
}

h3{
    color:white;
}

/* ===========================
   FILE UPLOADER
=========================== */

[data-testid="stFileUploader"]{
    border:2px dashed #8B5CF6;
    border-radius:18px;
    background:#1E293B;
    padding:25px;
}

[data-testid="stFileUploader"]:hover{
    border-color:#A855F7;
}

/* ===========================
   SUCCESS CARD
=========================== */

.result-card{
    background: linear-gradient(135deg,#14532D,#166534);
    border-left:6px solid #22C55E;
    padding:20px;
    border-radius:18px;
    font-size:24px;
    font-weight:bold;
    margin-top:20px;
    margin-bottom:20px;
    box-shadow:0 10px 25px rgba(0,0,0,.35);
}

/* ===========================
   SCORE CARD
=========================== */

.score-card{
    background:#1F2937;
    border:1px solid #8B5CF6;
    border-radius:18px;
    padding:22px;
    margin-bottom:35px;
    box-shadow:0 10px 25px rgba(0,0,0,.35);
}

/* ===========================
   IMAGE STYLE
=========================== */

[data-testid="stImage"] img{

    border-radius:20px;

    border:3px solid rgba(255,255,255,.08);

    box-shadow:0 15px 35px rgba(0,0,0,.45);

    transition:.3s;

}

[data-testid="stImage"] img:hover{

    transform:scale(1.03);

    box-shadow:0 20px 40px rgba(168,85,247,.45);

}

/* ===========================
   METRIC STYLE
=========================== */

[data-testid="metric-container"]{

    background:#1F2937;

    border-radius:18px;

    padding:15px;

    border:1px solid #8B5CF6;

}

/* ===========================
   BUTTONS
=========================== */

.stButton>button{

    background:#8B5CF6;

    color:white;

    border:none;

    border-radius:12px;

    height:50px;

    font-weight:bold;

    transition:.3s;

}

.stButton>button:hover{

    background:#A855F7;

    transform:translateY(-2px);

}

/* ===========================
   SPINNER
=========================== */

[data-testid="stSpinner"]{

    color:#A855F7;

}

/* ===========================
   HORIZONTAL LINE
=========================== */

hr{

    border:1px solid #374151;

}

/* ===========================
   SCROLLBAR
=========================== */

::-webkit-scrollbar{

    width:10px;

}

::-webkit-scrollbar-thumb{

    background:#8B5CF6;

    border-radius:10px;

}

::-webkit-scrollbar-track{

    background:#111827;

}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;font-size:55px;'>
🎬 Celebrity Face Matcher
</h1>

<p style='text-align:center;font-size:22px;color:#D1D5DB;'>
Upload your photo and discover which Bollywood celebrity you resemble.
</p>
""", unsafe_allow_html=True)

left, center, right = st.columns([1,2,1])

with center:
    uploaded_image = st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )

if uploaded_image is not None:

    if save_uploaded_image(uploaded_image):

        display_image = Image.open(uploaded_image).convert("RGB")

        with st.spinner(" Analyzing your face..."):
            features = extract_features(
                os.path.join('uploads', uploaded_image.name),
                model,
                detector
            )

            index_pos, score = recommend(feature_list, features)

        predicted_actor = " ".join(
            filenames[index_pos].split('\\')[1].split('_')
        )

        st.markdown(f"""
        <div class="result-card">
         You resemble <b>{predicted_actor}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="score-card">
            <h3> Similarity Score</h3>
            <h1 style="color:#A855F7;">
                {score * 100:.2f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(
                "<h3 style='text-align:center;'> Your Image</h3>",
                unsafe_allow_html=True
            )
            st.image(display_image, width=350)

        with col2:
            st.markdown(
                f"<h3 style='text-align:center;'> {predicted_actor}</h3>",
                unsafe_allow_html=True
            )
            st.image(filenames[index_pos].replace('\\', '/'), width=350)

st.markdown("---")

st.markdown("""
<div style="text-align:center;
            color:#9CA3AF;
            font-size:18px;
            line-height:1.8;">

 Powered by <b>TensorFlow</b> • <b>VGGFace</b> • <b>Streamlit</b>

<br>

Made with heart by <b>Qandeel Imran | </b> <b> M.Umar | </b> <b> M.Hashir | </b> <b> Raja Aizaz | </b> <b> M.Hamza Akbar</b>

</div>
""", unsafe_allow_html=True)
