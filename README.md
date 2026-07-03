# 🎬 Celebrity Face Matcher

Upload a photo and discover which Bollywood celebrity you resemble. The app detects
the face in your image, extracts a deep facial embedding with **VGGFace (ResNet-50)**,
and finds the closest match among **8,600+** celebrity images using cosine similarity.

> ### 🔗 Live Demo
> **👉 [Try it here](https://Hamzurna-celebrity-face-match.hf.space)**
>
> Deployed on Hugging Face Spaces.

---

## ✨ Features

- 📤 Upload any face photo (`jpg`, `jpeg`, `png`)
- 🧠 Face detection with **MTCNN**
- 🧬 Deep feature extraction with **VGGFace / ResNet-50**
- 📊 Closest celebrity match via **cosine similarity** + similarity score
- 🎨 Clean, modern Streamlit UI
- 🐳 Fully containerized with Docker

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| UI | Streamlit |
| Face detection | MTCNN |
| Feature extraction | keras-vggface (VGGFace, ResNet-50) |
| Deep learning | TensorFlow 2.3.1 / Keras 2.4.3 |
| Similarity | scikit-learn (cosine similarity) |
| Image processing | OpenCV, Pillow |
| Containerization | Docker |

---

## 🚀 Run Locally

> **Note:** The precomputed model artifacts `embedding.pkl` and `filenames.pkl` are
> **not** included in this repository (large / generated files). Generate them once
> from the included `data/` folder using `feature_extract.py`, then run the app.

```bash
# 1. Install dependencies (Python 3.7 recommended)
pip install -r requirements.txt

# 2. Generate embeddings (creates filenames.pkl + embedding.pkl)
python feature_extract.py

# 3. Launch the app
streamlit run app.py
```

Then open <http://localhost:8501>.

---

## 🐳 Run with Docker

```bash
# Build the image
docker build -t final-project:v1 .

# Run the container
docker run -p 8000:8000 final-project:v1
```

Then open <http://localhost:8000>.

See **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** for a full walkthrough and troubleshooting.

---

## 📁 Project Structure

```
celebrity_face_match/
├── app.py               # Streamlit application
├── feature_extract.py   # Builds embedding.pkl from the data/ images
├── test.py              # Standalone matching test script
├── data/                # Celebrity image dataset (100 actors)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── .dockerignore
├── DOCKER_GUIDE.md      # Docker reference / instructions
└── README.md
```

---

## ⚙️ How It Works

1. **Detect** – MTCNN locates the face and returns a bounding box.
2. **Embed** – the cropped face (224×224) is passed through VGGFace (ResNet-50,
   `include_top=False`, average pooling) to produce a feature vector.
3. **Match** – cosine similarity is computed against every celebrity embedding; the
   highest-scoring image is returned as your look-alike, along with the score.

---

## 👤 Author

**Hamza Akbar** — [@Hamza71110](https://github.com/Hamza71110)

🚀 Powered by TensorFlow • VGGFace • Streamlit
