import base64
import hashlib
import inspect
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Optional usage-logging module (kept out of the repo). The app runs fine
# without it; logging simply stays off when it's absent.
try:
    import gsheet_logger
except Exception:
    gsheet_logger = None

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Celebrity Face Matcher",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
# Version-compat shims (deploy target is Streamlit 1.23.1; local dev may be
# newer). These keep the app working across versions.
# --------------------------------------------------------------------------- #
def _rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()


try:
    _has_container_w = "use_container_width" in inspect.signature(st.image).parameters
except Exception:
    _has_container_w = False
_IMG_KW = {"use_container_width": True} if _has_container_w else {"use_column_width": True}


def show_img(img):
    st.image(img, **_IMG_KW)


# --------------------------------------------------------------------------- #
# Tunable detection thresholds (adjust after real-world testing)
# --------------------------------------------------------------------------- #
BLUR_THRESHOLD = 30.0      # lower  -> image treated as "too blurry"
AMBIGUITY_RATIO = 0.85     # 2nd-largest / largest face; higher -> "ambiguous"

MSG_BLUR = "Image is too blurry or low quality — please upload a clearer photo."
MSG_NO_FACE = "No face detected — please upload a clear photo that contains a face."
MSG_MULTI = ("Multiple faces detected — please upload a photo where you are the "
             "main person.")


# --------------------------------------------------------------------------- #
# Cached heavy resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_detector():
    from mtcnn import MTCNN
    return MTCNN()


@st.cache_resource(show_spinner=False)
def load_vggface():
    from keras_vggface.vggface import VGGFace
    return VGGFace(model="resnet50", include_top=False,
                   input_shape=(224, 224, 3), pooling="avg")


@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load precomputed embeddings; paths resolved next to this file so the
    process cwd doesn't matter."""
    import pickle
    base = Path(__file__).parent
    with open(base / "embedding.pkl", "rb") as f:
        feats = pickle.load(f)
    with open(base / "filenames.pkl", "rb") as f:
        names = pickle.load(f)
    return np.asarray(feats, dtype="float32"), names


def get_secrets():
    """Return st.secrets ONLY if a secrets.toml exists (avoids the red
    'No secrets files found' error box). Otherwise None -> env vars are used."""
    candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    if any(p.exists() for p in candidates):
        try:
            return st.secrets
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Detection + matching
# --------------------------------------------------------------------------- #
def decode_bgr(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def is_blurry(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD


def analyze_faces(bgr):
    """Return {'face': crop} for the largest face, or {'error': message}."""
    if bgr is None:
        return {"error": MSG_NO_FACE}
    detector = load_detector()
    results = detector.detect_faces(bgr)

    boxes = []
    for r in results:
        x, y, w, h = r["box"]
        x, y = max(0, x), max(0, y)
        if w > 0 and h > 0:
            boxes.append((w * h, x, y, w, h))
    boxes.sort(reverse=True, key=lambda b: b[0])

    if not boxes:
        return {"error": MSG_BLUR if is_blurry(bgr) else MSG_NO_FACE}
    if len(boxes) >= 2 and boxes[1][0] / boxes[0][0] >= AMBIGUITY_RATIO:
        return {"error": MSG_MULTI}

    _, x, y, w, h = boxes[0]
    face = bgr[y:y + h, x:x + w]
    return {"face": face} if face.size else {"error": MSG_NO_FACE}


def embed_face(face_bgr):
    from keras_vggface.utils import preprocess_input
    image = Image.fromarray(face_bgr).resize((224, 224))
    arr = np.asarray(image).astype("float32")
    arr = preprocess_input(np.expand_dims(arr, axis=0))
    return load_vggface().predict(arr).flatten()


def actor_name(path):
    parts = path.replace("\\", "/").split("/")
    folder = parts[-2] if len(parts) >= 2 else parts[-1]
    return " ".join(folder.split("_"))


def top_matches(features, matrix, names, k=3):
    """Up to k DISTINCT celebrities: [{'idx','score','name','path'}, ...]."""
    sims = cosine_similarity(features.reshape(1, -1), matrix)[0]
    order = np.argsort(sims)[::-1]
    out, seen = [], set()
    for i in order:
        nm = actor_name(names[i])
        if nm in seen:
            continue
        seen.add(nm)
        out.append({"idx": int(i), "score": float(sims[i]), "name": nm,
                    "path": names[i].replace("\\", "/")})
        if len(out) == k:
            break
    return out


# =========================================================================== #
# STYLES
# =========================================================================== #
BASE_CSS = """
<style>
.stApp{ background:radial-gradient(1200px 600px at 50% -10%,#182238,#0b1120 60%);
  color:#E5E7EB; }
.block-container{ max-width:1040px; padding-top:2.2rem; padding-bottom:3rem; }

/* make all text bold */
.stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp a,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp input, .stApp button,
.stApp [data-baseweb="tab"]{ font-weight:700 !important; }
.stApp input::placeholder{ font-weight:700 !important; }

#MainMenu, footer, header{ visibility:hidden; }
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarNav"]{ display:none !important; }

h1{ font-size:clamp(28px,5.4vw,50px) !important; font-weight:800; letter-spacing:-.6px; }
.subtitle{ text-align:center; font-size:clamp(13px,2.2vw,18px); color:#8b98ad;
  margin-top:4px; }

/* segmented pill tabs */
[data-baseweb="tab-list"]{ justify-content:center; gap:4px; background:#111a2e;
  padding:5px; border-radius:12px; border:1px solid #232f49; width:fit-content;
  margin:10px auto 16px; }
[data-baseweb="tab"]{ height:auto; padding:9px 22px; border-radius:9px;
  color:#93a1b8; font-weight:600; font-size:14px; }
button[aria-selected="true"][data-baseweb="tab"]{
  background:linear-gradient(135deg,#7C3AED,#9333EA); color:#fff; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{ display:none !important; }

/* file uploader */
[data-testid="stFileUploader"]{ border:1.5px solid #2a3550; border-radius:14px;
  background:#0f1830; padding:14px 16px; }
[data-testid="stFileUploader"] section{ background:transparent; border:none; padding:0; }

/* camera */
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img{ border-radius:14px; }

/* buttons */
.stButton>button{ background:linear-gradient(135deg,#7C3AED,#9333EA); color:#fff;
  border:none; border-radius:11px; height:46px; font-weight:600; letter-spacing:.2px;
  width:100%; transition:.18s ease; box-shadow:0 8px 20px rgba(124,58,237,.28); }
.stButton>button:hover{ filter:brightness(1.08); transform:translateY(-1px);
  box-shadow:0 12px 26px rgba(124,58,237,.42); }
.stButton>button:active{ transform:translateY(0); }
[data-testid="stSpinner"]{ color:#A855F7; }
hr{ border-color:#1b2740; }

/* result images — uniform portrait cards */
[data-testid="stImage"] img{ border-radius:14px; aspect-ratio:3/4; object-fit:cover;
  width:100%; border:1px solid rgba(255,255,255,.06);
  box-shadow:0 14px 30px rgba(0,0,0,.45); }

/* empty state */
.empty-card{ background:#0f1830; border:1px solid #202d49; border-radius:18px;
  padding:26px 22px; text-align:center; }
.empty-card h3{ margin:0 0 4px; font-size:21px; font-weight:700; }
.empty-card p{ color:#8b98ad; margin:0; }
.steps{ display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin-top:18px; }
.step{ background:#0b1428; border:1px solid #263453; border-radius:11px;
  padding:10px 14px; font-size:13px; color:#c3cede; }
.step b{ color:#A855F7; }

/* result card */
.match-header{ background:#0f1830; border:1px solid #26324f; border-radius:12px;
  padding:13px 20px; text-align:center; font-size:clamp(16px,3vw,22px); font-weight:600;
  color:#c3cede; margin:2px 0 16px; }
.match-header b{ color:#A855F7; }
.pic-label{ text-align:center; font-weight:600; font-size:14px; color:#93a1b8;
  margin:0 0 8px; letter-spacing:.4px; }

/* confidence ring (sits between the two photos) */
.ring-wrap{ display:flex; flex-direction:column; align-items:center;
  justify-content:center; height:100%; min-height:210px; }
.ring-cap{ color:#5f6d84; font-size:11px; letter-spacing:2px; margin-bottom:10px; }
.ring{ position:relative; width:110px; height:110px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; background:#1b2740; }
.ring::before{ content:""; position:absolute; inset:12px;
  border-radius:50%; background:#0b1120; }
.ring .val{ position:relative; font-weight:800; font-size:21px; color:#fff; }

.top3-title{ text-align:center; color:#5f6d84; font-size:12px; margin:24px 0 8px;
  letter-spacing:2px; }
.top3-name{ text-align:center; font-weight:600; margin:8px 0 0; font-size:13px;
  color:#c3cede; }
.top3-name span{ color:#A855F7; }

.error-card{ background:#241521; border:1px solid #6d1d36; border-left:4px solid #EF4444;
  padding:14px 20px; border-radius:12px; font-size:clamp(14px,2.4vw,17px);
  font-weight:600; color:#f2c9d3; margin:12px 0; text-align:center; }

.app-footer{ text-align:center; color:#5f6d84; font-size:14px; line-height:1.9;
  margin-top:8px; }
.app-footer b{ color:#93a1b8; }

/* feedback modal card */
.feedback-card{ background:#0f1830; border:1px solid #2a3550; border-radius:18px;
  padding:24px 22px 12px; text-align:center; box-shadow:0 24px 60px rgba(0,0,0,.55); }
.feedback-card h3{ margin:0 0 4px; font-size:22px; font-weight:700; }
.feedback-card p{ color:#8b98ad; margin:0; }

/* floating "Contact me" chick */
.chick-launcher{ position:fixed; top:16px; right:22px; z-index:9999;
  text-decoration:none; display:flex; align-items:center; gap:8px;
  animation:zigzag 4s ease-in-out infinite; }
.chick-launcher .chick{ font-size:30px; display:inline-block; position:relative;
  animation:flap 1s ease-in-out infinite;
  filter:drop-shadow(0 3px 5px rgba(0,0,0,.4)); }
.chick-launcher .chick::before{ content:""; position:absolute; left:-5px; top:8px;
  width:28px; height:28px; z-index:-1; border-radius:50%; filter:blur(6px);
  background:radial-gradient(circle,rgba(203,213,225,.5),transparent 70%);
  animation:smoke 2.6s ease-in-out infinite; }
.chick-launcher .chick-label{ background:linear-gradient(135deg,#7C3AED,#9333EA);
  color:#fff; font-weight:600; font-size:13px; padding:7px 13px; border-radius:999px;
  white-space:nowrap; box-shadow:0 6px 16px rgba(124,58,237,.4); }
@keyframes zigzag{ 0%,100%{transform:translate(0,0);} 25%{transform:translate(-8px,6px);}
  50%{transform:translate(6px,-5px);} 75%{transform:translate(-5px,4px);} }
@keyframes flap{ 0%,100%{transform:rotate(-6deg);} 50%{transform:rotate(6deg);} }
@keyframes smoke{ 0%{opacity:.15;transform:scale(.7);} 50%{opacity:.5;transform:scale(1.15);}
  100%{opacity:.15;transform:scale(.7);} }

@media (max-width:768px){
  .block-container{ padding-top:1rem; }
  .ring-wrap{ min-height:80px; }
  .chick-launcher{ top:8px; right:10px; }
  .chick-launcher .chick{ font-size:26px; }
  .chick-launcher .chick-label{ font-size:11px; padding:5px 9px; }
}
::-webkit-scrollbar{ width:9px; }
::-webkit-scrollbar-thumb{ background:#2a3550; border-radius:10px; }
::-webkit-scrollbar-track{ background:#0b1120; }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

if "stage" not in st.session_state:
    st.session_state.stage = "idle"
STAGE = st.session_state.stage


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def show_error(msg):
    st.markdown('<div class="error-card">' + msg + '</div>', unsafe_allow_html=True)


def render_results(user_data, matches):
    main = matches[0]
    st.markdown('<div class="match-header">You resemble <b>' + main["name"]
                + '</b></div>', unsafe_allow_html=True)

    pct = main["score"] * 100
    c1, cmid, c2 = st.columns([5, 3, 5])
    with c1:
        st.markdown('<p class="pic-label">YOU</p>', unsafe_allow_html=True)
        show_img(Image.open(io.BytesIO(user_data)).convert("RGB"))
    with cmid:
        arc, disp = ("%.1f" % pct), ("%.0f" % pct)
        st.markdown(
            '<div class="ring-wrap"><div class="ring-cap">MATCH</div>'
            '<div class="ring" style="background:conic-gradient(#7C3AED '
            + arc + '%,#1b2740 0)"><span class="val">' + disp
            + '%</span></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<p class="pic-label">' + main["name"] + '</p>',
                    unsafe_allow_html=True)
        show_img(main["path"])

    others = matches[1:]
    if others:
        st.markdown('<p class="top3-title">YOU ALSO RESEMBLE</p>',
                    unsafe_allow_html=True)
        cols = st.columns(len(others))
        for col, m in zip(cols, others):
            with col:
                show_img(m["path"])
                st.markdown('<p class="top3-name">' + m["name"] + '<br><span>'
                            + ("%.1f" % (m["score"] * 100)) + '%</span></p>',
                            unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <h1 style='text-align:center;'>Celebrity Face Matcher</h1>
    <p class="subtitle">Upload or capture your photo and discover which Bollywood
    celebrity you resemble.</p>
    """, unsafe_allow_html=True)


def render_chick():
    st.markdown("""
    <a class="chick-launcher" href="/Contact_Me" target="_self" title="Contact me">
      <span class="chick">🐤</span><span class="chick-label">Contact me</span>
    </a>
    """, unsafe_allow_html=True)


def reset_to_idle():
    for k in ("stage", "data", "method", "fname", "guess", "result",
              "guess_box", "logged_id"):
        st.session_state.pop(k, None)
    st.session_state.stage = "idle"


# =========================================================================== #
# FLOW:  idle  ->  ask  ->  done
# =========================================================================== #

# ---- ASK: focused question over a blurred version of the user's own photo ----
if STAGE == "ask":
    data = st.session_state.get("data", b"")
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    b64 = base64.b64encode(data).decode()
    st.markdown(
        '<style>'
        '.stApp::before{content:"";position:fixed;inset:0;z-index:0;'
        "background:url('data:" + mime + ";base64," + b64 + "') center/cover no-repeat;"
        'filter:blur(22px) brightness(.45);transform:scale(1.15);}'
        '.stApp::after{content:"";position:fixed;inset:0;z-index:1;'
        'background:linear-gradient(180deg,rgba(8,12,24,.35),rgba(8,12,24,.7));}'
        '[data-testid="stAppViewContainer"] .block-container{position:relative;z-index:2;}'
        '</style>', unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""
        <div class="feedback-card">
          <h3>Before the reveal</h3>
          <p>Who do <b>you</b> think you look like? <i>(optional)</i></p>
        </div>
        """, unsafe_allow_html=True)
        guess = st.text_input("guess", key="guess_box",
                              label_visibility="collapsed",
                              placeholder="Type a name — or just skip")
        b1, b2 = st.columns(2)
        reveal = b1.button("Reveal my match", key="reveal")
        skip = b2.button("Skip", key="skip")
        if reveal or skip:
            st.session_state.guess = "" if skip else guess
            st.session_state.stage = "done"
            _rerun()

# ---- IDLE + DONE ----
else:
    render_chick()
    render_header()

    if STAGE == "idle":
        _, center, _ = st.columns([1, 2, 1])
        with center:
            tab_upload, tab_camera = st.tabs(["Upload Photo", "Take Photo"])
            with tab_upload:
                uploaded_image = st.file_uploader("up", type=["jpg", "jpeg", "png"],
                                                  label_visibility="collapsed")
            with tab_camera:
                camera_image = st.camera_input("cam", label_visibility="collapsed")

        image_input = uploaded_image or camera_image
        if image_input is not None:
            st.session_state.data = image_input.getvalue()
            st.session_state.method = "upload" if uploaded_image else "camera"
            st.session_state.fname = getattr(image_input, "name", "capture.jpg")
            st.session_state.guess = ""
            st.session_state.pop("result", None)
            st.session_state.stage = "ask"
            _rerun()
        else:
            st.markdown("""
            <div class="empty-card">
              <h3>Ready when you are</h3>
              <p>Pick a tab above — upload a clear face photo or snap one live.</p>
              <div class="steps">
                <div class="step"><b>1</b> &nbsp;Detect your face</div>
                <div class="step"><b>2</b> &nbsp;Read facial features</div>
                <div class="step"><b>3</b> &nbsp;Find your celebrity twin</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    elif STAGE == "done":
        data = st.session_state.get("data")
        if not data:
            reset_to_idle()
            _rerun()

        img_id = hashlib.md5(data).hexdigest()
        cache = st.session_state.get("result")
        if not cache or cache.get("id") != img_id:
            try:
                matrix, names = load_embeddings()
                with st.spinner("Analyzing your face…"):
                    outcome = analyze_faces(decode_bgr(data))
                if "error" in outcome:
                    cache = {"id": img_id, "error": outcome["error"]}
                else:
                    with st.spinner("Finding your celebrity twin…"):
                        feats = embed_face(outcome["face"])
                        cache = {"id": img_id,
                                 "matches": top_matches(feats, matrix, names, 3)}
            except Exception:
                cache = {"id": img_id,
                         "error": "Something went wrong analyzing this photo. "
                                  "Please try another."}
            st.session_state.result = cache

        if "error" in cache:
            show_error(cache["error"])
        else:
            render_results(data, cache["matches"])
            # log AFTER rendering so results never wait on network I/O
            cfg = get_secrets()
            if (gsheet_logger is not None
                    and st.session_state.get("logged_id") != img_id
                    and gsheet_logger.is_configured(cfg)):
                st.session_state.logged_id = img_id
                m = cache["matches"][0]
                with st.spinner("Saving…"):
                    gsheet_logger.log_match(
                        method=st.session_state.get("method", "upload"),
                        celebrity=m["name"], score=m["score"],
                        user_guess=st.session_state.get("guess", ""),
                        user_image_bytes=data,
                        user_image_name=st.session_state.get("fname", "capture.jpg"),
                        celeb_image_path=m["path"], config=cfg)

        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if st.button("Try another photo", key="again"):
                reset_to_idle()
                _rerun()

    st.markdown("---")
    st.markdown("""
    <div class="app-footer">
      Powered by <b>TensorFlow</b> · <b>VGGFace</b> · <b>Streamlit</b><br>
      Made by <b>Hamzurna</b>
    </div>
    """, unsafe_allow_html=True)
