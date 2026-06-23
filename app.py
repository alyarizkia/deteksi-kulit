import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
import os
import gdown

# PAGE CONFIG
st.set_page_config(
    page_title="Skin Type Detection",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# DOWNLOAD MODEL DARI GOOGLE DRIVE
def download_models():

    os.makedirs("models", exist_ok=True)

    male_model = "models/faster_rcnn_best.pth"
    female_model = "models/faster_rcnn_best_tuned.pth"

    # ganti dengan ID file Google Drive milikmu
    male_url = "https://drive.google.com/file/d/1MKg7gCsY3Obu9llGV60rpNp6YndACgkC/view?usp=drive_link"
    female_url = "https://drive.google.com/file/d/1AqUh5AEGEZj2-akzRGkted4CbPxiSuNc/view?usp=drive_link"

    if not os.path.exists(male_model):
        with st.spinner("Mengunduh model laki-laki..."):
            gdown.download(male_url, male_model, quiet=False, fuzzy=True)

    if not os.path.exists(female_model):
        with st.spinner("Mengunduh model perempuan..."):
            gdown.download(female_url, female_model, quiet=False, fuzzy=True)

download_models()

# CONSTANTS
MODEL_PATHS = {
    "Male":   "models/faster_rcnn_best.pth",
    "Female": "models/faster_rcnn_best_tuned.pth",
}

CLASS_NAMES = {
    "Male":   ["men_acne", "men_dry", "men_normal", "men_oily"],
    "Female": ["women_acne", "women_dry", "women_normal", "women_oily"],
}

CLASS_COLORS = {
    "acne":   "#E8605A",
    "dry":    "#6DA3D4",
    "normal": "#52C48A",
    "oily":   "#E8B85A",
}

SKIN_TYPE_DESC = {
    "acne": {
        "label": "Berjerawat",
        "desc": "terdapat jerawat atau peradangan pada kulit",
    },
    "dry": {
        "label": "Kering",
        "desc": "kulit tampak kering dan kurang kelembapan",
    },
    "normal": {
        "label": "Normal",
        "desc": "kulit tampak sehat dan seimbang",
    },
    "oily": {
        "label": "Berminyak",
        "desc": "kulit tampak berminyak terutama di area T-zone",
    },
}

CONF_THRESHOLD = 0.5
IOU_THRESHOLD  = 0.45

# STYLING
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp {
    background: #0c0c12;
    color: #dddde8;
}

footer { visibility: hidden; }

.progress-wrap {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2.5rem;
}
.prog-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    flex: 1;
    position: relative;
}
.prog-circle {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
    z-index: 2; position: relative;
}
.prog-circle.done   { background: #52C48A; color: #0c0c12; }
.prog-circle.active { background: #7c6fff; color: #fff; box-shadow: 0 0 0 4px #7c6fff30; }
.prog-circle.idle   { background: #1e1e2e; color: #444460; border: 1.5px solid #2a2a3e; }
.prog-label {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #444460;
}
.prog-label.active  { color: #7c6fff; }
.prog-label.done    { color: #52C48A; }
.prog-line {
    flex: 1;
    height: 2px;
    background: #1e1e2e;
    margin-bottom: 22px;
}
.prog-line.done { background: #52C48A; }

.gender-card {
    background: #13131e;
    border: 2px solid #1e1e2e;
    border-color: #7c6fff; background: #18182a;
    border-radius: 20px;
    padding: 2rem 1rem;
    text-align: center;
    cursor: default !important;
    transition: all 0.18s ease;
}
.gender-card.active { border-color: #7c6fff; background: #18182a; box-shadow: 0 0 0 4px #7c6fff18; }
.gender-icon  { font-size: 2.8rem; margin-bottom: 0.5rem; }
.gender-title { font-size: 1rem; font-weight: 600; color: #dddde8; }
.gender-sub   { font-size: 0.75rem; color: #555570; margin-top: 0.15rem; }

.result-summary {
    background: linear-gradient(135deg, #13131e 0%, #18182a 100%);
    border: 1px solid #2a2a3e;
    border-radius: 20px;
    padding: 1.75rem;
    margin-bottom: 1rem;
}
.summary-eyebrow {
    font-size: 0.7rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: #555570;
    margin-bottom: 0.4rem;
}
.summary-headline {
    font-size: 1.4rem; font-weight: 700; color: #dddde8;
    margin-bottom: 0.5rem; line-height: 1.3;
}
.summary-body { font-size: 0.88rem; color: #8888a8; line-height: 1.6; }

.det-card {
    background: #13131e;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.det-badge {
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    white-space: nowrap;
}
.det-info { flex: 1; }
.det-conf  { font-size: 0.75rem; color: #555570; }
.conf-bar-wrap { width: 90px; }
.conf-bar-bg   { background: #1e1e2e; border-radius: 999px; height: 5px; }
.conf-bar-fill { height: 5px; border-radius: 999px; }

/* ── Tombol secondary (default) ── */
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background-color: #1e1e2e !important;
    color: #dddde8 !important;
    border: 1.5px solid #2a2a3e !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.65rem 1rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
    background-color: #2a2a3e !important;
    color: #ffffff !important;
    border-color: #7c6fff !important;
}
.stButton > button[kind="secondary"]:active,
.stButton > button:not([kind="primary"]):active {
    background-color: #7c6fff !important;
    color: #ffffff !important;
    border-color: #7c6fff !important;
}
 
/* ── Tombol primary ── */
.stButton > button[kind="primary"] {
    background-color: #7c6fff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.65rem 1rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #6a5dee !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:active {
    background-color: #5a4ddd !important;
    color: #ffffff !important;
}
 
/* ── Tombol browse file (file uploader) ── */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploadDropzone"] button {
    background-color: #1e1e2e !important;
    color: #dddde8 !important;
    border: 1.5px solid #2a2a3e !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploadDropzone"] button:hover {
    background-color: #2a2a3e !important;
    color: #ffffff !important;
    border-color: #7c6fff !important;
}
[data-testid="stFileUploaderDropzone"] button:active,
[data-testid="stFileUploadDropzone"] button:active {
    background-color: #7c6fff !important;
    color: #ffffff !important;
}

.section-title {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #555570; margin: 1.25rem 0 0.75rem 0;
}

hr.divider { border: none; border-top: 1px solid #1e1e2e; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# SESSION STATE
for key, default in {
    "page": 1,
    "gender": None,
    "model_loaded": False,
    "detections": [],
    "annotated_img": None,
    "original_img": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# HELPERS
@st.cache_resource(show_spinner=False)
def load_model(gender: str):
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    import pathlib

    path = MODEL_PATHS[gender]
    if not pathlib.Path(path).exists():
        return None

    num_classes = len(CLASS_NAMES[gender]) + 1  # +1 untuk background

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    # Coba deteksi otomatis: state dict atau full model
    if isinstance(checkpoint, dict) and ("model_state_dict" in checkpoint or any(k.startswith("backbone") for k in checkpoint.keys())):
        # Ini state dict
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        # Ini full model
        model = checkpoint

    model.to(device)
    model.eval()
    return model


def get_skin_key(class_name: str) -> str:
    for key in SKIN_TYPE_DESC:
        if key in class_name.lower():
            return key
    return "normal"


def get_badge_style(class_name: str) -> str:
    for key, color in CLASS_COLORS.items():
        if key in class_name.lower():
            return f"background:{color}22; color:{color}; border:1px solid {color}55;"
    return "background:#2a2a3e; color:#8888a8;"


def get_bar_color(class_name: str) -> str:
    for key, color in CLASS_COLORS.items():
        if key in class_name.lower():
            return color
    return "#7c6fff"


def run_inference(model, image: Image.Image):
    import torch
    import torchvision.transforms.functional as F
    import cv2
    import numpy as np
    from torchvision.ops import nms

    device = next(model.parameters()).device
    img_rgb = image.convert("RGB")
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    detections = []
    img_draw = np.array(img_rgb).copy()

    boxes  = outputs["boxes"]
    scores = outputs["scores"]
    labels = outputs["labels"]

    # Filter confidence dulu
    keep_conf = scores >= CONF_THRESHOLD
    boxes, scores, labels = boxes[keep_conf], scores[keep_conf], labels[keep_conf]

    # Lalu NMS
    keep_nms = nms(boxes, scores, iou_threshold=IOU_THRESHOLD)
    boxes  = boxes[keep_nms]
    scores = scores[keep_nms]
    labels = labels[keep_nms]

    for i in range(len(outputs["boxes"])):
        score = float(outputs["scores"][i])
        if score < CONF_THRESHOLD:
            continue

        cls_id   = int(outputs["labels"][i])
        cls_name = CLASS_NAMES[st.session_state.gender][cls_id - 1]  # -1 karena index 0 = background
        bbox     = outputs["boxes"][i].tolist()  # [x1, y1, x2, y2]

        detections.append({"class": cls_name, "conf": score, "bbox": bbox})

        # Gambar bounding box manual
        color = tuple(int(get_bar_color(cls_name).lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_draw, f"{cls_name} {score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    detections.sort(key=lambda d: d["conf"], reverse=True)
    annotated = Image.fromarray(img_draw)
    return annotated, detections


def build_summary(detections: list) -> dict:
    if not detections:
        return None

    skin_keys = [get_skin_key(d["class"]) for d in detections]
    counts    = Counter(skin_keys)
    dominant  = counts.most_common(1)[0][0]
    total     = len(detections)
    others    = [k for k in counts if k != dominant]

    label         = SKIN_TYPE_DESC[dominant]["label"]
    headline      = f"Kulit {label}"
    dominant_pct  = round(counts[dominant] / total * 100)

    body_parts = [
        f"Kulit wajah ini {SKIN_TYPE_DESC[dominant]['desc']} "
        f"({dominant_pct}% dari area terdeteksi)."
    ]

    if others:
        other_descs = []
        for o in others:
            pct = round(counts[o] / total * 100)
            other_descs.append(f"{SKIN_TYPE_DESC[o]['desc']} ({pct}%)")
        body_parts.append("Selain itu, " + " dan ".join(other_descs) + ".")
    else:
        body_parts.append("Tidak ditemukan kondisi lain yang signifikan.")

    return {
        "headline": headline,
        "body":     " ".join(body_parts),
        "dominant": dominant,
        "counts":   counts,
        "total":    total,
    }


def progress_bar(current_page: int):
    steps = [("Pilih Gender", 1), ("Load Model", 2), ("Deteksi", 3)]
    html  = '<div class="progress-wrap">'
    for i, (label, num) in enumerate(steps):
        if num < current_page:
            circle_cls, label_cls, icon = "done",   "done",   "✓"
        elif num == current_page:
            circle_cls, label_cls, icon = "active", "active", str(num)
        else:
            circle_cls, label_cls, icon = "idle",   "",       str(num)

        if i > 0:
            line_cls = "done" if num <= current_page else ""
            html += f'<div class="prog-line {line_cls}"></div>'

        html += f"""
        <div class="prog-step">
            <div class="prog-circle {circle_cls}">{icon}</div>
            <div class="prog-label {label_cls}">{label}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# PAGE 1 — GENDER SELECTION
def page_gender():
    progress_bar(1)
    st.markdown("## Pilih Gender")
    st.markdown(
        "<p style='color:#555570; margin-top:-0.4rem; margin-bottom:1.5rem; font-size:0.9rem;'>"
        "Sistem akan memuat model yang sesuai dengan gender yang dipilih."
        "</p>",
        unsafe_allow_html=True,
    )

    male_active   = "active" if st.session_state.gender == "Male"   else ""
    female_active = "active" if st.session_state.gender == "Female" else ""

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="gender-card {male_active}">
            <div class="gender-icon">👨</div>
            <div class="gender-title">Laki-laki</div>
            <div class="gender-sub">Model Faster R-CNN — Men</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Pilih Laki-laki", key="sel_male", use_container_width=True):
            st.session_state.gender = "Male"
            st.session_state.model_loaded = False
            st.session_state.page = 2
            st.rerun()

    with col2:
        st.markdown(f"""
        <div class="gender-card {female_active}">
            <div class="gender-icon">👩</div>
            <div class="gender-title">Perempuan</div>
            <div class="gender-sub">Model Faster R-CNN — Women</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Pilih Perempuan", key="sel_female", use_container_width=True):
            st.session_state.gender = "Female"
            st.session_state.model_loaded = False
            st.session_state.page = 2
            st.rerun()


# PAGE 2 — LOAD MODEL
def page_load_model():
    progress_bar(2)
    gender = st.session_state.gender
    icon   = "👨" if gender == "Male" else "👩"

    st.markdown(f"## Memuat Model {icon}")
    st.markdown(
        f"<p style='color:#555570; font-size:0.9rem; margin-top:-0.4rem; margin-bottom:1.5rem;'>"
        f"Menggunakan model <strong style='color:#dddde8;'>Faster R-CNN — {gender}</strong> "
        f"dari <code>{MODEL_PATHS[gender]}</code>"
        f"</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.model_loaded:
        with st.spinner("Memuat model, harap tunggu..."):
            model = load_model(gender)
        if model is None:
            st.error(
                f"⚠️ File model tidak ditemukan di `{MODEL_PATHS[gender]}`.\n\n"
                "Pastikan file `.pt` sudah ada di folder `models/`."
            )
            if st.button("← Kembali", key="back_from_err"):
                st.session_state.page = 1
                st.rerun()
            return
        st.session_state.model_loaded = True

    st.markdown(f"""
    <div style='background:#13131e; border:1px solid #1e1e2e; border-radius:16px; padding:1.75rem; margin-bottom:1.5rem;'>
        <div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;'>
            <span style='font-size:1.5rem;'>✅</span>
            <span style='font-size:1rem; font-weight:600; color:#52C48A;'>Model siap digunakan</span>
        </div>
        <div style='font-size:0.8rem; color:#555570; line-height:1.7;'>
            Gender: <strong style='color:#dddde8;'>{gender}</strong><br>
            Path: <code style='color:#8888a8;'>{MODEL_PATHS[gender]}</code><br>
            Confidence threshold: <strong style='color:#dddde8;'>{CONF_THRESHOLD}</strong> &nbsp;|&nbsp;
            IoU threshold: <strong style='color:#dddde8;'>{IOU_THRESHOLD}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("← Ganti Gender", key="go_back_gender", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    with col2:
        if st.button("Mulai Deteksi →", key="go_detect", type="primary", use_container_width=True):
            st.session_state.page = 3
            st.rerun()


# PAGE 3 — DETECTION
def page_detect():
    progress_bar(3)
    gender = st.session_state.gender
    model  = load_model(gender)
    icon   = "👨" if gender == "Male" else "👩"

    st.markdown(f"## Deteksi Tipe Kulit {icon}")
    st.markdown(
        f"<p style='color:#555570; font-size:0.9rem; margin-top:-0.4rem; margin-bottom:1.5rem;'>"
        f"Gender: <strong style='color:#dddde8;'>{gender}</strong> — Upload foto wajah untuk dianalisis."
        f"</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload foto wajah (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"uploader_{gender}",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.original_img = image

        with st.spinner("Menjalankan deteksi..."):
            annotated, detections = run_inference(model, image)
            st.session_state.annotated_img = annotated
            st.session_state.detections    = detections

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-title'>Foto Asli</div>", unsafe_allow_html=True)
            st.image(image, width="stretch")
        with col_b:
            st.markdown("<div class='section-title'>Hasil Deteksi</div>", unsafe_allow_html=True)
            st.image(annotated, width="stretch")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Kesimpulan</div>", unsafe_allow_html=True)

        summary = build_summary(detections)

        if summary is None:
            st.markdown("""
            <div class='result-summary'>
                <div class='summary-eyebrow'>Hasil Analisis</div>
                <div class='summary-headline'>Tidak ada deteksi</div>
                <div class='summary-body'>
                    Sistem tidak mendeteksi area kulit yang jelas. Coba gunakan foto wajah
                    yang lebih terang dan resolusi lebih tinggi.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            dominant_color = CLASS_COLORS.get(summary["dominant"], "#7c6fff")
            st.markdown(f"""
            <div class='result-summary'>
                <div class='summary-eyebrow'>Hasil Analisis</div>
                <div class='summary-headline' style='color:{dominant_color};'>{summary['headline']}</div>
                <div class='summary-body'>{summary['body']}</div>
            </div>
            """, unsafe_allow_html=True)

        if detections:
            st.markdown("<div class='section-title'>Detail Bounding Box</div>", unsafe_allow_html=True)
            for det in detections:
                conf_pct    = det["conf"] * 100
                badge_style = get_badge_style(det["class"])
                bar_color   = get_bar_color(det["class"])
                bar_width   = round(conf_pct)

                st.markdown(f"""
                <div class='det-card'>
                    <span class='det-badge' style='{badge_style}'>{det['class']}</span>
                    <div class='det-info'>
                        <div class='det-conf'>Conf: {conf_pct:.1f}%</div>
                    </div>
                    <div class='conf-bar-wrap'>
                        <div class='conf-bar-bg'>
                            <div class='conf-bar-fill' style='width:{bar_width}%; background:{bar_color};'></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Lakukan Deteksi Lagi?</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄  Ganti Gender", key="switch_gender", use_container_width=True):
                st.session_state.page         = 1
                st.session_state.gender       = None
                st.session_state.model_loaded = False
                st.session_state.detections   = []
                st.rerun()
        with col2:
            if st.button("📷  Ganti Foto (Gender Sama)", key="same_gender", type="primary", use_container_width=True):
                st.session_state.detections    = []
                st.session_state.annotated_img = None
                st.rerun()

    else:
        st.markdown("""
        <div style='
            background:#13131e; border:1.5px dashed #2a2a3e; border-radius:20px;
            padding:3rem; text-align:center; color:#333350; margin: 1rem 0 1.5rem 0;
        '>
            <div style='font-size:2.5rem; margin-bottom:0.75rem;'>📷</div>
            <div style='font-size:1rem; font-weight:600; color:#555570;'>Upload foto wajah</div>
            <div style='font-size:0.8rem; margin-top:0.3rem;'>JPG, PNG, atau WEBP</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Kembali", key="back_to_load", use_container_width=True):
            st.session_state.page = 2
            st.rerun()


# ROUTER
page = st.session_state.page

if page == 1:
    page_gender()
elif page == 2:
    page_load_model()
elif page == 3:
    page_detect()
