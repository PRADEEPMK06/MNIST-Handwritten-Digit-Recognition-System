"""
app.py
------
Interactive Streamlit demo for the handwritten character recognition model.
Draw a digit or letter on the board (or upload an image) and get a
prediction with confidence scores. Every graded attempt is kept in a
persistent "homework pile" gallery below (saved to disk, so it survives
restarts) along with a chart of average confidence per character.
 
Usage:
    streamlit run app.py
 
CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
"""

import base64
import io
import json
import os
import sys
import time
from collections import defaultdict
 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
 
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from labels import LABEL_MAP  # noqa: E402
 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "handwritten_char_cnn.keras")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "data", "prediction_history.json")
# Caps how much history persists to disk so the file doesn't grow forever
# across many sessions - still far more than one sitting would produce.
MAX_HISTORY = 300
 
st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✏️",
    layout="wide",
)
 
# --------------------------------------------------------------------------
# Design: a chalkboard. MNIST/EMNIST images ARE white-strokes-on-black, so
# the board isn't a decorative skin - it's the same convention the model
# was trained on. Three muted chalk colors do the accent work instead of
# one neon highlight, and the "signature" moment is a hand-drawn chalk
# circle that draws itself around the answer, like a teacher grading
# homework.
#
# Styling note: widgets are wrapped with st.container(key=...) rather than
# raw st.markdown("<div>...</div>") pairs. Streamlit renders each markdown
# call as its own sibling node, so an opening/closing div written in two
# separate calls never actually encloses the native widgets in between -
# container(key=...) is what gives a real wrapping element (as .st-key-*)
# that CSS can target.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Permanent+Marker&family=Work+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
 
    :root {
        --board: #F6F1E1;
        --board-rail: #16241C;
        --chalk-white: #F2EFE4;
        --chalk-yellow: #E8C468;
        --chalk-coral: #E08577;
        --chalk-blue: #8FB4D1;
        --chalk-dust: rgba(242, 239, 228, 0.22);
        /* "ink" set: for text/borders that sit on the light --board itself,
           where the pale chalk-* colors above would be unreadable. */
        --ink: #33312A;
        --ink-soft: rgba(51, 49, 42, 0.62);
        --ink-dust: rgba(51, 49, 42, 0.22);
        --ink-yellow: #9C6F17;
    }
 
    .stApp { background: var(--board); }
    [data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    * { font-family: 'Work Sans', sans-serif; }
 
    .rail {
        background: var(--board-rail);
        border-bottom: 3px solid var(--chalk-dust);
        margin: -1.5rem -1rem 1.75rem -1rem;
        padding: 1.6rem 2rem 1.3rem 2rem;
    }
    .rail h1 {
        font-family: 'Permanent Marker', cursive;
        color: var(--chalk-white);
        font-size: 2.6rem;
        margin: 0;
        letter-spacing: 0.5px;
        transform: rotate(-0.6deg);
    }
    .rail p { color: var(--chalk-dust); font-size: 1rem; margin: 0.3rem 0 0 0.15rem; }
 
    /* Real wrapping elements via container(key=...) -> .st-key-<key> */
    .st-key-draw_tile, .st-key-result_tile, .st-key-history_tile, .st-key-chart_tile {
        background: rgba(0,0,0,0.03);
        border: 2px dashed var(--ink-dust);
        border-radius: 6px;
        padding: 1.2rem 1.4rem 1.4rem 1.4rem;
        margin-bottom: 1rem;
    }
    .tile-label {
        font-family: 'JetBrains Mono', monospace;
        color: var(--ink-yellow);
        font-size: 0.78rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }
 
    /* streamlit-drawable-canvas' outer iframe is wider than the 280px
       canvas drawn inside it, leaving a plain box around the board -
       constrain the iframe itself. */
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        max-width: 282px !important;
        border-radius: 4px;
    }
 
    [data-testid="stButton"] button {
        background: transparent;
        color: var(--ink);
        border: 2px solid var(--ink-dust);
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        padding: 0.5rem 1.2rem;
    }
    [data-testid="stButton"] button:hover,
    [data-testid="stButton"] button:focus:not(:active) {
        border-color: var(--ink-yellow);
        color: var(--ink-yellow);
    }
 
    [data-testid="stExpander"] { border-color: var(--ink-dust) !important; }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] p { color: var(--ink-soft) !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(0,0,0,0.03) !important;
        border: 1px dashed var(--ink-dust) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: transparent !important;
        color: var(--ink) !important;
        border: 1px solid var(--ink-dust) !important;
    }
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] div { color: var(--ink-soft) !important; }
 
    .grade-slip { display: flex; align-items: center; gap: 1.6rem; min-height: 190px; }
    .circle-wrap { position: relative; width: 150px; height: 150px; flex-shrink: 0; }
    .circle-wrap svg { position: absolute; top: 0; left: 0; }
    .circle-char {
        position: absolute; inset: 0;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Permanent Marker', cursive;
        font-size: 4.2rem;
        color: var(--ink);
    }
    .chalk-path {
        stroke-dasharray: 420;
        stroke-dashoffset: 420;
        animation: draw 0.7s ease-out forwards;
    }
    @keyframes draw { to { stroke-dashoffset: 0; } }
 
    .bar-row {
        display: flex; align-items: center; gap: 0.7rem;
        margin-bottom: 0.55rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--ink);
    }
    .bar-char { width: 1.4rem; opacity: 0.9; }
    .bar-track { flex: 1; background: rgba(0,0,0,0.08); border-radius: 3px; height: 10px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 3px; }
    .bar-pct { width: 3.2rem; text-align: right; opacity: 0.85; }
 
    .empty-slip { color: var(--ink-soft); font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }
    .empty-slip code {
        background: rgba(0,0,0,0.06);
        color: var(--ink-yellow);
        padding: 0.15rem 0.4rem;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
    }
 
    /* Homework pile - the prediction history gallery */
    .pile-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(72px, 1fr));
        gap: 0.7rem;
    }
    .pile-card {
        background: rgba(0,0,0,0.03);
        border: 1px solid var(--ink-dust);
        border-radius: 4px;
        padding: 0.4rem;
        text-align: center;
    }
    .pile-card img {
        width: 100%;
        aspect-ratio: 1;
        border-radius: 3px;
        background: #000;
        display: block;
        margin-bottom: 0.3rem;
    }
    .pile-char {
        font-family: 'Permanent Marker', cursive;
        font-size: 1.3rem;
        color: var(--ink);
        line-height: 1;
    }
    .pile-conf {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--ink-soft);
    }
    .pile-count {
        font-family: 'JetBrains Mono', monospace;
        color: var(--ink-soft);
        font-size: 0.78rem;
        margin-bottom: 0.8rem;
    }

    /* Report-card confidence chart legend */
    .legend-row {
        display: flex;
        gap: 1.4rem;
        flex-wrap: wrap;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--ink-soft);
        margin-top: 0.2rem;
    }
    .legend-row i {
        display: inline-block;
        width: 0.6rem;
        height: 0.6rem;
        border-radius: 2px;
        margin-right: 0.35rem;
        vertical-align: middle;
    }
 
    .foot-rail {
        background: var(--board-rail);
        border-top: 3px solid var(--chalk-dust);
        margin: 2rem -1rem -1rem -1rem;
        padding: 0.9rem 2rem;
        color: var(--chalk-dust);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
st.markdown(
    """
    <div class="rail">
        <h1>Handwritten Character Recognition</h1>
        <p>Draw a digit (0-9) or a letter (A-Z) — I'll grade it like homework.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
 
@st.cache_resource
def load_char_model():
    if not os.path.exists(MODEL_PATH):
        return None
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)
 
 
def preprocess_array(img_gray_28):
    """img_gray_28: 2D array, values 0-255, white stroke on black."""
    img = img_gray_28.astype(np.float32) / 255.0
    return img[np.newaxis, ..., np.newaxis]


def crop_and_center(gray, margin_frac=0.18, threshold=20):
    """Crop a white-stroke-on-black image to the drawn character's
    bounding box, then pad it back into a square with a margin.

    MNIST/EMNIST characters are always tightly cropped and centered in
    their 28x28 frame with a small border - never edge-to-edge. Without
    this step, a small or off-center canvas drawing gets squashed by the
    raw resize into something that looks nothing like the training
    distribution, which is why predictions (especially for digits) were
    inconsistent depending on where/how big the user drew on the board.

    Returns the original array unchanged if nothing was drawn (caller
    already handles the "empty board" case separately).
    """
    ys, xs = np.where(gray > threshold)
    if len(ys) == 0 or len(xs) == 0:
        return gray

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    cropped = gray[y0:y1 + 1, x0:x1 + 1]

    h, w = cropped.shape
    side = max(h, w)
    pad = max(1, int(side * margin_frac))
    side += pad * 2

    square = np.zeros((side, side), dtype=gray.dtype)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped
    return square
 
 
def canvas_to_input(image_data):
    gray = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    gray = crop_and_center(gray)
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    return preprocess_array(gray), gray
 
 
def upload_to_input(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img.mean() > 127:  # dark text on light background -> invert
        img = 255 - img
    img = crop_and_center(img)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    return preprocess_array(img), img
 
 
def gray28_to_thumb_b64(img_gray_28):
    """Encode a 28x28 white-on-black array as a small base64 PNG for the
    history gallery (kept tiny so many thumbnails stay cheap to render)."""
    img = Image.fromarray(img_gray_28.astype(np.uint8), mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
 
 
def load_history():
    """Load the persisted homework pile from disk (all-time, not just this
    session). Missing/corrupt files just start a fresh, empty pile."""
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f)


def log_prediction(thumb_gray_28, probs):
    """Append one graded attempt to the homework pile and persist it to
    disk, so it's still there next time the app is opened."""
    top_idx = int(np.argmax(probs))
    entry = {
        "thumb": gray28_to_thumb_b64(thumb_gray_28),
        "char": LABEL_MAP[top_idx],
        "conf": float(probs[top_idx]) * 100,
        "ts": time.time(),
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:MAX_HISTORY]
    save_history(st.session_state.history)
 
 
def render_grade_slip(probs):
    order = np.argsort(probs)[::-1]
    top_char = LABEL_MAP[int(order[0])]
 
    st.markdown(
        f"""
        <div class="grade-slip">
            <div class="circle-wrap">
                <svg viewBox="0 0 100 100" width="150" height="150">
                    <path class="chalk-path" d="M 50,8 C 22,6 6,26 8,50 C 10,76 28,94 54,92
                        C 80,90 95,70 92,44 C 90,18 74,5 50,8 Z"
                        fill="none" stroke="#E8C468" stroke-width="4" stroke-linecap="round" />
                </svg>
                <div class="circle-char">{top_char}</div>
            </div>
            <div style="flex:1;">
        """,
        unsafe_allow_html=True,
    )
 
    colors = ["#E8C468", "#E08577", "#8FB4D1"]
    for rank, idx in enumerate(order[:3]):
        pct = probs[idx] * 100
        st.markdown(
            f"""
            <div class="bar-row">
                <div class="bar-char">{LABEL_MAP[int(idx)]}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{colors[rank % 3]};"></div></div>
                <div class="bar-pct">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)
 
 
def render_history_chart():
    """Standalone tile: a small 'report card' bar chart of average
    confidence per character, built from the full persisted history (all
    sessions, not just this one). Bars are colour-tiered like a grade -
    solid pass / borderline / needs-practice - so it doubles as a quick
    view of which characters the model struggles with most."""
    history = st.session_state.history

    with st.container(key="chart_tile"):
        st.markdown('<div class="tile-label">Report card — avg. confidence by character</div>', unsafe_allow_html=True)

        if len(history) < 3:
            st.markdown(
                '<p class="empty-slip">Grade a few more characters to see the breakdown here.</p>',
                unsafe_allow_html=True,
            )
            return

        conf_sum, count = defaultdict(float), defaultdict(int)
        for item in history:
            conf_sum[item["char"]] += item["conf"]
            count[item["char"]] += 1

        chars = sorted(count.keys())  # '0'-'9' sort before 'A'-'Z'
        avg_conf = [conf_sum[c] / count[c] for c in chars]

        PASS, BORDERLINE, NEEDS_WORK = "#8FB4D1", "#E8C468", "#E08577"
        INK, GRID = "#33312A", "#DCD3B8"
        bar_colors = [
            PASS if v >= 85 else BORDERLINE if v >= 60 else NEEDS_WORK
            for v in avg_conf
        ]

        fig, ax = plt.subplots(figsize=(max(7, len(chars) * 0.36), 2.9))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=(0, (3, 4)), linewidth=0.8, color=GRID)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])

        bars = ax.bar(chars, avg_conf, color=bar_colors, width=0.62, zorder=3)
        for rect, v in zip(bars, avg_conf):
            ax.text(
                rect.get_x() + rect.get_width() / 2, v + 3, f"{v:.0f}",
                ha="center", va="bottom", fontsize=6.5, color=INK,
                family="monospace",
            )

        ax.set_ylabel("Avg. confidence %", color=INK, fontsize=8, family="monospace")
        ax.tick_params(colors=INK, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_family("monospace")
        for label in ax.get_yticklabels():
            label.set_family("monospace")
            label.set_fontsize(7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.markdown(
            f"""
            <div class="legend-row">
                <span><i style="background:{PASS};"></i>85%+ solid</span>
                <span><i style="background:{BORDERLINE};"></i>60–84% borderline</span>
                <span><i style="background:{NEEDS_WORK};"></i>&lt;60% needs practice</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history_pile():
    with st.container(key="history_tile"):
        top_row = st.columns([4, 1])
        with top_row[0]:
            st.markdown('<div class="tile-label">Homework pile (all-time)</div>', unsafe_allow_html=True)
        with top_row[1]:
            if st.session_state.history:
                if st.button("Clear", key="clear_history_btn"):
                    st.session_state.history = []
                    save_history([])
                    st.rerun()
 
        if not st.session_state.history:
            st.markdown(
                '<p class="empty-slip">Nothing graded yet — every character you draw or upload '
                "will land here.</p>",
                unsafe_allow_html=True,
            )
            return
 
        st.markdown(
            f'<div class="pile-count">{len(st.session_state.history)} graded</div>',
            unsafe_allow_html=True,
        )

        # Rendered with st.image() + st.columns() rather than one big
        # hand-built HTML string of many base64 <img> tags: with enough
        # entries that combined string occasionally got truncated or
        # mis-escaped by Streamlit's markdown sanitizer, showing raw
        # "<div..." text instead of the thumbnail. Native widgets don't
        # have that failure mode.
        cols_per_row = 8
        history = st.session_state.history
        for row_start in range(0, len(history), cols_per_row):
            row_items = history[row_start:row_start + cols_per_row]
            row_cols = st.columns(cols_per_row)
            for col, item in zip(row_cols, row_items):
                with col:
                    st.image(base64.b64decode(item["thumb"]), width=64)
                    st.markdown(
                        f'<div class="pile-char">{item["char"]}</div>'
                        f'<div class="pile-conf">{item["conf"]:.0f}%</div>',
                        unsafe_allow_html=True,
                    )
 
 
model = load_char_model()
 
# Persist the last prediction across reruns. streamlit-drawable-canvas
# syncs its image_data to Python slightly after the click that triggers a
# rerun (its own component value update fires a second, separate rerun) -
# without storing the result in session_state, that follow-up rerun would
# see predict_clicked=False and blank out the answer we just computed.
if "last_probs" not in st.session_state:
    st.session_state.last_probs = None
if "board_was_empty" not in st.session_state:
    st.session_state.board_was_empty = False
if "history" not in st.session_state:
    # Loaded once per browser session from disk, so every attempt ever
    # graded - across restarts and different sessions - shows up here,
    # not just what was drawn since this page load.
    st.session_state.history = load_history()
if "last_logged_id" not in st.session_state:
    # Guards against re-logging the same prediction on the follow-up
    # rerun that streamlit-drawable-canvas triggers after syncing its
    # image_data (see note above) - without this a single "Grade it"
    # click would add two identical entries to the pile.
    st.session_state.last_logged_id = None
if "canvas_key" not in st.session_state:
    # Bumping this forces streamlit-drawable-canvas to remount with a
    # blank board. Without it, strokes from the previous character stay
    # on screen and the next "Grade it" click predicts on a mix of two
    # overlapping characters instead of a fresh one.
    st.session_state.canvas_key = 0
 
col_draw, col_result = st.columns([1, 1], gap="large")
 
with col_draw:
    with st.container(key="draw_tile"):
        if CANVAS_AVAILABLE:
            st.markdown('<div class="tile-label">Draw here</div>', unsafe_allow_html=True)
            canvas_result = st_canvas(
                fill_color="#000000",
                stroke_width=16,
                stroke_color="#F2EFE4",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state.canvas_key}",
                return_image_data=True,
            )
            predict_clicked = st.button("Grade it")
 
            with st.expander("Or upload an image instead"):
                uploaded = st.file_uploader(
                    "Character image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
                )
        else:
            # streamlit-drawable-canvas didn't install (e.g. a pip/network
            # issue in this environment) - fall back to upload-only mode so
            # the app still fully works instead of crashing.
            canvas_result = None
            predict_clicked = False
            st.markdown(
                '<div class="tile-label">Upload here</div>'
                '<p class="empty-slip">Drawing canvas isn\'t installed '
                '(<code>pip install streamlit-drawable-canvas</code> failed) — '
                "upload an image instead.</p>",
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Character image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
            )
 
with col_result:
    with st.container(key="result_tile"):
        st.markdown('<div class="tile-label">Result</div>', unsafe_allow_html=True)
 
        if model is None:
            st.markdown(
                '<p class="empty-slip">No trained model found yet.<br>'
                "Run <code>python src/train.py</code> first, then reload this page.</p>",
                unsafe_allow_html=True,
            )
        else:
            if predict_clicked and canvas_result.image_data is not None:
                if canvas_result.image_data[..., :3].sum() < 1e-3:
                    st.session_state.last_probs = None
                    st.session_state.board_was_empty = True
                else:
                    model_input, gray28 = canvas_to_input(canvas_result.image_data)
                    probs = model.predict(model_input, verbose=0)[0]
                    st.session_state.last_probs = probs
                    st.session_state.board_was_empty = False
 
                    # Log once per click, not once per rerun.
                    log_id = f"canvas-{hash(canvas_result.image_data.tobytes())}"
                    if st.session_state.last_logged_id != log_id:
                        log_prediction(gray28, probs)
                        st.session_state.last_logged_id = log_id
                        # Reset the board for the next character and
                        # rerun once so the fresh (blank) canvas shows
                        # immediately, while the result/history above
                        # stay intact via session_state.
                        st.session_state.canvas_key += 1
                        st.rerun()
            elif uploaded is not None:
                log_id = f"upload-{uploaded.file_id}"
                if st.session_state.last_logged_id != log_id:
                    model_input, gray28 = upload_to_input(uploaded.getvalue())
                    probs = model.predict(model_input, verbose=0)[0]
                    st.session_state.last_probs = probs
                    st.session_state.board_was_empty = False
                    log_prediction(gray28, probs)
                    st.session_state.last_logged_id = log_id
 
            if st.session_state.last_probs is not None:
                render_grade_slip(st.session_state.last_probs)
            elif st.session_state.board_was_empty:
                st.markdown(
                    '<p class="empty-slip">The board is empty — draw a character first.</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="empty-slip">Draw a character and press "Grade it", or upload an image.</p>',
                    unsafe_allow_html=True,
                )
 
render_history_chart()
render_history_pile()
 
st.markdown(
    """
    <div class="foot-rail">
        CodeAlpha Machine Learning Internship — Task 3 of 4 — Handwritten Character Recognition
    </div>
    """,
    unsafe_allow_html=True,
)
