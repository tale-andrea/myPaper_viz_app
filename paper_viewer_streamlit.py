"""
Paper Viewer — Streamlit version
Run with:  streamlit run paper_viewer_streamlit.py

Install deps:  pip install streamlit PyMuPDF Pillow requests
"""

import io
import json
import re
import urllib.parse
import urllib.request

import fitz  # PyMuPDF
import requests
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Viewer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Overall background */
    .stApp { background-color: #f8f9fa; }

    /* Left panel card */
    .meta-card {
        background: white;
        border-radius: 10px;
        padding: 20px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        height: 100%;
    }
    .paper-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.4;
        margin-bottom: 14px;
    }
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin-bottom: 6px;
    }
    .abstract-text {
        font-size: 0.88rem;
        color: #444;
        line-height: 1.6;
        max-height: 420px;
        overflow-y: auto;
    }
    .divider { border-top: 1px solid #eee; margin: 14px 0; }

    /* Figure panel */
    .fig-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .page-counter {
        font-size: 0.82rem;
        color: #888;
        text-align: center;
        margin-top: 8px;
    }

    /* Hide default Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Tighter column gap */
    [data-testid="column"] { padding: 0 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def render_pdf_pages(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """Render every PDF page to a PNG bytes object."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages


@st.cache_data(show_spinner=False)
def extract_text_from_bytes(pdf_bytes: bytes, max_chars: int = 5000) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for i, page in enumerate(doc):
        if i >= 5:
            break
        text += page.get_text()
    doc.close()
    return text[:max_chars]


def guess_title_abstract(text: str) -> tuple[str, str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0] if lines else "Unknown Title"

    abstract = ""
    idx = text.lower().find("abstract")
    if idx != -1:
        chunk = text[idx + 8: idx + 2000].strip()
        for kw in ["introduction", "keywords", "1.", "background", "highlights"]:
            ki = chunk.lower().find(kw)
            if ki > 80:
                chunk = chunk[:ki]
        abstract = chunk.strip()

    if not abstract and len(lines) > 1:
        abstract = " ".join(lines[1:25])

    return title, abstract


@st.cache_data(show_spinner=False)
def fetch_doi(doi: str) -> tuple[str, str]:
    doi = doi.strip().lstrip("https://doi.org/").lstrip("http://dx.doi.org/")
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "PaperViewer/1.0"})
        r.raise_for_status()
        msg = r.json().get("message", {})
        titles = msg.get("title", ["Unknown Title"])
        title = titles[0] if titles else "Unknown Title"
        abstract = msg.get("abstract", "Abstract not available via CrossRef.")
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        return title, abstract
    except Exception as e:
        return "Error fetching DOI", str(e)


# ─────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────
defaults = {
    "pages": [],        # list of PNG bytes
    "page_idx": 0,
    "title": "",
    "abstract": "",
    "loaded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────
st.markdown("## 📄 Paper Viewer")

col_doi, col_btn, col_or, col_upload = st.columns([3, 0.8, 0.3, 2])

with col_doi:
    doi_input = st.text_input("DOI", placeholder="e.g. 10.1016/j.neuron.2024.01.001",
                               label_visibility="collapsed")
with col_btn:
    fetch_btn = st.button("Fetch DOI", use_container_width=True, type="primary")

with col_or:
    st.markdown("<div style='padding-top:8px;text-align:center;color:#aaa'>or</div>",
                unsafe_allow_html=True)

with col_upload:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"],
                                      label_visibility="collapsed")

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
if fetch_btn and doi_input.strip():
    with st.spinner("Fetching metadata from CrossRef…"):
        title, abstract = fetch_doi(doi_input.strip())
    st.session_state.title = title
    st.session_state.abstract = abstract
    st.session_state.pages = []
    st.session_state.page_idx = 0
    st.session_state.loaded = True

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    if not st.session_state.loaded or st.session_state.pages == []:
        with st.spinner("Rendering PDF pages at high resolution…"):
            pages = render_pdf_pages(pdf_bytes, dpi=200)
            text = extract_text_from_bytes(pdf_bytes)
            title, abstract = guess_title_abstract(text)
        st.session_state.pages = pages
        st.session_state.page_idx = 0
        # Only overwrite metadata if DOI wasn't already fetched
        if not st.session_state.title:
            st.session_state.title = title
            st.session_state.abstract = abstract
        st.session_state.loaded = True


# ─────────────────────────────────────────────────────────────
# Main layout  ← left panel | right figure panel →
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2.4], gap="medium")

# ── Left: metadata ────────────────────────────────────────────
with left_col:
    st.markdown('<div class="meta-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Title</div>', unsafe_allow_html=True)
    title_display = st.session_state.title or "—"
    st.markdown(f'<div class="paper-title">{title_display}</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Abstract</div>', unsafe_allow_html=True)
    abstract_display = st.session_state.abstract or "—"
    st.markdown(f'<div class="abstract-text">{abstract_display}</div>',
                unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Right: figure / page viewer ───────────────────────────────
with right_col:
    pages = st.session_state.pages

    if not pages:
        st.markdown("""
        <div class="fig-card" style="text-align:center;padding:80px 20px;color:#bbb;">
            <div style="font-size:3rem">📑</div>
            <div style="font-size:1rem;margin-top:12px;">
                Upload a PDF to browse figures as high-res screenshots
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        idx = st.session_state.page_idx
        total = len(pages)

        # Navigation buttons
        nav_left, nav_center, nav_right = st.columns([1, 3, 1])
        with nav_left:
            if st.button("◀  Prev", use_container_width=True,
                         disabled=(idx == 0)):
                st.session_state.page_idx -= 1
                st.rerun()
        with nav_center:
            st.markdown(
                f'<div class="page-counter">Page {idx + 1} of {total}</div>',
                unsafe_allow_html=True,
            )
        with nav_right:
            if st.button("Next  ▶", use_container_width=True,
                         disabled=(idx == total - 1), type="primary"):
                st.session_state.page_idx += 1
                st.rerun()

        # Page image
        st.markdown('<div class="fig-card">', unsafe_allow_html=True)
        img = Image.open(io.BytesIO(pages[idx]))
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Thumbnail strip (up to 8 pages)
        if total > 1:
            st.markdown("**Pages**")
            thumb_cols = st.columns(min(total, 8))
            for i, col in enumerate(thumb_cols):
                with col:
                    thumb = Image.open(io.BytesIO(pages[i]))
                    thumb.thumbnail((120, 160))
                    st.image(thumb, use_container_width=True)
                    if st.button(f"{i+1}", key=f"thumb_{i}",
                                 use_container_width=True,
                                 type="primary" if i == idx else "secondary"):
                        st.session_state.page_idx = i
                        st.rerun()
