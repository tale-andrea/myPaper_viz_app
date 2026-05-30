"""
Paper Viewer — Streamlit version with auto figure detection
Run with:  streamlit run paper_viewer_streamlit.py
Install:   pip install streamlit PyMuPDF Pillow requests
"""

import io
import json
import re
import urllib.parse
import urllib.request
import zipfile

import fitz  # PyMuPDF
import requests
import streamlit as st
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Viewer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .meta-card {
        background: white; border-radius: 10px;
        padding: 20px 22px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .paper-title {
        font-size: 1.05rem; font-weight: 700; color: #1a1a2e;
        line-height: 1.4; margin-bottom: 14px;
    }
    .section-label {
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: #888; margin-bottom: 6px;
    }
    .abstract-text {
        font-size: 0.88rem; color: #444; line-height: 1.6;
        max-height: 340px; overflow-y: auto;
    }
    .divider { border-top: 1px solid #eee; margin: 14px 0; }
    .fig-card {
        background: white; border-radius: 10px;
        padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .fig-badge {
        display:inline-block; background:#e8f4fd; color:#1a6fa8;
        border-radius:6px; padding:2px 10px; font-size:0.8rem;
        font-weight:600; margin-bottom:8px;
    }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="column"] { padding: 0 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Figure detection logic
# ─────────────────────────────────────────────────────────────

def detect_figures(page: fitz.Page, dpi: int = 200) -> list[dict]:
    """
    Detect figure regions on a page using three strategies:
    1. Embedded raster/vector images
    2. Large vector drawing clusters (matplotlib-style figures)
    3. Caption anchors (lines starting with Fig. / Figure)
    Returns list of dicts with keys: bbox (fitz.Rect), label, source
    """
    zoom = dpi / 72
    page_rect = page.rect
    figures = []

    # ── Strategy 1: embedded images ──────────────────────────
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        rects = page.get_image_rects(xref)
        for r in rects:
            if r.width > 40 and r.height > 40:
                figures.append({"bbox": r, "label": "Embedded image", "source": "image"})

    # ── Strategy 2: vector drawing clusters ──────────────────
    drawings = page.get_drawings()
    if drawings:
        # Collect all drawing rects and union nearby ones
        rects = [fitz.Rect(d["rect"]) for d in drawings if d["rect"].width > 5 and d["rect"].height > 5]
        if rects:
            # Cluster by proximity (merge rects within 20pt of each other)
            clusters = []
            used = [False] * len(rects)
            for i, r in enumerate(rects):
                if used[i]:
                    continue
                cluster = fitz.Rect(r)
                used[i] = True
                changed = True
                while changed:
                    changed = False
                    for j, r2 in enumerate(rects):
                        if used[j]:
                            continue
                        expanded = fitz.Rect(cluster.x0 - 20, cluster.y0 - 20,
                                             cluster.x1 + 20, cluster.y1 + 20)
                        if expanded.intersects(r2):
                            cluster = cluster | r2
                            used[j] = True
                            changed = True
                clusters.append(cluster)

            # Keep only clusters large enough to be figures (>15% page area or >100x100pt)
            page_area = page_rect.width * page_rect.height
            for c in clusters:
                area = c.width * c.height
                if area > page_area * 0.03 and c.width > 80 and c.height > 80:
                    # Avoid duplicating already-found embedded images
                    overlap = any(
                        fitz.Rect(f["bbox"]).intersects(c) for f in figures
                    )
                    if not overlap:
                        figures.append({"bbox": c, "label": "Vector figure", "source": "drawing"})

    # ── Strategy 3: caption anchors (Fig. X / Figure X) ──────
    blocks = page.get_text("dict")["blocks"]
    caption_ys = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = " ".join(s["text"] for s in line.get("spans", []))
            if re.match(r"^\s*(Fig\.?|Figure)\s*\d", line_text, re.IGNORECASE):
                caption_ys.append((block["bbox"][1], block["bbox"]))  # top y, bbox

    for cap_y, cap_bbox in caption_ys:
        # Look for figure content ABOVE the caption (within 300pt)
        search_rect = fitz.Rect(
            cap_bbox[0] - 10,
            max(0, cap_y - 320),
            cap_bbox[2] + 10,
            cap_y + 2
        )
        # Check if any known figure overlaps this region
        already_covered = any(
            fitz.Rect(f["bbox"]).intersects(search_rect) for f in figures
        )
        if not already_covered and search_rect.height > 40:
            figures.append({
                "bbox": search_rect,
                "label": "Caption-anchored figure",
                "source": "caption"
            })

    # ── Deduplicate heavily overlapping regions ───────────────
    filtered = []
    for i, f in enumerate(figures):
        r = fitz.Rect(f["bbox"])
        dominated = False
        for j, g in enumerate(figures):
            if i == j:
                continue
            r2 = fitz.Rect(g["bbox"])
            inter = r & r2
            if inter.is_empty:
                continue
            inter_area = inter.width * inter.height
            r_area = r.width * r.height
            if r_area > 0 and inter_area / r_area > 0.85 and r2.width * r2.height > r_area:
                dominated = True
                break
        if not dominated:
            filtered.append(f)

    return filtered


@st.cache_data(show_spinner=False)
def render_pdf_pages(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
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
def extract_all_figures(pdf_bytes: bytes, dpi: int = 200) -> list[dict]:
    """
    Run figure detection on all pages, return cropped figure images as PNG bytes.
    Each entry: {page_num, label, source, image_bytes}
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    results = []
    fig_counter = 1

    for page_num, page in enumerate(doc):
        figures = detect_figures(page, dpi=dpi)
        for fig in figures:
            bbox = fitz.Rect(fig["bbox"])
            # Add padding
            padded = fitz.Rect(
                max(0, bbox.x0 - 8),
                max(0, bbox.y0 - 8),
                min(page.rect.width,  bbox.x1 + 8),
                min(page.rect.height, bbox.y1 + 8),
            )
            clip_mat = mat
            pix = page.get_pixmap(matrix=clip_mat, clip=padded)
            img_bytes = pix.tobytes("png")

            results.append({
                "fig_num": fig_counter,
                "page_num": page_num + 1,
                "label": fig["label"],
                "source": fig["source"],
                "image_bytes": img_bytes,
                "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
            })
            fig_counter += 1

    doc.close()
    return results


@st.cache_data(show_spinner=False)
def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for i, page in enumerate(doc):
        if i >= 5:
            break
        text += page.get_text()
    doc.close()
    return text[:5000]


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
        r = requests.get(url, timeout=10, headers={"User-Agent": "PaperViewer/1.0"})
        r.raise_for_status()
        msg = r.json().get("message", {})
        titles = msg.get("title", ["Unknown Title"])
        title = titles[0] if titles else "Unknown Title"
        abstract = msg.get("abstract", "Abstract not available via CrossRef.")
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        return title, abstract
    except Exception as e:
        return "Error fetching DOI", str(e)


def make_zip(figures: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in figures:
            fname = f"figure_{f['fig_num']:02d}_page{f['page_num']}.png"
            zf.writestr(fname, f["image_bytes"])
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
for k, v in {
    "pages": [], "page_idx": 0,
    "title": "", "abstract": "",
    "figures": [], "loaded": False,
    "active_tab": "pages",
}.items():
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
    st.session_state.figures = []
    st.session_state.page_idx = 0
    st.session_state.loaded = True

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    file_key = uploaded_file.name + str(len(pdf_bytes))
    if st.session_state.get("_last_file") != file_key:
        st.session_state["_last_file"] = file_key
        with st.spinner("Rendering pages and detecting figures…"):
            pages = render_pdf_pages(pdf_bytes, dpi=200)
            figures = extract_all_figures(pdf_bytes, dpi=200)
            text = extract_text_from_bytes(pdf_bytes)
            title, abstract = guess_title_abstract(text)
        st.session_state.pages = pages
        st.session_state.figures = figures
        st.session_state.page_idx = 0
        if not st.session_state.title:
            st.session_state.title = title
            st.session_state.abstract = abstract
        st.session_state.loaded = True


# ─────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2.4], gap="medium")

# ── Left: metadata ────────────────────────────────────────────
with left_col:
    st.markdown('<div class="meta-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Title</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="paper-title">{st.session_state.title or "—"}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Abstract</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="abstract-text">{st.session_state.abstract or "—"}</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Right: tabs (Pages | Figures) ────────────────────────────
with right_col:
    pages   = st.session_state.pages
    figures = st.session_state.figures

    tab_pages, tab_figs = st.tabs([
        f"📑 Pages ({len(pages)})",
        f"🖼️ Detected Figures ({len(figures)})",
    ])

    # ── TAB 1: full page viewer ───────────────────────────────
    with tab_pages:
        if not pages:
            st.markdown("""
            <div class="fig-card" style="text-align:center;padding:80px 20px;color:#bbb;">
                <div style="font-size:3rem">📑</div>
                <div style="margin-top:12px;">Upload a PDF to browse pages</div>
            </div>""", unsafe_allow_html=True)
        else:
            idx   = st.session_state.page_idx
            total = len(pages)
            n1, n2, n3 = st.columns([1, 3, 1])
            with n1:
                if st.button("◀  Prev", use_container_width=True, disabled=(idx == 0)):
                    st.session_state.page_idx -= 1
                    st.rerun()
            with n2:
                st.markdown(f"<div style='text-align:center;color:#888;padding-top:8px'>"
                            f"Page {idx+1} of {total}</div>", unsafe_allow_html=True)
            with n3:
                if st.button("Next  ▶", use_container_width=True,
                             disabled=(idx == total-1), type="primary"):
                    st.session_state.page_idx += 1
                    st.rerun()

            st.markdown('<div class="fig-card">', unsafe_allow_html=True)
            st.image(Image.open(io.BytesIO(pages[idx])), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Thumbnail strip
            if total > 1:
                st.markdown("**Jump to page:**")
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

    # ── TAB 2: detected figures ───────────────────────────────
    with tab_figs:
        if not figures:
            if not pages:
                st.markdown("""
                <div class="fig-card" style="text-align:center;padding:80px 20px;color:#bbb;">
                    <div style="font-size:3rem">🖼️</div>
                    <div style="margin-top:12px;">Upload a PDF to auto-detect figures</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("No figures were automatically detected in this PDF.")
        else:
            # Download all button
            zip_bytes = make_zip(figures)
            st.download_button(
                label=f"⬇️ Download all {len(figures)} figures as ZIP",
                data=zip_bytes,
                file_name="figures.zip",
                mime="application/zip",
                type="primary",
            )

            st.markdown(f"**{len(figures)} figures detected** across {len(pages)} pages")
            st.markdown("---")

            # Show figures in a 2-column grid
            source_icons = {
                "image":   "🖼️ Embedded image",
                "drawing": "📊 Vector figure",
                "caption": "📝 Caption-anchored",
            }

            cols = st.columns(2)
            for i, fig in enumerate(figures):
                with cols[i % 2]:
                    icon_label = source_icons.get(fig["source"], fig["label"])
                    st.markdown(
                        f'<div class="fig-badge">{icon_label} — page {fig["page_num"]}</div>',
                        unsafe_allow_html=True,
                    )
                    img = Image.open(io.BytesIO(fig["image_bytes"]))
                    st.image(img, use_container_width=True)
                    st.download_button(
                        label="⬇️ Download",
                        data=fig["image_bytes"],
                        file_name=f"figure_{fig['fig_num']:02d}_page{fig['page_num']}.png",
                        mime="image/png",
                        key=f"dl_{i}",
                    )
                    st.markdown("---")
