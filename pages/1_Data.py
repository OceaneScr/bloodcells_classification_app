import streamlit as st
import pandas as pd
from glob import glob
from PIL import Image
import random
import plotly.express as px
from utils.config import vspace
from utils.config import (COMMON_CSS, SAMPLE_DIR, CLASSES_EN, IMG_SIZE_DL, QUALITY_IMAGES)
 
st.set_page_config(page_title="Data", page_icon="🔬", layout="wide")
st.markdown(COMMON_CSS, unsafe_allow_html=True)
 
 
 
def distribution_show():
    df = pd.read_csv("files/cells_distribution.csv")
    fig = px.bar(
        df,
        x="proportion",
        y="category",
        orientation="h",
        text="proportion",
        color="category",
        color_discrete_sequence=["#8b8fa8"]
    )
    fig.update_traces(texttemplate=" %{text:.1f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_title="",
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Proportion (%)"
    )
    fig.update_xaxes(range=[0, 22])
    return fig
 
 
def get_images_of_class(cls, n=5):
    paths = glob(f"{SAMPLE_DIR}/{cls}/*")
    imgs = []
    for path in random.sample(paths, min(n, len(paths))):
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE_DL)
        imgs.append(img)
    return imgs
 
 
# ── Titre ────────────────────────────────────────────────────────────────────
 
st.title("🔬 Data")
st.divider()
 
# ── Dataset ──────────────────────────────────────────────────────────────────
 
st.subheader("PBC Dataset — Peripheral Blood Cells")
st.write("""
The PBC (Peripheral Blood Cells) dataset is a public hematology microscopy
dataset composed of images of peripheral blood cells acquired
with the CellaVision DM96 at the Hospital Clinic of Barcelona laboratory.
""")
 
vspace(20)

st.markdown("""
<div style="display:flex; gap:10px; align-items:stretch; margin-bottom:1rem;">
    <div class="metric-card"><div class="metric-label">Images</div><div class="metric-value">17 092</div></div>
    <div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">8</div></div>
    <div class="metric-card"><div class="metric-label">Resolution</div><div class="metric-value">360×363</div></div>
    <div class="metric-card"><div class="metric-label">Format</div><div class="metric-value">JPG</div></div>
</div>
""", unsafe_allow_html=True)

vspace(10)
st.divider()
 
# ── Classes cellulaires ───────────────────────────────────────────────────────
 
st.subheader("Cellular classes")
st.write(
    "This dataset groups **8 types of blood cells** observed on microscopic smears. "
    "Each category has specific morphological characteristics that models must learn to distinguish."
)
 
vspace(30)
col1, col2 = st.columns([1.2, 2])
 
with col1:
    vspace(10)
    classes = [
        "Basophil", "Eosinophil",
        "Erythroblast", "IG",
        "Lymphocyte", "Monocyte",
        "Neutrophil", "Platelet"
    ]
 
    for i in range(0, len(classes), 2):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="class-card">{classes[i]}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="class-card">{classes[i+1]}</div>', unsafe_allow_html=True)
 
    st.caption("IG : Immature Granulocyte")
 
with col2:
    fig = distribution_show()
    st.plotly_chart(fig, use_container_width=True)
 
st.divider()
 
# ── Exemples d'images ─────────────────────────────────────────────────────────
 
st.subheader("Image examples")
 
selected = st.selectbox("Category", CLASSES_EN)
imgs = get_images_of_class(selected, 5)
 
if st.button("🎲 Show more examples"):
    imgs = get_images_of_class(selected, 5)
 
cols = st.columns(5)
for col, img in zip(cols, imgs):
    with col:
        st.image(img)
 
st.divider()
 
# ── Qualité ───────────────────────────────────────────────────────────────────
 
st.subheader("Image quality and limitations")
st.write("""
The images in the PBC dataset generally present good quality :
- Sharp images
- Homogeneous contrast
- Centered cells
 
However, few images contain multiple nuclei or lack the target cell.
It should also be noted that a small proportion of images present a pink background (2.6 %) .
""")
  
vspace(20)
cols = st.columns(5)
for i, (key, path) in enumerate(QUALITY_IMAGES.items()):
    with cols[i+1]:
        st.image(path)
        st.caption(key) 
    i += 1

st.divider()
 
# ── Source ────────────────────────────────────────────────────────────────────
 
st.markdown("""
<div style="padding:18px; border-radius:8px; background:#f7f7f5; font-size:15px;">
    📄 <b>Dataset Source</b><br><br>
    PBC (Peripheral Blood Cells) Dataset — Images acquired with the <b>CellaVision DM96</b>
    at the Hospital Clinic of Barcelona.<br><br>
    🔗 <a href="https://data.mendeley.com/datasets/snkd93bnjr/1" target="_blank">
    PBC Dataset - Mendeley Data</a>
</div>
""", unsafe_allow_html=True)
