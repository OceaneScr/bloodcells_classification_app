import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px

from utils.config import vspace
from utils.config import (
    COMMON_CSS, PALETTE, UMAP_CSV,
)

st.set_page_config(page_title="Preprocessing",
                   page_icon="⚙️", layout="wide")

st.markdown(COMMON_CSS, unsafe_allow_html=True)

@st.cache_data
def load_umap(path):
    return pd.read_csv(path, sep=';')


st.title("⚙️ Preprocessing")

vspace(40)

tab1,tab2 = st.tabs([
    "ML Pipeline ",
    "DL Pipeline",
])


with tab1:
    st.subheader("Feature extraction")

    st.markdown("The machine learning pipeline transforms each image into a **99-dimensional feature vector**.")

    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin:20px 0;">
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:20px;">🔬</div>
            <div style="font-weight:600; margin:4px 0;">RGB Images</div>
            <div style="font-size:12px; color:#888;">360 × 363</div>
        </div>
        <div style="font-size:20px; color:#ccc;">→</div>
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:20px;">✂️</div>
            <div style="font-weight:600; margin:4px 0;">Resize + Crop</div>
            <div style="font-size:12px; color:#888;">90 × 90</div>
        </div>
        <div style="font-size:20px; color:#ccc;">→</div>
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:20px;">🎨</div>
            <div style="font-weight:600; margin:4px 0;">RGB → L*a*b*</div>
            <div style="font-size:12px; color:#888;">conversion</div>
        </div>
        <div style="font-size:20px; color:#ccc;">→</div>
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:15px;">🟠</div>
            <div style="font-weight:600; margin:4px 0;">K-Means</div>
            <div style="font-size:12px; color:#888;">k = 8 clusters</div>
        </div>
        <div style="font-size:20px; color:#ccc;">→</div>
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:20px;">📐</div>
            <div style="font-weight:600; margin:4px 0;">Extraction</div>
            <div style="font-size:12px; color:#888;">99 features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Feature description")


    df = pd.DataFrame({
            "Catégorie": [
                "K-Means",
                "K-Means",
                "K-Means",
                "K-Means",
                "Global",
                "Texture",
                "Texture",
            ],
            "Feature"   : [
                "Cluster histograms",
                "Intra-cluster means",
                "Intra-cluster standard deviations",
                "Radial histograms",
                "Global statistics",
                "GLCM",
                "Sobel",
            ],
            "Description": [
                "Overall distribution across the 8 clusters",
                "Mean per cluster on L*, a* et b*",
                "Standard deviation per cluster on L*, a* et b*",
                "4 concentric rings × 8 clusters",
                "Mean and standard deviation on L*, a* et b*",
                "Contrast, homogeneity and energy",
                "Mean and standard deviation of the gradient magnitude ",
            ],
        "Dimension":[8, 24, 24, 32, 6, 3, 2]
    })

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
    st.metric("Total dimension", "99")


    # with col2:
    #     img = Image.open(f"./images/figures/kmeans_radii.png").convert("RGB")
    #     st.image(img, width="stretch")



    #st.caption("Dimension initiale : 90 × 90 × 3 = 24 300, rendant la classification directe difficile sur CPU.")

    vspace(10)
    with st.expander("See a K-Means segmentation example by cell type"):
        classes = [
            ("basophile", "Basophil"),
            ("eosinophile", "Eosinophil"),
            ("erythroblaste", "Erythroblast"),
            ("ig", "IG"),
            ("lymphocyte", "Lymphocyte"),
            ("monocyte", "Monocyte"),
            ("neutrophile", "Neutrophil"),
            ("plaquette", "Platelet"),
        ]
        
        row1 = st.columns(4)
        row2 = st.columns(4)
        
        for i, (filename, label) in enumerate(classes):
            col = row1[i] if i < 4 else row2[i - 4]
            with col:
                img = Image.open(f"images/kmeans/kmeans_{filename}.png")
                st.image(img, width="stretch")
                st.markdown(f"<p style='text-align:center; font-size:0.875rem; color:grey; margin-top:-1.5rem'>{label}</p>", unsafe_allow_html=True)

        st.caption("The colors correspond to the colors of the cluster centers, converted to RGB.")
    st.divider()


    st.subheader("UMAP — Feature projection")

    st.write("UMAP projection of the dataset images based on the 99 features extracted for each of them. Each point represents an image, colored according to its cell class.")

    df_umap = load_umap(UMAP_CSV)
    classes = sorted(df_umap["label"].unique())

    def reset():
        st.session_state.selected_classes = classes
        st.session_state.point_size = 2.5
        st.session_state.opacity = 0.5
        #st.rerun()

    if "selected_classes" not in st.session_state:
        st.session_state.selected_classes = classes

    if "point_size" not in st.session_state:
        st.session_state.point_size = 2.5

    if "opacity" not in st.session_state:
        st.session_state.opacity = 0.5

    before ,left, middle, right, after = st.columns([0.05, 2.,0.3, 0.8, 0.2], vertical_alignment="top")

    with right:
        
        vspace(40)
        selected_classes = st.multiselect(
            "Classes",
            classes,
            key="selected_classes",
        # label_visibility="collapsed"
        )

        point_size = st.slider(
            min_value=0.5,
            max_value=5.0,
            value=2.5,
            step=0.5,
            key="point_size",
            label="Point size"
        )

        opacity = st.slider(
            "Opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="opacity",
                    )

        st.button("Reset", on_click=reset)

    df_plot = df_umap[
        df_umap["label"].isin(selected_classes)
    ]


    with left:
        
        fig_umap = px.scatter(
            df_plot,
            x="umap_1",
            y="umap_2",
            color="label",
            color_discrete_map=PALETTE,
            hover_name="label",
            labels={
                "umap_1":"UMAP dim 1",
                "umap_2":"UMAP dim 2",
                "label":"Class"
            },
           # title=f"{len(df_plot):,} cellules · {len(selected_classes)} classes",
            opacity=opacity,
            render_mode="svg"
        )

        fig_umap.update_traces(
            marker=dict(size=point_size)
        )

        fig_umap.update_layout(
            height=550,
            legend_title="Cellular Class",
            legend_title_font=dict(size=15),
            plot_bgcolor="white",
            #title_font=dict(size=20, color="#1a1a1a"),
            xaxis=dict(showgrid=True, gridcolor="#f2f2f2", gridwidth=0.25),
            yaxis=dict(showgrid=True, gridcolor="#f2f2f2", gridwidth=0.25),
            legend=dict(orientation="h", y=-0.2, font=dict(size=14), itemsizing="constant")
        )

        st.plotly_chart(fig_umap)

    vspace(20)

    st.caption("This projection shows that the extracted features are overall discriminative, with well-separated clusters for most classes. However, some classes show partial overlap, corresponding to the most morphologically similar cell categories.")
    
with tab2:

    st.subheader("Image format")

    st.markdown("""
    <div style="display:flex; gap:10px; align-items:stretch; margin-bottom:1rem;">
        <div class="metric-card"><div class="metric-label">Resolution</div><div class="metric-value">360 × 360</div></div>
        <div class="metric-card"><div class="metric-label">Color space</div><div class="metric-value">RGB</div></div>
        <div class="metric-card"><div class="metric-label">Type</div><div class="metric-value">Float32</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Data augmentation")
    st.caption("Augmentation of the training set images to improve the models' generalization ability.")
    st.dataframe(
        pd.DataFrame({
            "Transformation": ["RandomFlip", "RandomRotation"],
            "Parameter"     : ["Horizontal + Vertical", "factor = 0.2 (± 72°)"],
        }),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    st.subheader("Image normalization")
    st.caption("Image normalization specific to each architecture, matching the one applied during pretraining on ImageNet.")
    st.dataframe(
        pd.DataFrame({
            "Architecture" : ["EfficientNetV2S/M", "ResNet50V2", "DenseNet121", "VGG19", "Xception"],
            "Normalization":  ["Scaling to [-1, 1]", "Subtraction of the per-channel ImageNet mean (RGB)",
                                "Scaling to [0, 1] then centering using the ImageNet mean and standard deviation",
                                "Subtraction of the pixel-wise ImageNet mean (BGR)", "Scaling to [-1, 1]"],
        }),
        use_container_width=True, hide_index=True,
    )



