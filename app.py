# app.py

import streamlit as st
import torch
import numpy as np
import pandas as pd
import string
import os
import tempfile
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline

from model import get_model
from dataset import GlyphDataset
from latent import LatentManifold

# basic configurations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# default resolution
RESOLUTION = (56, 56)
# Maxwell grid resolution / density of clickable points
N = 20 

PRESETS = {
    "High α, low β, low γ": {
        "vae": "checkpoints/mg-vae_lowercase_abc.pth",
        "cls": "checkpoints/cls_lowercase_56px.pth",
        "dataset": "data/lowercase.zip"
    },
    "Lowercase": {
        "vae": "checkpoints/mg-vae_lowercase_56px.pth",
        "cls": "checkpoints/cls_lowercase_56px.pth",
        "dataset": "data/lowercase.zip"
    },
    "Uppercase": {
        "vae": "checkpoints/mg-vae_uppercase_56px.pth",
        "cls": "checkpoints/cls_uppercase_56px.pth",
        "dataset": "data/uppercase.zip"
    },
    "All letters": {
        "vae": "checkpoints/mg-vae_letters_56px.pth",
        "cls": "checkpoints/cls_letters_56px.pth",
        "dataset": "data/letters.zip"
    },
    "Numbers": {
        "vae": "checkpoints/mg-vae_numbers_56px.pth",
        "cls": "checkpoints/cls_numbers_56px.pth",
        "dataset": "data/numbers.zip"
    },
    # "Base glyphs": {
    #     "vae": "checkpoints/mg-vae_base_glyphs_56px.pth",
    #     "cls": "checkpoints/cls_glyphs_56px.pth",
    #     "dataset": "data/base_glyphs.zip"
    # },
    # "All glyphs": {
    #     "vae": "checkpoints/mg-vae_glyphs.pth",
    #     "cls": "checkpoints/cls_glyphs_56px.pth",
    #     "dataset": "data/glyphs.zip"
    # }
}

st.set_page_config(page_title="Latent space operations", layout="wide")

@st.cache_resource
def load_unified_models(vae_path, classification_path):
    
    ckpt = torch.load(vae_path, map_location=DEVICE)
    cfg = ckpt['config'] if 'config' in ckpt else {
        'use_vae': True, 
        'resolution': (56, 56),
        'latent_dim': 128, 
        'value_dims': 8, 
        'backbone': {'model_name': 'resnet18'}, 
        'head': {
            'type': 'binned_regression',
            'min_val': -10,
            'max_val': 110,
            'temperature': 0.35
        }
    }
    res = cfg.get('resolution', RESOLUTION)
    vae = get_model(cfg, res)
    
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    fixed_state = {k.replace('estimator.', 'value_head.head.'): v for k, v in state.items()}
    vae.load_state_dict(fixed_state, strict=False)
    vae.to(DEVICE).eval()
    
    ckpt_c = torch.load(classification_path, map_location=DEVICE)
    cfg_c = ckpt_c['config'] if 'config' in ckpt_c else {
        'use_vae': False,
        'latent_dim': 128, 
        'value_dims': 8,
        'backbone': { 'name': 'timm', 'model_name': 'mobilenetv3_small_100'},
        'head': {'type': 'classification', 'classes': list(string.ascii_lowercase)}
    }
    
    classifier_classes = cfg.get('head', {}).get('classes', list(string.ascii_lowercase))
    classifier = get_model(cfg_c, res)
    
    cls_state = ckpt_c['state_dict'] if 'state_dict' in ckpt_c else ckpt_c
    fixed_cls_state = {k.replace('module.', ''): v for k, v in cls_state.items()}
    
    classifier.load_state_dict(fixed_cls_state, strict=False)
    classifier.to(DEVICE).eval()
    
    return vae, classifier, res, classifier_classes

@st.cache_resource
def process_dataset(_vae, zip_path, res, vae_path, vae_mtime):
    # glyphs contain only a few exmaples, therefore 'all' split is selected
    ds_split = "test" if zip_path.find("glyphs.zip") < 0 else "all"
    ds = GlyphDataset(zip_path, split=ds_split, resize=res)
    manifold = LatentManifold(_vae, DEVICE, res)
    all_z, all_true, all_classes = manifold.extract_latents(ds)
    splines, unique_classes = manifold.build_splines(all_z, all_true, all_classes)
    return all_z, all_true, all_classes, splines, unique_classes

@st.cache_data(show_spinner="Computing Multi-dimensional Reduction...")
def apply_dim_reduction(method, X_true, X_spline):
    if len(X_true) == 0:
        return np.array([]), np.array([])
        
    n_true = len(X_true)
    
    if method == "PCA":
        reducer = PCA(n_components=3)
        X_true_3d = reducer.fit_transform(X_true)
        X_spline_3d = reducer.transform(X_spline) if len(X_spline) else np.array([])
        
    elif method == "t-SNE":
        combined = np.vstack([X_true, X_spline]) if len(X_spline) else X_true
        reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(combined)-1))
        combined_3d = reducer.fit_transform(combined)
        
        X_true_3d = combined_3d[:n_true]
        X_spline_3d = combined_3d[n_true:] if len(X_spline) else np.array([])
        
    return X_true_3d, X_spline_3d

def plot_3d_latent_trajectory(method, all_z, all_true, all_classes, splines, selected_classes):
    if not selected_classes:
        return go.Figure()

    true_points, true_colors, true_counts = [], [],[]
    spline_points =[]
    
    for c in selected_classes:
        class_mask = (all_classes == c)
        true_points.append(all_z[class_mask])
        true_colors.append(all_true[class_mask])
        true_counts.append(class_mask.sum())
        
        if c in splines:
            v_interp = np.linspace(0, 100, 100)
            z_interp = np.array([splines[c](v) for v in v_interp])
            spline_points.append(z_interp)

    X_true = np.vstack(true_points) if true_points else np.array([])
    X_spline = np.vstack(spline_points) if spline_points else np.array([])
    
    X_true_3d, X_spline_3d = apply_dim_reduction(method, X_true, X_spline)

    fig = go.Figure()
    colors =['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#106dc4', '#FF6692', '#B6E880', '#FF97FF']
    
    t_idx = 0
    s_idx = 0

    for i, c in enumerate(selected_classes):
        c_color = colors[i % len(colors)]
        
        n_c = true_counts[i]
        c_3d = X_true_3d[t_idx : t_idx + n_c]
        c_val = true_colors[i]
        
        fig.add_trace(go.Scatter3d(
            x=c_3d[:, 0], y=c_3d[:, 1], z=c_3d[:, 2],
            mode='markers',
            marker=dict(size=6, color=c_val, colorscale='Viridis', opacity=0.75, showscale=(i==0)),
            name=f'{c} (True)',
            hovertemplate="Val: %{marker.color:.1f}<br>Dim 1: %{x:.2f}<br>Dim 2: %{y:.2f}<br>Dim 3: %{z:.2f}"
        ))
        t_idx += n_c

        if c in splines:
            s_3d = X_spline_3d[s_idx : s_idx + 100]
            
            fig.add_trace(go.Scatter3d(
                x=[s_3d[0,0], s_3d[-1,0]], y=[s_3d[0,1], s_3d[-1,1]], z=[s_3d[0,2], s_3d[-1,2]],
                mode='markers+text',
                marker=dict(size=6, color='black'),
                text=["0", "100"], textposition="top center",
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter3d(
                x=s_3d[:, 0], y=s_3d[:, 1], z=s_3d[:, 2],
                mode='lines',
                line=dict(width=8, color=c_color),
                name=f'{c} (Path)'
            ))
            s_idx += 100

    fig.update_layout(
        title=f"Latent Space Transformation ({method} Projection)",
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig
    
def decode_value(model, raw_pred):
    if hasattr(model.value_head, 'num_bins'):
        if raw_pred.dim() == 1:
            raw_pred = raw_pred.unsqueeze(0)
            
        temp = model.value_head.temperature
        min_v = model.value_head.min_val
        max_v = model.value_head.max_val
        bin_centers = torch.arange(min_v, max_v + 1, dtype=torch.float32).to(raw_pred.device)
        
        probs = F.softmax(raw_pred / temp, dim=1)
        decoded = torch.sum(probs * bin_centers, dim=1).squeeze()
        return decoded
        
    return raw_pred.squeeze()




# STREAMLIT APP UI
st.title("Latent space")

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

with st.sidebar:
    st.header("Model Selection")
    
    # Let user choose the mode
    mode = st.radio("Select Input Mode", ["Preset", "Custom"])
    
    vae_path, cls_path, zip_path = None, None, None
    ready_to_load = False

    if mode == "Preset":
        preset_choice = st.selectbox("Choose a dataset", list(PRESETS.keys()))
        
        vae_path = PRESETS[preset_choice]["vae"]
        cls_path = PRESETS[preset_choice]["cls"]
        zip_path = PRESETS[preset_choice]["dataset"]
        
        st.markdown(f"**VAE**: `{os.path.basename(vae_path)}`")
        st.markdown(f"**Classifier**: `{os.path.basename(cls_path)}`")
        st.markdown(f"**Dataset**: `{os.path.basename(zip_path)}`")
        
        # Check if preset files actually exist
        missing = [p for p in [vae_path, cls_path, zip_path] if not os.path.exists(p)]
        if missing:
            st.error(f"Missing files for preset: {missing}")
        else:
            if st.button("Load Preset", type="primary"):
                ready_to_load = True

    else:
        st.subheader("Upload Custom Files")
        vae_file = st.file_uploader("Upload VAE Checkpoint (.pth)", type=["pth"])
        cls_file = st.file_uploader("Upload Classifier Checkpoint (.pth)", type=["pth"])
        dataset_file = st.file_uploader("Upload Dataset (.zip)", type=["zip"])
        
        if vae_file and cls_file and dataset_file:
            if st.button("Load Custom Models", type="primary"):
                with st.spinner("Saving uploaded files temporarily..."):
                    temp_dir = tempfile.mkdtemp()
                    
                    vae_path = os.path.join(temp_dir, "vae.pth")
                    with open(vae_path, "wb") as f: f.write(vae_file.read())
                    
                    cls_path = os.path.join(temp_dir, "cls.pth")
                    with open(cls_path, "wb") as f: f.write(cls_file.read())
                    
                    zip_path = os.path.join(temp_dir, "dataset.zip")
                    with open(zip_path, "wb") as f: f.write(dataset_file.read())
                    
                    ready_to_load = True

    # Loading the models
    if ready_to_load:
        with st.spinner("Loading models and computing latent space..."):
            try:
                vae, classifier, res, classifier_classes = load_unified_models(vae_path, cls_path)
                st.session_state.vae = vae
                st.session_state.classifier = classifier
                st.session_state.resolution = res
                st.session_state.classifier_classes = classifier_classes
                st.session_state.zip_path = zip_path
                st.session_state.vae_path = vae_path
                st.session_state.vae_mtime = os.path.getmtime(vae_path) if os.path.exists(vae_path) else 0
                st.session_state.models_loaded = True
                st.success("Loaded")
            except Exception as e:
                st.error(f"Failed to load models {e}")
                

if st.session_state.get('models_loaded'):
    vae = st.session_state.vae
    classifier = st.session_state.classifier
    res = st.session_state.resolution
    classifier_classes = st.session_state.classifier_classes
    
    all_z, all_true, all_classes, splines, unique_classes = process_dataset(
        vae, 
        st.session_state.zip_path, 
        res,
        st.session_state.vae_path,
        st.session_state.vae_mtime
    )

    if classifier_classes != unique_classes:
        cs, ce = classifier_classes[0], classifier_classes[-1]
        ds, de = unique_classes[0], unique_classes[-1]
        st.warning(f"Classes from the dataset and the classifier may not match. Classifier: {cs}...{ce} Dataset: {ds}...{de}.")
    
    # 3D latent space manifold
    st.markdown("---")
    st.header("3D Latent space manifold")
    st.markdown("Visualizes the trajectory of specific classes as their scale value changes from 0 to 100.")
    
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        plot_method = st.radio("Dimensionality reduction:", ["PCA", "t-SNE"])
        selected_plot_classes = st.multiselect(
            "Classes to plot:",
            options=unique_classes,
            default=[unique_classes[0]] if len(unique_classes) > 0 else []
        )
        
    with col_p2:
        fig = plot_3d_latent_trajectory(
            plot_method, all_z, all_true, all_classes, splines, selected_plot_classes
        )
        st.plotly_chart(fig, width='stretch')

    # generation
    st.markdown("---")
    st.header("Glyph generation")
    col1, col2 = st.columns([1, 2])
    with col1:
        c = st.selectbox("Class/glyph:", unique_classes)
        val = st.slider("Scale value:", 0.0, 100.0, 80.0)
    
    with col2:
        z = torch.tensor(splines[c](val), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_pred, recon, _, _, _ = vae.forward_with_z(z)
            raw_pred = out_pred[0] if isinstance(out_pred, tuple) else out_pred
            
            pred = decode_value(vae, raw_pred)
            out_cls, _, _, _, _ = classifier(recon)
            logits = out_cls[1] if isinstance(out_cls, tuple) else out_cls
            
            if logits.dim() == 1: logits = logits.unsqueeze(0)
            prob = F.softmax(logits, dim=1)[0].cpu().numpy()
            
            top_idx = np.argsort(prob)[-3:][::-1]
            top3_list =[]
            for i in top_idx:
                class_name = classifier_classes[i] if i < len(classifier_classes) else f"Unknown({i})"
                top3_list.append(f"{class_name} ({prob[i]*100:.1f}%)")
                
            top_class = classifier_classes[top_idx[0]] if top_idx[0] < len(classifier_classes) else "?"
            
        st.image(np.clip(recon.squeeze().cpu().permute(1,2,0).numpy(), 0, 1), 
                 caption=f"Reconstruction (Classifier: {top_class}, Pred: {pred.item():.1f})")
    
    with col1:
        st.subheader("Top 3 classifier predictions")
        st.markdown(" ".join([f"{i+1}. **{p}**\n " for i, p in enumerate(top3_list)]))
        
    # Latent Interpolation
    st.markdown("---")
    st.header("Interpolation between glyphs")
    st.markdown("Transforms between two classes and scale values along the spherical prior.")
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        class_start = st.selectbox("Start glyph:", unique_classes, index=0)
        val_start = st.slider("Start value:", 0.0, 100.0, 80.0, key='val_start')
    with col_i2:
        class_end = st.selectbox("End glyph:", unique_classes, index=min(1, len(unique_classes)-1))
        val_end = st.slider("End value:", 0.0, 100.0, 80.0, key='val_end')
        
    if st.button("Generate Interpolation", type="primary"):
        with torch.no_grad():
            z_s = splines[class_start](val_start)
            z_e = splines[class_end](val_end)
            
            alphas = np.linspace(0, 1, 10)
            imgs =[]
            for a in alphas:
                z_t = LatentManifold.slerp(a, z_s, z_e)
                z_tensor = torch.tensor(z_t, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                recon_feat = vae.decoder_input(z_tensor)
                img = vae.decoder(recon_feat)
                if img.shape[-2:] != res:
                    img = F.interpolate(img, size=res, mode='bilinear', align_corners=False)
                imgs.append(np.clip(img.squeeze(0).cpu().permute(1,2,0).numpy(), 0, 1))
            
            st.image(imgs, width=120, caption=[f"t = {a:.1f}" for a in alphas])

    # Barycentric Triangle Interpolation
    st.markdown("---")
    st.header("Barycentric interpolation")
    st.markdown("Click on a point within the interactive Maxwell triangle to interpolate between 3 selected glyphs.")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    
    # Shade the selectors according to the RGB tips using styled markdown containers
    with col_b1:
        st.markdown("<div style='background-color: rgba(255,0,0,0.1); padding:8px; border-radius:5px; border-left: 4px solid #EF553B; margin-bottom:10px;'><b>Glyph A (Top)</b></div>", unsafe_allow_html=True)
        class_ta = st.selectbox("Select Class:", unique_classes, index=0, key='ta')
        val_ta = st.slider("Scale value:", 0.0, 100.0, 80.0, key='vta')
    with col_b2:
        st.markdown("<div style='background-color: rgba(0,204,150,0.1); padding:8px; border-radius:5px; border-left: 4px solid #00CC96; margin-bottom:10px;'><b>Glyph B (Bottom Left)</b></div>", unsafe_allow_html=True)
        class_tb = st.selectbox("Select Class:", unique_classes, index=min(1, len(unique_classes)-1), key='tb')
        val_tb = st.slider("Scale value:", 0.0, 100.0, 80.0, key='vtb')
    with col_b3:
        st.markdown("<div style='background-color: rgba(25,211,243,0.1); padding:8px; border-radius:5px; border-left: 4px solid #106dc4; margin-bottom:10px;'><b>Glyph C (Bottom Right)</b></div>", unsafe_allow_html=True)
        class_tc = st.selectbox("Select Class:", unique_classes, index=min(2, len(unique_classes)-1), key='tc')
        val_tc = st.slider("Scale value:", 0.0, 100.0, 80.0, key='vtc')

    V_A = np.array([0.0, 1.0])
    V_B = np.array([-1.0, -0.5])
    V_C = np.array([1.0, -0.5])
    
    points =[]
    colors_list =[]
    for i in range(N+1):
        for j in range(N+1-i):
            k = N - i - j
            u, v, w = i / N, j / N, k / N
            p = u * V_A + v * V_B + w * V_C
            points.append((p[0], p[1], u, v, w))
            
            r_c = int((u ** 0.5) * 255)
            g_c = int((v ** 0.5) * 255)
            b_c = int((w ** 0.5) * 255)
            colors_list.append(f'rgb({r_c}, {g_c}, {b_c})')
            
    df_tri = pd.DataFrame(points, columns=['x', 'y', 'u', 'v', 'w'])
    fig_tri = go.Figure()
    
    # interactive points
    fig_tri.add_trace(go.Scatter(
        x=df_tri['x'], y=df_tri['y'], mode='markers',
        marker=dict(size=16, color=colors_list, opacity=0.9, line=dict(width=0.5, color='white')),
        customdata=df_tri[['u', 'v', 'w']],
        hovertemplate="A: %{customdata[0]:.2f}<br>B: %{customdata[1]:.2f}<br>C: %{customdata[2]:.2f}<extra></extra>",
        showlegend=False,
    ))
    
    # Annotations positioned at the tips of the triangle
    fig_tri.add_trace(go.Scatter(
        x=[V_A[0], V_B[0], V_C[0]], 
        y=[V_A[1]+0.15, V_B[1]-0.15, V_C[1]-0.15],
        mode='text',
        text=[f"<b>{class_ta}</b>", f"<b>{class_tb}</b>", f"<b>{class_tc}</b>"],
        textfont=dict(size=15, color=['#EF553B', '#00CC96', '#106dc4']),
        hoverinfo='skip',
        showlegend=False
    ))
    
    fig_tri.update_layout(
        xaxis=dict(visible=False, range=[-1.4, 1.4], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.8, 1.4], fixedrange=True),
        margin=dict(l=0, r=0, t=0, b=0),
        height=350,
        clickmode='event+select',
        dragmode=False,
        hovermode='closest',
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    col_t1, col_t2 = st.columns([2, 1])
    
    with col_t1:
        try:
            # Streamlit >= 1.35 supports fetching data from Plotly chart selections
            selection = st.plotly_chart(fig_tri, on_select="rerun", selection_mode="points", width='stretch')
            if selection and selection.get('selection') and selection['selection']['points']:
                pt = selection['selection']['points'][0]
                idx = pt['point_index']
                u, v, w = df_tri.iloc[idx][['u', 'v', 'w']]
            else:
                u, v, w = 0.33, 0.33, 0.34 # Default to middle
        except TypeError:
            st.plotly_chart(fig_tri, width='stretch')
            st.info("Interactive clicking requires Streamlit >= 1.35. Displaying center image.")
            u, v, w = 0.33, 0.33, 0.34

    with col_t2:
        st.write("**Barycentric Coordinates**")
        st.write(f"**{class_ta}:** {u:.2f}")
        st.write(f"**{class_tb}:** {v:.2f}")
        st.write(f"**{class_tc}:** {w:.2f}")
        
        with torch.no_grad():
            Z_ta = torch.tensor(splines[class_ta](val_ta), dtype=torch.float32).to(DEVICE)
            Z_tb = torch.tensor(splines[class_tb](val_tb), dtype=torch.float32).to(DEVICE)
            Z_tc = torch.tensor(splines[class_tc](val_tc), dtype=torch.float32).to(DEVICE)
            
            Z_target = u * Z_ta + v * Z_tb + w * Z_tc
            
            recon_feat = vae.decoder_input(Z_target.unsqueeze(0))
            img = vae.decoder(recon_feat)
            if img.shape[-2:] != res:
                img = F.interpolate(img, size=res, mode='bilinear', align_corners=False)
                
            img_np = np.clip(img.squeeze(0).cpu().permute(1,2,0).numpy(), 0, 1)
            
        st.image(img_np, caption="Generated interpolation", width=200)


    # Latent Algebra
    st.markdown("---")
    st.header("Latent algebra")
    
    op = st.selectbox(
        "Mathematical operation:",["(A + B) / 2 = Result", "A - B + C = Result", "A + B = Result", "A - B = Result"], 
        index=0
    )
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        class_a = st.selectbox("Glyph A:", unique_classes, index=0, key='ca')
        val_a = st.slider("Value A:", 0.0, 100.0, 100.0, key='va')
    with col_m2:
        class_b = st.selectbox("Glyph B:", unique_classes, index=min(1, len(unique_classes)-1), key='cb')
        val_b = st.slider("Value B:", 0.0, 100.0, 100.0, key='vb')
        
    uses_c = "C" in op
    if uses_c:
        with col_m3:
            class_c = st.selectbox("Glyph C:", unique_classes, index=min(2, len(unique_classes)-1), key='cc')
            val_c = st.slider("Value C:", 0.0, 100.0, 100.0, key='vc')

    if st.button("Show computed result", type="primary"):
        with torch.no_grad():
            Z_a = torch.tensor(splines[class_a](val_a), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            Z_b = torch.tensor(splines[class_b](val_b), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            if uses_c:
                Z_c = torch.tensor(splines[class_c](val_c), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            if op == "(A + B) / 2 = Result":
                Z_target = (Z_a + Z_b) / 2.0
                vecs =[Z_a, Z_b, Z_target]
                caps =[f"A: {class_a}", f"+ B: {class_b}", "Mean Result"]
            elif op == "A - B + C = Result":
                Z_target = Z_a - Z_b + Z_c
                vecs =[Z_a, Z_b, Z_c, Z_target]
                caps =[f"A: {class_a}", f"- B: {class_b}", f"+ C: {class_c}", "Result"]
            elif op == "A + B = Result":
                Z_target = Z_a + Z_b
                vecs = [Z_a, Z_b, Z_target]
                caps =[f"A: {class_a}", f"+ B: {class_b}", "Result"]
            elif op == "A - B = Result":
                Z_target = Z_a - Z_b
                vecs =[Z_a, Z_b, Z_target]
                caps =[f"A: {class_a}", f"- B: {class_b}", "Result"]
            
            imgs =[]
            for v in vecs:
                recon_feat = vae.decoder_input(v)
                img = vae.decoder(recon_feat)
                if img.shape[-2:] != res:
                    img = F.interpolate(img, size=res, mode='bilinear', align_corners=False)
                imgs.append(np.clip(img.squeeze(0).cpu().permute(1,2,0).numpy(), 0, 1))
            
            st.image(imgs, width=150, caption=caps)