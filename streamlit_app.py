import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from backend.ml.processor import process_csv
from backend.ml.core_engine import execute_federated_training

# --- PAGE CONFIG ---
st.set_page_config(page_title="SegFL Analytics", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZE STATE ---
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'is_run' not in st.session_state:
    st.session_state.is_run = False

# --- PREMIUM CSS ---
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    @keyframes fadein {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #18181b !important;
    }
    
    /* Crisp Light Background */
    .stApp {
        background-color: #fafafa !important;
        background-image: none;
    }
    
    .main-header {
        color: #09090b !important;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 0;
        letter-spacing: -1px;
        animation: fadein 0.8s ease-out;
    }
    .sub-header {
        color: #71717a;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
        animation: fadein 1s ease-out;
    }
    
    /* Clean Dashboard Cards */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
        background: #ffffff !important;
        border: 1px solid #e4e4e7 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        animation: fadein 0.6s ease-out forwards;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e4e4e7;
    }
    
    /* Primary Action Buttons */
    .stButton>button, [data-testid="stDownloadButton"] button {
        width: 100% !important;
        background: #18181b !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    .stButton>button:hover, [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        color: #ffffff !important;
    }
    .stButton>button p, [data-testid="stDownloadButton"] button p {
        color: #ffffff !important;
    }
    
    /* Input Fields */
    .stTextInput input, .stNumberInput input {
        background-color: #ffffff !important;
        border: 1px solid #e4e4e7 !important;
        color: #18181b !important;
        border-radius: 6px !important;
    }
    
    /* Plotly transparent overrides */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }
    
    /* Scrollable Execution Terminal */
    .stCodeBlock pre {
        max-height: 250px;
        overflow-y: auto !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
    .stCodeBlock code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
</style>
''', unsafe_allow_html=True)

# --- REPRODUCIBILITY ---
def enforce_reproducibility(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LOGIC ---
@st.cache_data
def cached_process_csv(file_path, nrows):
    return process_csv(file_path, nrows=nrows)

# --- UI INTERFACE ---
st.markdown("<h1 class='main-header'>SegFL Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Privacy-Preserving Federated Behavioural Segmentation</p>", unsafe_allow_html=True)

st.markdown("""
**SegFL Analytics** is a decentralized machine learning framework designed to cluster user behavioral profiles across multiple isolated organizations (tenants) without ever sharing raw data. It utilizes a **Tenant Adapter Layer (TAL)** to unify heterogeneous data structures, **Federated Learning** to securely aggregate global intelligence, and **Differential Privacy (DP-SGD)** to mathematically guarantee user anonymity.
""")

with st.sidebar:
    st.header("Protocol Settings")
    use_local = st.checkbox("Use Local Project Data (backend/data/)", value=True)
    
    uploaded_file = None
    if not use_local:
        uploaded_file = st.file_uploader("Upload behavioural CSV", type=["csv"])
    
    nrows = st.number_input("Scan Rows", min_value=1000, max_value=100000000, value=50000, step=5000)
    g_epochs = st.slider("Global Rounds", 1, 30, 10)
    l_epochs = st.slider("Local Epochs", 1, 10, 3)
    n_clusters = st.slider("Number of Clusters (K)", 2, 10, 5)
    use_fedprox = st.checkbox("Use FedProx (Proximal Term)", value=False)
    st.divider()
    btn_run = st.button("Execute SegFL Sequence")

# ── PERSISTENT EXECUTION TERMINAL ──
st.markdown("---")
st.markdown("### 💻 System Execution Terminal")
with st.container(height=300):
    log_container = st.empty()
    if st.session_state.logs:
        log_container.code("\n".join(st.session_state.logs), language="bash")

# --- EXECUTION LOGIC ---
if btn_run:
    dynamic_seed = int(time.time() * 1000) % 10000
    enforce_reproducibility(dynamic_seed)
    file_to_process = None
    
    if use_local:
        data_dir = os.path.join("backend", "data")
        os.makedirs(data_dir, exist_ok=True)
        click_path = os.path.join(data_dir, "clicks_train.csv")
        if os.path.exists(click_path): 
            file_to_process = click_path
        else:
            csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            if csvs: 
                file_to_process = os.path.join(data_dir, csvs[0])
            else:
                st.warning("No local datasets found in `backend/data/`. Please copy `clicks_train.csv` there.")
    elif uploaded_file:
        os.makedirs("temp", exist_ok=True)
        file_to_process = os.path.join("temp", uploaded_file.name)
        with open(file_to_process, "wb") as f: f.write(uploaded_file.getbuffer())
    
    if file_to_process:
        st.session_state.is_run = False
        st.session_state.logs = []
        
        def add_log(msg):
            ts = time.strftime("%H:%M:%S")
            st.session_state.logs.append(f"[{ts}] {msg}")
            log_container.code("\n".join(st.session_state.logs), language="bash")

        base_params = {'g_epochs': g_epochs, 'l_epochs': l_epochs, 'n_clusters': n_clusters, 'use_fedprox': use_fedprox}
        
        add_log("Loading dataset into memory...")
        raw_df = cached_process_csv(file_to_process, nrows=nrows)
        
        # 1. Baselines
        add_log("Initializing Baseline Comparisons...")
        results_list = []
        modes = ['cent', 'intersect', 'local', 'tal']
        display_names = {'cent': 'Centralized', 'intersect': 'Intersection-Only', 'local': 'Local-Isolated', 'tal': 'TAL-FL'}
        for m in modes:
            res = execute_federated_training(raw_df, {**base_params, 'mode': m, 'sigma': 0.0}, log_callback=add_log)
            results_list.append({
                'Mode': display_names[m], 
                'Avg Silhouette': f"{res['silhouette']:.3f}",
                'Comm. Cost (MB)': f"{res['comm_cost_mb']:.4f}"
            })
        base_df = pd.DataFrame(results_list)
        
        # 2. Ablation
        add_log("Running Component Ablation Analysis...")
        abl_results = []
        conditions = [
            {'name': 'Full SegFL (TAL + DP)', 'mode': 'tal', 'sigma': 0.1},
            {'name': 'Ablated (No DP Noise)', 'mode': 'tal', 'sigma': 0.0},
            {'name': 'Ablated (No Federated Agg)', 'mode': 'local', 'sigma': 0.1}
        ]
        for cond in conditions:
            res = execute_federated_training(raw_df, {**base_params, 'mode': cond['mode'], 'sigma': cond['sigma']}, log_callback=add_log)
            abl_results.append({
                'Experimental Condition': cond['name'], 
                'Utility (Silhouette)': f"{res['silhouette']:.3f}",
                'Privacy (ε)': f"≈ {res['epsilon']:.2f}" if cond['sigma'] > 0 else "∞ (None)",
                'Comm. Cost (MB)': f"{res['comm_cost_mb']:.4f}"
            })
        abl_df = pd.DataFrame(abl_results)

        # 3. DP Sweep
        add_log("Executing Differential Privacy Utility Sweep...")
        dp_results = []
        for s in [0.0, 0.05, 0.2, 0.5, 1.0]:
            res = execute_federated_training(raw_df, {**base_params, 'sigma': s}, log_callback=None)
            dp_results.append({'DP Sigma': s, 'Privacy Budget (ε)': res['epsilon'], 'Retained Utility': res['silhouette']})
        dp_df = pd.DataFrame(dp_results)
        
        # 4. Final Profile
        add_log("Finalizing High-Fidelity Persona Matrix (Sigma=0.1)...")
        results = execute_federated_training(raw_df, {**base_params, 'sigma': 0.1}, log_callback=add_log)
        
        add_log("Full Research Cycle Complete. Rendering Dashboard...")
        
        # Save results globally
        st.session_state.results = results
        st.session_state.base_df = base_df
        st.session_state.abl_df = abl_df
        st.session_state.dp_df = dp_df
        st.session_state.is_run = True

    else:
        st.warning("Please provide a dataset.")

# --- RENDERING DASHBOARD OUTSIDE THE BUTTON LOOP ---
if st.session_state.get('is_run', False):
    results = st.session_state.results
    base_df = st.session_state.base_df
    abl_df = st.session_state.abl_df
    dp_df = st.session_state.dp_df

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # KPI Summary Bar (always visible above tabs)
    st.markdown("### Performance Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Final Silhouette Score", f"{results['silhouette']:.3f}")
    kpi2.metric("Davies-Bouldin Index", f"{results['dbi']:.3f}")
    kpi3.metric("Privacy Budget (ε)", f"≈ {results['epsilon']:.2f}")
    kpi4.metric("Discovered Archetypes", len(results['profile']))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- TABBED DASHBOARD ---
    tab_convergence, tab_baselines, tab_privacy, tab_topology, tab_artifacts = st.tabs([
        "Convergence",
        "Baselines & Ablation",
        "Privacy Analysis",
        "Latent Topology & Personas",
        "Archetypes & Downloads"
    ])
    
    # ── TAB 1: Convergence ──
    with tab_convergence:
        st.markdown("#### TAL-FL Global Convergence (Loss)")
        st.caption("Reconstruction loss of the GlobalBottleneckAE across federated training rounds.")
        loss_df = pd.DataFrame({
            'Global Round': range(1, len(results['loss_history']) + 1),
            'MSE Loss': results['loss_history']
        })
        fig_loss = px.line(loss_df, x='Global Round', y='MSE Loss', template='plotly_white', markers=True)
        fig_loss.update_traces(line=dict(color='#18181b', width=2.5), marker=dict(size=7))
        fig_loss.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Reconstruction Loss",
            xaxis_title="Global Round"
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # ── TAB 2: Baselines & Ablation ──
    with tab_baselines:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("#### Formal Baseline Comparisons")
            st.caption("Silhouette scores and communication costs across standard FL approaches.")
            st.dataframe(base_df, use_container_width=True, hide_index=True)
        with col_b2:
            st.markdown("#### Component Ablation Study")
            st.caption("Impact of removing individual SegFL components on utility and privacy.")
            st.dataframe(abl_df, use_container_width=True, hide_index=True)
    
    # ── TAB 3: Privacy Analysis ──
    with tab_privacy:
        st.markdown("#### Differential Privacy Utility Tradeoff")
        st.caption("Effect of DP noise magnitude (Sigma) on clustering utility.")
        fig_dp = px.line(
            dp_df, x='DP Sigma', y='Retained Utility',
            template='plotly_white', markers=True,
            labels={'DP Sigma': 'DP Noise (σ)', 'Retained Utility': 'Silhouette Score'}
        )
        fig_dp.update_traces(line=dict(color='#18181b', width=2.5), marker=dict(size=8))
        fig_dp.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_dp, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(dp_df.style.format({
            'DP Sigma': '{:.2f}',
            'Privacy Budget (ε)': '{:.2f}',
            'Retained Utility': '{:.3f}'
        }).background_gradient(cmap='Greys', subset=['Retained Utility']), use_container_width=True, hide_index=True)
    
    # ── TAB 4: Latent Topology & Personas ──
    with tab_topology:
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### Latent Topology (t-SNE)")
            st.caption("2D projection of the 4-dimensional TAL latent space.")
            if len(results['lats']) > 1500:
                sample_idx = np.random.choice(len(results['lats']), 1500, replace=False)
                tsne_data = results['lats'][sample_idx]
                tsne_labels = results['labels'][sample_idx]
            else:
                tsne_data = results['lats']
                tsne_labels = results['labels']
            
            n_samples = len(tsne_data)
            safe_perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
            tsne = TSNE(n_components=2, perplexity=safe_perplexity, random_state=42)
            l_reduced = tsne.fit_transform(tsne_data)
            viz_df = pd.DataFrame(l_reduced, columns=['x', 'y'])
            viz_df['Cluster'] = tsne_labels.astype(str)
            fig_p = px.scatter(viz_df, x='x', y='y', color='Cluster', template='plotly_white')
            fig_p.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
                margin=dict(l=0, r=0, t=20, b=0),
                height=420
            )
            st.plotly_chart(fig_p, use_container_width=True)

        with viz_col2:
            st.markdown("#### Semantic Feature Importance")
            st.caption("Normalized behavioural radar per discovered persona.")
            
            # Dynamically determine columns to display on the radar chart
            all_profile_cols = [c for c in results['profile'].columns if c not in ['Persona', 'Cluster Size', 'Platform ID']]
            # Limit to at most 5 columns for readability
            radar_cols = all_profile_cols[:5]
            
            if len(radar_cols) >= 3:
                fig_radar = go.Figure()
                
                norm_profile = results['profile'][radar_cols].copy()
                col_mins = norm_profile.min()
                col_maxs = norm_profile.max()
                norm_profile = (norm_profile - col_mins) / (col_maxs - col_mins + 1e-5)
                
                for idx, row in norm_profile.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[c] for c in radar_cols],
                        theta=radar_cols,
                        fill='toself',
                        name=f"{results['profile'].loc[idx, 'Persona']}"
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=False, showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                        angularaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                    ),
                    showlegend=True,
                    template='plotly_white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=20, b=20),
                    height=420
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Insufficient numeric features for radar chart display.")
    
    # ── TAB 5: Archetypes & Downloads ──
    with tab_artifacts:
        st.markdown("#### Discovered Behavioural Archetypes")
        st.caption("Mean feature values per cluster with auto-assigned persona labels.")
        st.dataframe(results['profile'].style.background_gradient(cmap='Greys'), use_container_width=True)
        
        with st.expander("📖 View Persona Legend", expanded=True):
            st.markdown("""
            - **High-Intent Engager:** Clicks on a high percentage of ads (High Click-Through Rate).
            - **High-Velocity Consumer:** Interacts frequently with the platform (High Interaction Volume).
            - **Exploratory Navigator:** Views a highly diverse set of ads (High Ad Entropy).
            - **Passive Observer:** Rarely clicks on ads despite seeing them (Low Click-Through Rate).
            - **Infrequent Visitor:** Rarely interacts with the platform (Low Interaction Volume).
            - **Balanced Generalist:** Displays average behavior across all metrics without extreme spikes.
            - **Unpopulated Segment:** The algorithm found no users matching this geometric centroid.
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        st.markdown("#### Download Research Artifacts")
        
        dl_col1, dl_col2 = st.columns([1, 2])
        
        with dl_col1:
            st.markdown("**Master Profile Table**")
            st.download_button(
                label="Download Archetypes Profile (CSV)",
                data=results['profile'].to_csv(index=True).encode('utf-8'),
                file_name="segfl_profile_results.csv",
                mime="text/csv"
            )
            
        with dl_col2:
            st.markdown("**Segregated Cluster Data**")
            for cluster_id in results['profile'].index:
                cluster_df = results['clustered_data'][results['clustered_data']['cluster'] == cluster_id]
                csv_data = cluster_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Cluster {cluster_id} Data (CSV)",
                    data=csv_data,
                    file_name=f"cluster_{cluster_id}_data.csv",
                    mime="text/csv",
                    key=f"dl_cluster_{cluster_id}"
                )
