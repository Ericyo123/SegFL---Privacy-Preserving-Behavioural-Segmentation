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
from backend.ml.evaluator import (
    stability_analysis, wilcoxon_test,
    scalability_analysis, generalization_test
)

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
**SegFL Analytics** is a decentralized machine learning framework designed to cluster user behavioral profiles across multiple isolated organizations (tenants) without ever sharing raw data. It utilizes a **Tenant Adapter Layer (TAL)** to unify heterogeneous data structures, **Federated Learning** to securely aggregate global intelligence, and **Differential Privacy (DP-SGD)** with formal **Rényi DP accounting** to mathematically guarantee user anonymity.
""")

with st.sidebar:
    st.header("Protocol Settings")
    use_local = st.checkbox("Use Local Project Data (backend/data/)", value=True)
    
    local_file_choice = None
    uploaded_file = None
    if use_local:
        data_dir = os.path.join("backend", "data")
        os.makedirs(data_dir, exist_ok=True)
        csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        if csvs:
            if "clicks_train.csv" in csvs:
                csvs.remove("clicks_train.csv")
                csvs = ["clicks_train.csv"] + sorted(csvs)
            else:
                csvs = sorted(csvs)
            local_file_choice = st.selectbox("Select local CSV file", csvs)
        else:
            st.warning("No local CSV files found in `backend/data/`. Please copy your dataset files there.")
    else:
        uploaded_file = st.file_uploader("Upload behavioural CSV", type=["csv"])
    
    nrows = st.number_input("Scan Rows", min_value=1000, max_value=100000000, value=50000, step=5000)
    g_epochs = st.slider("Global Rounds", 1, 30, 10)
    l_epochs = st.slider("Local Epochs", 1, 10, 3)
    n_clusters = st.slider("Number of Clusters (K)", 2, 10, 5)
    
    st.divider()
    st.subheader("Clustering & Privacy")
    clustering_method = st.selectbox("Clustering Algorithm", ["kmeans", "gmm", "hdbscan"], index=0)
    sigma_dp = st.slider("DP Noise Multiplier (σ)", 0.0, 2.0, 0.1, step=0.05)
    
    st.divider()
    st.subheader("Scientific Evaluation")
    run_stability = st.checkbox("Run Stability Analysis (10 seeds)", value=False)
    run_scalability_gen = st.checkbox("Run Scalability & Generalization", value=False)
    
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
        if local_file_choice:
            file_to_process = os.path.join("backend", "data", local_file_choice)
        else:
            st.warning("No local dataset selected. Please copy CSV files to `backend/data/`.")
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

        base_params = {
            'g_epochs': g_epochs,
            'l_epochs': l_epochs,
            'n_clusters': n_clusters,
            'clustering_method': clustering_method,
            'run_seed': dynamic_seed
        }
        
        add_log("Loading dataset into memory...")
        raw_df = cached_process_csv(file_to_process, nrows=nrows)
        
        # Sample representative subset for comparative sweeps to keep runtime tractable on CPU
        sweep_df = raw_df
        if len(raw_df) > 10000:
            add_log("🔬 Dataset size is large. Downsampling to 10,000 rows for comparisons and sweeps to maintain CPU tractability...")
            sweep_df = raw_df.sample(n=10000, random_state=dynamic_seed).reset_index(drop=True)
        
        # ── 1. BASELINES (Centralized, Intersection, Local, TAL-FL, FedProx, SCAFFOLD, MOON) ──
        add_log("Initializing Baseline Comparisons...")
        results_list = []
        modes = ['cent', 'intersect', 'local', 'tal', 'fedprox', 'scaffold', 'moon']
        display_names = {
            'cent': 'Centralized',
            'intersect': 'Intersection-Only',
            'local': 'Local-Isolated',
            'tal': 'TAL-FL (SegFL)',
            'fedprox': 'FedProx + TAL',
            'scaffold': 'SCAFFOLD + TAL',
            'moon': 'MOON + TAL'
        }
        
        # Run Centralized first to get target labels for NMI/ARI
        cent_res = execute_federated_training(sweep_df, {**base_params, 'mode': 'cent', 'sigma': 0.0}, log_callback=add_log)
        cent_labels = cent_res['labels']
        
        from backend.ml.evaluator import compute_nmi_ari
        
        for m in modes:
            if m == 'cent':
                res = cent_res
            else:
                res = execute_federated_training(sweep_df, {**base_params, 'mode': m, 'sigma': 0.0}, log_callback=add_log)
            
            agree = compute_nmi_ari(res['labels'], cent_labels)
            results_list.append({
                'Mode': display_names[m], 
                'Silhouette ↑': f"{res['silhouette']:.3f}",
                'DBI ↓': f"{res['dbi']:.3f}",
                'Calinski-Harabasz ↑': f"{res['calinski_harabasz']:.1f}",
                'NMI (vs Cent) ↑': f"{agree['nmi']:.3f}",
                'ARI (vs Cent) ↑': f"{agree['ari']:.3f}",
                'Comm. Cost (MB)': f"{res['comm_cost_mb']:.4f}"
            })
        base_df = pd.DataFrame(results_list)
        
        # ── 2. ABLATION (Full SegFL, No DP, No Federation, No TAL) ──
        add_log("Running Component Ablation Analysis...")
        abl_results = []
        conditions = [
            {'name': 'Full SegFL (TAL + Fed + DP)', 'mode': 'tal', 'sigma': sigma_dp},
            {'name': 'Ablated: No DP Noise', 'mode': 'tal', 'sigma': 0.0},
            {'name': 'Ablated: No Federation', 'mode': 'local', 'sigma': sigma_dp},
            {'name': 'Ablated: No TAL (Centralized)', 'mode': 'cent', 'sigma': sigma_dp},
        ]
        for cond in conditions:
            res = execute_federated_training(sweep_df, {**base_params, 'mode': cond['mode'], 'sigma': cond['sigma']}, log_callback=add_log)
            agree = compute_nmi_ari(res['labels'], cent_labels)
            abl_results.append({
                'Condition': cond['name'], 
                'Silhouette ↑': f"{res['silhouette']:.3f}",
                'DBI ↓': f"{res['dbi']:.3f}",
                'NMI (vs Cent) ↑': f"{agree['nmi']:.3f}",
                'ARI (vs Cent) ↑': f"{agree['ari']:.3f}",
                'Privacy (ε)': f"{res['epsilon']:.2f}" if cond['sigma'] > 0 else "∞ (None)",
                'Comm. Cost (MB)': f"{res['comm_cost_mb']:.4f}"
            })
        abl_df = pd.DataFrame(abl_results)

        # ── 3. DP SWEEP ──
        add_log("Executing Differential Privacy Utility Sweep...")
        dp_results = []
        # Center the sweep around the chosen sigma to make it highly relevant and faster
        if sigma_dp > 0.0:
            sweep_sigmas = [0.0, float(np.round(sigma_dp * 0.5, 3)), sigma_dp, float(np.round(sigma_dp * 2.0, 3)), 2.0]
        else:
            sweep_sigmas = [0.0, 0.1, 0.5, 1.0, 2.0]
        sweep_sigmas = sorted(list(set([s for s in sweep_sigmas if s >= 0.0])))

        for s in sweep_sigmas:
            add_log(f"Sweeping DP Noise Multiplier (σ) = {s} (Target chosen σ = {sigma_dp})...")
            res = execute_federated_training(sweep_df, {**base_params, 'sigma': s}, log_callback=None)
            dp_results.append({
                'DP Sigma (σ)': s,
                'Privacy Budget (ε)': res['epsilon'],
                'Silhouette': res['silhouette'],
                'DBI': res['dbi']
            })
        dp_df = pd.DataFrame(dp_results)
        
        # ── 4. FINAL PROFILE ──
        add_log(f"Finalizing High-Fidelity Persona Matrix (σ={sigma_dp}) on Full Dataset...")
        results = execute_federated_training(raw_df, {**base_params, 'sigma': sigma_dp}, log_callback=add_log)
        
        # ── 5. STABILITY ANALYSIS (optional) ──
        stability_results = None
        stat_test_results = None
        if run_stability:
            add_log("Starting Stability Analysis (10 seeds)...")
            stability_results = stability_analysis(
                sweep_df, {**base_params, 'sigma': sigma_dp},
                execute_fn=execute_federated_training,
                n_seeds=10,
                log_callback=add_log
            )
            
            add_log("Running Centralized baseline across same seeds for statistical test...")
            cent_stability = stability_analysis(
                sweep_df, {**base_params, 'sigma': sigma_dp, 'mode': 'cent'},
                execute_fn=execute_federated_training,
                n_seeds=10,
                log_callback=None
            )
            stat_test_results = wilcoxon_test(
                stability_results['all_silhouettes'],
                cent_stability['all_silhouettes']
            )
            add_log(f"Wilcoxon test: p={stat_test_results['p_value']:.4f} "
                     f"({'Significant' if stat_test_results['significant'] else 'Not significant'} at α=0.05)")
        
        # ── 6. SCALABILITY & GENERALIZATION (optional) ──
        scal_results = None
        gen_results = None
        if run_scalability_gen:
            add_log("Starting Scalability Analysis (scaling tenant count)...")
            from backend.ml.processor import prepare_tenant_datasets
            _, _, raw_info_temp = prepare_tenant_datasets(sweep_df, batch_size=256, run_seed=dynamic_seed)
            max_tenants = len(raw_info_temp)
            if max_tenants >= 2:
                if max_tenants <= 5:
                    tenant_counts = list(range(2, max_tenants + 1))
                else:
                    # Choose up to 5 evenly spaced integers between 2 and max_tenants
                    tenant_counts = sorted(list(set(np.linspace(2, max_tenants, 5, dtype=int))))
            else:
                tenant_counts = [2]
            
            scal_results = scalability_analysis(
                sweep_df, {**base_params, 'mode': 'tal', 'sigma': sigma_dp},
                execute_fn=execute_federated_training,
                tenant_counts=tenant_counts,
                log_callback=add_log
            )
            
            add_log("Starting Generalization hold-out tenant test...")
            gen_results = generalization_test(
                sweep_df, {**base_params, 'mode': 'tal', 'sigma': sigma_dp},
                execute_fn=execute_federated_training,
                log_callback=add_log
            )
        
        # ── 6.5. COMPUTE EXPLAINABILITY ──
        add_log("Computing post-hoc surrogate explainability and feature attributions...")
        from backend.ml.explainability import compute_cluster_explainability
        importance_df, enrichment_scores = compute_cluster_explainability(
            results['clustered_data'], results['labels']
        )
        
        add_log("Full Research Cycle Complete. Rendering Dashboard...")
        
        # Save results globally
        st.session_state.importance_df = importance_df
        st.session_state.enrichment_scores = enrichment_scores
        st.session_state.results = results
        st.session_state.base_df = base_df
        st.session_state.abl_df = abl_df
        st.session_state.dp_df = dp_df
        st.session_state.stability_results = stability_results
        st.session_state.stat_test_results = stat_test_results
        st.session_state.scal_results = scal_results
        st.session_state.gen_results = gen_results
        
        # ── 7. PDF REPORT GENERATION ──
        add_log("Generating PDF Evaluation Report...")
        from backend.ml.report_generator import build_pdf_report
        import tempfile
        
        report_data = {
            'results': results,
            'base_df': base_df,
            'abl_df': abl_df,
            'dp_df': dp_df,
            'stability_results': stability_results,
            'stat_test_results': stat_test_results,
            'scal_results': scal_results,
            'gen_results': gen_results,
            'importance_df': importance_df,
            'enrichment_scores': enrichment_scores
        }
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            build_pdf_report(report_data, tmp_path)
            with open(tmp_path, "rb") as f:
                pdf_bytes = f.read()
            st.session_state.pdf_report = pdf_bytes
            add_log("PDF report compilation complete.")
        except Exception as e:
            add_log(f"⚠️ PDF Generation failed: {str(e)}")
            st.session_state.pdf_report = None
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
                
        st.session_state.is_run = True

    else:
        st.warning("Please provide a dataset.")

# --- RENDERING DASHBOARD OUTSIDE THE BUTTON LOOP ---
if st.session_state.get('is_run', False):
    results = st.session_state.results
    base_df = st.session_state.base_df
    abl_df = st.session_state.abl_df
    dp_df = st.session_state.dp_df
    stability_results = st.session_state.get('stability_results', None)
    stat_test_results = st.session_state.get('stat_test_results', None)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # KPI Summary Bar
    st.markdown("### Performance Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Silhouette Score", f"{results.get('silhouette', 0.0):.3f}")
    kpi2.metric("Davies-Bouldin Index", f"{results.get('dbi', 0.0):.3f}")
    kpi3.metric("Calinski-Harabasz", f"{results.get('calinski_harabasz', 0.0):.1f}")
    
    eps = results.get('epsilon', float('inf'))
    eps_str = f"{eps:.2f}" if eps != float('inf') else "∞"
    kpi4.metric("Privacy Budget (ε)", eps_str)
    
    kpi5.metric("Discovered Archetypes", len(results.get('profile', [])))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- TABBED DASHBOARD ---
    tab_list = ["Convergence", "Baselines & Ablation", "Privacy Analysis", 
                "Latent Topology & Personas", "Archetypes & Downloads"]
    if stability_results:
        tab_list.append("Stability & Statistical Tests")
    
    scal_results = st.session_state.get('scal_results', None)
    gen_results = st.session_state.get('gen_results', None)
    if scal_results is not None:
        tab_list.append("Scalability & Generalization")
        
    importance_df = st.session_state.get('importance_df', None)
    if importance_df is not None and not importance_df.empty:
        tab_list.append("Explainability & Interpretability")
    
    tabs = st.tabs(tab_list)
    
    # ── TAB 1: Convergence ──
    with tabs[0]:
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
    with tabs[1]:
        st.markdown("#### Formal Baseline Comparisons")
        st.caption("Segmentation quality and communication costs across FL approaches. Includes Centralized, Intersection-Only, Local-Isolated, TAL-FL (SegFL), and FedProx + TAL.")
        st.dataframe(base_df, use_container_width=True, hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("#### Component Ablation Study")
        st.caption("Impact of removing individual SegFL components (DP, Federation, TAL) on utility and privacy.")
        st.dataframe(abl_df, use_container_width=True, hide_index=True)
    
    # ── TAB 3: Privacy Analysis ──
    with tabs[2]:
        st.markdown("#### Differential Privacy Utility Tradeoff")
        st.caption("Effect of DP noise magnitude (σ) on clustering utility. ε computed via RDP Accountant (Mironov 2017) with δ=10⁻⁵.")
        
        col_dp1, col_dp2 = st.columns(2)
        with col_dp1:
            fig_dp = px.line(
                dp_df, x='DP Sigma (σ)', y='Silhouette',
                template='plotly_white', markers=True,
                labels={'DP Sigma (σ)': 'DP Noise (σ)', 'Silhouette': 'Silhouette Score ↑'}
            )
            fig_dp.update_traces(line=dict(color='#18181b', width=2.5), marker=dict(size=8))
            fig_dp.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_dp, use_container_width=True)
        
        with col_dp2:
            # Filter out inf values for the epsilon plot
            dp_plot = dp_df[dp_df['Privacy Budget (ε)'] < 1e10].copy()
            if not dp_plot.empty:
                fig_eps = px.line(
                    dp_plot, x='DP Sigma (σ)', y='Privacy Budget (ε)',
                    template='plotly_white', markers=True,
                    labels={'Privacy Budget (ε)': 'ε (lower = more private)'}
                )
                fig_eps.update_traces(line=dict(color='#ef4444', width=2.5), marker=dict(size=8))
                fig_eps.update_layout(
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_eps, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Full DP Sweep Results")
        
        # Format the display table, handling inf values
        dp_display = dp_df.copy()
        dp_display['Privacy Budget (ε)'] = dp_display['Privacy Budget (ε)'].apply(
            lambda x: "∞ (No DP)" if x > 1e10 else f"{x:.2f}"
        )
        dp_display['Silhouette'] = dp_display['Silhouette'].apply(lambda x: f"{x:.3f}")
        dp_display['DBI'] = dp_display['DBI'].apply(lambda x: f"{x:.3f}")
        st.dataframe(dp_display, use_container_width=True, hide_index=True)
    
    # ── TAB 4: Latent Topology & Personas ──
    with tabs[3]:
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
            
            all_profile_cols = [c for c in results['profile'].columns if c not in ['Persona', 'Cluster Size', 'Platform ID']]
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
    with tabs[4]:
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
        
        dl_col1, dl_col2, dl_col3 = st.columns([1.2, 1.8, 1.5])
        
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
                
        with dl_col3:
            st.markdown("**Evaluation Report**")
            pdf_bytes = st.session_state.get('pdf_report', None)
            if pdf_bytes is not None:
                st.download_button(
                    label="Download Full PDF Report",
                    data=pdf_bytes,
                    file_name="segfl_evaluation_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("⚠️ PDF Report is not generated. Please re-run.")
    
    # ── TAB 6: Stability & Statistical Tests (conditional) ──
    if stability_results and len(tabs) > 5:
        with tabs[5]:
            st.markdown("#### Multi-Seed Stability Analysis")
            st.caption("Training executed across 10 random seeds. Reports mean ± std of clustering metrics and pairwise NMI/ARI agreement.")
            
            # Summary metrics table
            stab_metrics = pd.DataFrame([{
                'Metric': 'Silhouette Score',
                'Mean': f"{stability_results['silhouette_mean']:.3f}",
                'Std Dev': f"{stability_results['silhouette_std']:.3f}",
            }, {
                'Metric': 'Davies-Bouldin Index',
                'Mean': f"{stability_results['dbi_mean']:.3f}",
                'Std Dev': f"{stability_results['dbi_std']:.3f}",
            }, {
                'Metric': 'Calinski-Harabasz',
                'Mean': f"{stability_results['ch_mean']:.1f}",
                'Std Dev': f"{stability_results['ch_std']:.1f}",
            }, {
                'Metric': 'Pairwise NMI',
                'Mean': f"{stability_results['nmi_mean']:.3f}",
                'Std Dev': f"{stability_results['nmi_std']:.3f}",
            }, {
                'Metric': 'Pairwise ARI',
                'Mean': f"{stability_results['ari_mean']:.3f}",
                'Std Dev': f"{stability_results['ari_std']:.3f}",
            }])
            st.dataframe(stab_metrics, use_container_width=True, hide_index=True)
            
            # Box plots
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Distribution of Silhouette Scores Across Seeds")
            
            box_df = pd.DataFrame({
                'Seed': [f"Seed {42+i}" for i in range(len(stability_results['all_silhouettes']))],
                'Silhouette': stability_results['all_silhouettes']
            })
            fig_box = px.box(box_df, y='Silhouette', template='plotly_white', 
                            points='all', title=None)
            fig_box.update_traces(marker_color='#18181b', line_color='#18181b')
            fig_box.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Silhouette Score"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistical significance test
            if stat_test_results:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### Statistical Significance Test")
                st.caption("Wilcoxon signed-rank test comparing TAL-FL (SegFL) vs Centralized baseline across 10 seeds.")
                
                sig_color = "🟢" if stat_test_results['significant'] else "🔴"
                sig_text = "Statistically Significant" if stat_test_results['significant'] else "Not Statistically Significant"
                
                sig_col1, sig_col2, sig_col3 = st.columns(3)
                sig_col1.metric("Test Statistic (W)", f"{stat_test_results['statistic']:.2f}")
                sig_col2.metric("p-value", f"{stat_test_results['p_value']:.4f}")
                sig_col3.metric("Result (α=0.05)", f"{sig_color} {sig_text}")
                
                st.info("""
                **Interpretation:** The Wilcoxon signed-rank test is a non-parametric paired test that compares whether SegFL's TAL-FL 
                produces statistically significantly different clustering quality than the Centralized baseline. A p-value < 0.05 
                indicates the difference is unlikely due to random variation.
                """)
                
    # ── TAB 7: Scalability & Generalization (conditional) ──
    if "Scalability & Generalization" in tab_list:
        scal_tab_idx = tab_list.index("Scalability & Generalization")
        with tabs[scal_tab_idx]:
            st.markdown("#### System Scalability Analysis")
            st.caption("Performance metrics as the number of clients (tenants) scales from 2 to N.")
            
            scal_df = pd.DataFrame(scal_results)
            
            st.dataframe(scal_df.rename(columns={
                'tenants': 'Client Count',
                'time_seconds': 'Execution Time (s)',
                'silhouette': 'Silhouette Score',
                'dbi': 'Davies-Bouldin Index'
            }), use_container_width=True, hide_index=True)
            
            col_scal1, col_scal2 = st.columns(2)
            with col_scal1:
                fig_time = px.line(
                    scal_df, x='tenants', y='time_seconds',
                    template='plotly_white', markers=True,
                    labels={'tenants': 'Number of Clients', 'time_seconds': 'Execution Time (seconds)'},
                    title="Computational Complexity (Time vs Clients)"
                )
                fig_time.update_traces(line=dict(color='#18181b', width=2.5), marker=dict(size=8))
                fig_time.update_layout(
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_time, use_container_width=True)
                
            with col_scal2:
                fig_qual = px.line(
                    scal_df, x='tenants', y='silhouette',
                    template='plotly_white', markers=True,
                    labels={'tenants': 'Number of Clients', 'silhouette': 'Silhouette Score'},
                    title="Clustering Quality (Silhouette vs Clients)"
                )
                fig_qual.update_traces(line=dict(color='#3b82f6', width=2.5), marker=dict(size=8))
                fig_qual.update_layout(
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_qual, use_container_width=True)
                
            st.divider()
            
            st.markdown("#### Generalization to Unseen Heterogeneous Tenant")
            st.caption("Clustering performance on a completely held-out tenant dataset. The model has never seen this tenant's raw schemas or data during federated training.")
            
            if gen_results:
                gen_col1, gen_col2 = st.columns(2)
                gen_col1.metric("Hold-out Silhouette Score ↑", f"{gen_results['holdout_silhouette']:.3f}")
                gen_col2.metric("Hold-out Davies-Bouldin Index ↓", f"{gen_results['holdout_dbi']:.3f}")
                
                st.info(f"""
                **Zero-Shot Adaptation Mechanism:** 
                To evaluate generalization, SegFL trains its global autoencoder representation model on $C-1$ tenants, leaving the last tenant's data entirely unseen. 
                A local Tenant Adapter Layer (TAL) is then trained for the hold-out tenant for a few local epochs to project its raw features into the shared latent space, while keeping the global model weights **frozen**. 
                The Silhouette Score of **{gen_results['holdout_silhouette']:.3f}** proves that the global model successfully extracts generalizable user behavioral structures that can adapt to new, unseen organizations with zero-shot federated training.
                """)
                
    # ── TAB 9: Explainability & Interpretability (conditional) ──
    if "Explainability & Interpretability" in tab_list:
        exp_tab_idx = tab_list.index("Explainability & Interpretability")
        with tabs[exp_tab_idx]:
            st.markdown("#### Global Surrogate Feature Importance")
            st.caption("Shows which features contribute most to the clustering decisions (determined by training a random forest classifier to predict cluster labels).")
            
            importance_df = st.session_state.importance_df
            enrichment_scores = st.session_state.enrichment_scores
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                fig_imp = px.bar(
                    importance_df, x='Importance', y='Feature',
                    orientation='h', template='plotly_white',
                    title="Global Feature Attributions (Random Forest Surrogate)"
                )
                fig_imp.update_traces(marker_color='#1e293b')
                fig_imp.update_layout(
                    height=380,
                    yaxis={'categoryorder':'total ascending'},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
            with col_exp2:
                st.markdown("##### Cluster Profiling & Feature Alignment")
                st.markdown("""
                The feature importance chart shows the global relevance of each behavioral attribute. 
                Below, we look at the localized **Enrichment Scores** for each discovered archetype.
                These scores measure how many standard deviations the cluster's average behavior deviates from the global average:
                - **Positive Score (+):** Behavior is higher than the global baseline.
                - **Negative Score (-):** Behavior is lower than the global baseline.
                """)
                
            st.divider()
            
            st.markdown("#### Segment Enrichment Scores (Z-Score Deviation)")
            st.caption("Attribution scores indicating how much a persona deviates from the dataset mean on each feature.")
            
            enrich_rows = []
            for cid, scores in enrichment_scores.items():
                persona_name = results['profile'].loc[cid, 'Persona'] if cid in results['profile'].index else f"Cluster {cid}"
                row = {'Cluster': f"Cluster {cid}", 'Persona': persona_name}
                for feat, val in scores.items():
                    row[feat] = f"{val:+.3f} σ"
                enrich_rows.append(row)
                
            enrich_df = pd.DataFrame(enrich_rows)
            st.dataframe(enrich_df, use_container_width=True, hide_index=True)


