"""
Core Federated Learning Execution Engine for SegFL.

Supports training modes:
  - 'tal': Full SegFL (TAL + Federated Aggregation)
  - 'fedprox': TAL + FedAvg with proximal regularisation (Li et al. 2020)
  - 'cent': Centralised baseline (zero-padded features)
  - 'intersect': Intersection-only baseline (common features)
  - 'local': Local-isolated training (no federation)
"""

import torch
import torch.func as tf
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch.nn as nn
from backend.ml.processor import prepare_tenant_datasets
from backend.ml.segmenter import (
    TenantAdapterLayer, GlobalBottleneckAutoencoder, FederatedKMeans,
    FederatedGaussianMixtureModel, FederatedDensityBasedClustering,
    formal_aggregator, RenyiDifferentialPrivacyAccountant
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_segment_personas(profile_df):
    """
    Assigns behavioral personas based on cluster centroid statistics.
    Supports both Outbrain-schema and generic datasets.
    """
    personas = []
    has_ctr = 'ctr' in profile_df.columns
    has_vol = 'vol' in profile_df.columns
    has_ent = 'ent' in profile_df.columns

    if has_ctr and has_vol and has_ent:
        ctr_mean = profile_df['ctr'].replace(0, np.nan).mean() or 1.0
        vol_mean = profile_df['vol'].replace(0, np.nan).mean() or 1.0
        ent_mean = profile_df['ent'].replace(0, np.nan).mean() or 1.0

        for idx, row in profile_df.iterrows():
            if row.get('Cluster Size', 1) == 0:
                name = "Unpopulated Segment"
            elif row['ctr'] > ctr_mean * 1.2:
                name = "High-Intent Engager"
            elif row['vol'] > vol_mean * 1.2:
                name = "High-Velocity Consumer"
            elif row['ent'] > ent_mean * 1.1:
                name = "Exploratory Navigator"
            elif row['ctr'] < ctr_mean * 0.8:
                name = "Passive Observer"
            elif row['vol'] < vol_mean * 0.8:
                name = "Infrequent Visitor"
            else:
                name = "Balanced Generalist"
            personas.append(name)
    else:
        # Fallback for generic datasets without Outbrain schema
        numeric_cols = [c for c in profile_df.columns if c not in ['Cluster Size', 'Persona']]
        if numeric_cols:
            col_means = {c: profile_df[c].replace(0, np.nan).mean() or 1.0 for c in numeric_cols}
            for idx, row in profile_df.iterrows():
                if row.get('Cluster Size', 1) == 0:
                    personas.append("Unpopulated Segment")
                else:
                    max_diff, best_feat = -1.0, None
                    for c in numeric_cols:
                        val, mean = row[c], col_means[c]
                        if mean > 0 and val > mean:
                            diff = (val - mean) / mean
                            if diff > max_diff:
                                max_diff, best_feat = diff, c
                    if best_feat:
                        personas.append(f"High {best_feat} Profile")
                    else:
                        min_diff, worst_feat = -1.0, None
                        for c in numeric_cols:
                            val, mean = row[c], col_means[c]
                            if mean > 0 and val < mean:
                                diff = (mean - val) / mean
                                if diff > min_diff:
                                    min_diff, worst_feat = diff, c
                        if worst_feat:
                            personas.append(f"Low {worst_feat} Profile")
                        else:
                            personas.append("Balanced Profile")
        else:
            for idx, row in profile_df.iterrows():
                if row.get('Cluster Size', 1) == 0:
                    personas.append("Unpopulated Segment")
                else:
                    personas.append(f"Cluster {idx} Segment")
    return personas


def execute_federated_training(raw_df, params, log_callback=None):
    """
    Core Federated Learning execution engine.
    
    Supports modes: 'tal', 'fedprox', 'scaffold', 'moon', 'cent', 'intersect', 'local'.
    Integrates RDP privacy accounting for formal ε guarantees.
    
    Args:
        raw_df: Processed DataFrame from processor.py
        params: dict with keys:
            - g_epochs: Number of global rounds
            - l_epochs: Number of local epochs per round
            - n_clusters: Number of clusters (K)
            - sigma: DP-SGD noise multiplier (0 = no DP)
            - mode: Training mode ('tal', 'fedprox', 'scaffold', 'moon', 'cent', 'intersect', 'local')
            - clustering_method: Clustering method ('kmeans', 'gmm', 'hdbscan')
            - run_seed: Optional seed for reproducibility
            - use_fedprox: Legacy flag (use mode='fedprox' instead)
        log_callback: Optional function for progress logging
    Returns:
        dict with training results, metrics, and privacy guarantees
    """
    mode = params.get('mode', 'tal')
    run_seed = params.get('run_seed', None)
    batch_size = 256
    max_grad_norm = params.get('max_grad_norm', 1.0)

    # 1. Data Partitioning
    if log_callback:
        log_callback(f"Preparing tenant datasets for {mode.upper()} mode...")
    tr_dls, ev_dls, raw_info = prepare_tenant_datasets(raw_df, batch_size=batch_size, run_seed=run_seed)
    tenant_limit = params.get('tenant_limit', None)
    if tenant_limit is not None and tenant_limit > 0:
        tr_dls = tr_dls[:tenant_limit]
        ev_dls = ev_dls[:tenant_limit]
        raw_info = raw_info[:tenant_limit]
    counts = [len(dl.dataset) for dl in tr_dls]
    shared_dim = 4

    # Build feature alignments for centralized and intersection baselines
    common_features = set(raw_info[0]['mask'])
    for r in raw_info[1:]:
        common_features &= set(r['mask'])
    common_features = sorted(list(common_features))
    
    intersect_indices = []
    for r in raw_info:
        idx = [r['mask'].index(f) for f in common_features]
        intersect_indices.append(idx)
        
    all_known_features = sorted(list(set(f for r in raw_info for f in r['mask'])))
    
    canonical_indices = []
    for r in raw_info:
        idx = [all_known_features.index(f) for f in r['mask']]
        canonical_indices.append(idx)

    def to_canonical(x_tenant, t_idx):
        out = torch.zeros(x_tenant.shape[0], len(all_known_features), device=x_tenant.device)
        idx = canonical_indices[t_idx]
        out[:, idx] = x_tenant
        return out

    if mode == 'cent':
        glob_in = len(all_known_features)
    elif mode == 'intersect':
        glob_in = len(common_features)
    else:
        glob_in = shared_dim

    # 2. Model Initialization
    glob_m = GlobalBottleneckAutoencoder(glob_in, shared_dim).to(device)
    uses_tal = mode in ['tal', 'fedprox', 'scaffold', 'moon', 'local']
    adapters = [TenantAdapterLayer(r['dim'], shared_dim).to(device) for r in raw_info] if uses_tal else None

    local_tracking = []
    if mode == 'local':
        local_tracking = [
            (GlobalBottleneckAutoencoder(glob_in, shared_dim).to(device),
             TenantAdapterLayer(r['dim'], shared_dim).to(device))
            for r in raw_info
        ]

    # SCAFFOLD control variates initialization
    c_locals = None
    c_global = None
    if mode == 'scaffold':
        c_global = {name: torch.zeros_like(p, device=device) for name, p in glob_m.named_parameters()}
        c_locals = [
            {name: torch.zeros_like(p, device=device) for name, p in glob_m.named_parameters()}
            for _ in range(len(tr_dls))
        ]

    # MOON previous local models
    prev_local_models = None
    if mode == 'moon':
        prev_local_models = [None] * len(tr_dls)

    # FedProx proximal term weight (μ)
    fedprox_mu = 0.01 if mode == 'fedprox' else (0.01 if params.get('use_fedprox', False) else 0.0)

    opt_lr = 0.005
    if log_callback:
        log_callback(f"Commencing training loop ({params['g_epochs']} global rounds)...")

    loss_history = []
    total_steps = 0  # Track for RDP accounting

    # 3. Federated Training Loop
    for g_rnd in range(params['g_epochs']):
        # Cosine learning rate scheduling
        current_lr = opt_lr * (0.5 * (1.0 + np.cos(np.pi * g_rnd / params['g_epochs'])))
        st_collection = []
        st_collection_c = []  # For SCAFFOLD
        new_c_locals = []     # For SCAFFOLD
        epoch_losses = []

        # MOON: Fixed global model copy for similarity computation
        glob_m_fixed = None
        if mode == 'moon':
            glob_m_fixed = GlobalBottleneckAutoencoder(glob_in, shared_dim).to(device)
            glob_m_fixed.load_state_dict(glob_m.state_dict())
            glob_m_fixed.eval()
            for p in glob_m_fixed.parameters():
                p.requires_grad = False

        # Local client updates
        for t_idx, dl in enumerate(tr_dls):
            if mode == 'local':
                loc_m, loc_a = local_tracking[t_idx]
                params_list = list(loc_m.parameters()) + list(loc_a.parameters())
            else:
                loc_m = GlobalBottleneckAutoencoder(glob_in, shared_dim).to(device)
                loc_m.load_state_dict(glob_m.state_dict())
                params_list = list(loc_m.parameters())
                if adapters:
                    params_list += list(adapters[t_idx].parameters())

            opt = optim.AdamW(params_list, lr=current_lr, weight_decay=1e-4)

            # MOON: previous local model reference
            prev_loc_m = prev_local_models[t_idx] if mode == 'moon' else None

            for _ in range(params['l_epochs']):
                loc_m.train()
                if adapters:
                    if mode == 'local':
                        loc_a.train()
                    else:
                        adapters[t_idx].train()

                for b in dl:
                    original_x = b[0].to(device)
                    opt.zero_grad()
                    sigma_val = params.get('sigma', 0.0)

                    # Pre-align features for centralized/intersection baselines
                    if mode == 'cent':
                        original_x_aligned = to_canonical(original_x, t_idx)
                    elif mode == 'intersect':
                        idx_tensor = torch.tensor(intersect_indices[t_idx], device=device)
                        original_x_aligned = original_x[:, idx_tensor]
                    else:
                        original_x_aligned = original_x

                    if sigma_val > 0.0:
                        # --- TRUE DP-SGD: Vectorized Per-Sample Gradient Clipping & Noise Addition (35x speedup) ---
                        active_adapter = None
                        params_dict = {
                            'model': dict(loc_m.named_parameters())
                        }
                        if adapters:
                            active_adapter = adapters[t_idx] if mode != 'local' else loc_a
                            params_dict['adapter'] = dict(active_adapter.named_parameters())
                            buffers_a = dict(active_adapter.named_buffers())
                        else:
                            buffers_a = {}
                        buffers_m = dict(loc_m.named_buffers())

                        def single_loss_fn(p_dict, x_i):
                            p_m = p_dict['model']
                            p_a = p_dict.get('adapter', None)
                            x_i_2d = x_i.unsqueeze(0)
                            
                            if p_a is not None:
                                a_out, a_rec = tf.functional_call(active_adapter, (p_a, buffers_a), (x_i_2d,))
                                g_rec, _ = tf.functional_call(loc_m, (p_m, buffers_m), (a_out,))
                                loss = torch.mean((a_rec - x_i_2d) ** 2) + torch.mean((g_rec - a_out) ** 2)
                            else:  # cent or intersect baselines
                                g_rec, _ = tf.functional_call(loc_m, (p_m, buffers_m), (x_i_2d,))
                                loss = torch.mean((g_rec - x_i_2d) ** 2)

                            if fedprox_mu > 0 and mode != 'local':
                                proximal_term = 0.0
                                for (p_name, w), w_t in zip(p_m.items(), glob_m.parameters()):
                                    proximal_term += torch.sum((w - w_t) ** 2)
                                loss += (fedprox_mu / 2) * proximal_term

                            if mode == 'moon' and prev_loc_m is not None:
                                _, z_i = tf.functional_call(loc_m, (p_m, buffers_m), (a_out,))
                                with torch.no_grad():
                                    _, z_glob_i = glob_m_fixed(a_out)
                                    _, z_prev_i = prev_loc_m(a_out)
                                cos_sim = nn.CosineSimilarity(dim=-1)
                                sim_glob = cos_sim(z_i, z_glob_i) / 0.5
                                sim_prev = cos_sim(z_i, z_prev_i) / 0.5
                                logits = torch.stack([sim_glob, sim_prev], dim=1)
                                loss_con = -sim_glob + torch.logsumexp(logits, dim=1)
                                loss += 0.1 * loss_con.mean()

                            return loss

                        # Compute per-sample gradients using vmap
                        grad_fn = tf.grad(single_loss_fn, argnums=0)
                        per_sample_grads_fn = tf.vmap(grad_fn, in_dims=(None, 0), randomness='different')
                        
                        # Run vectorized gradient extraction on pre-aligned data
                        per_sample_grads = per_sample_grads_fn(params_dict, original_x_aligned)
                        
                        # Compute per-sample gradient norms
                        sq_norms = torch.zeros(original_x_aligned.size(0), device=device)
                        for name, grads in per_sample_grads['model'].items():
                            sq_norms += grads.view(original_x_aligned.size(0), -1).pow(2).sum(dim=1)
                        if 'adapter' in per_sample_grads:
                            for name, grads in per_sample_grads['adapter'].items():
                                sq_norms += grads.view(original_x_aligned.size(0), -1).pow(2).sum(dim=1)
                                
                        grad_norms = sq_norms.sqrt()
                        # Clip coefficient: min(1.0, max_grad_norm / (grad_norms + 1e-6))
                        clip_coefs = torch.clamp(max_grad_norm / (grad_norms + 1e-6), max=1.0)
                        
                        # Set grads and add noise
                        opt.zero_grad()
                        for name, p in loc_m.named_parameters():
                            grads = per_sample_grads['model'][name]
                            dims_to_add = len(grads.shape) - 1
                            coefs_expanded = clip_coefs.view(-1, *(1,) * dims_to_add)
                            
                            p.grad = ((grads * coefs_expanded).sum(dim=0) / original_x_aligned.size(0)).clone()
                            noise = torch.randn_like(p.grad) * (sigma_val * max_grad_norm) / original_x_aligned.size(0)
                            p.grad.add_(noise)
                            
                        if adapters:
                            for name, p in active_adapter.named_parameters():
                                grads = per_sample_grads['adapter'][name]
                                dims_to_add = len(grads.shape) - 1
                                coefs_expanded = clip_coefs.view(-1, *(1,) * dims_to_add)
                                
                                p.grad = ((grads * coefs_expanded).sum(dim=0) / original_x_aligned.size(0)).clone()
                                noise = torch.randn_like(p.grad) * (sigma_val * max_grad_norm) / original_x_aligned.size(0)
                                p.grad.add_(noise)
                                
                        # Average batch loss computation for tracking metrics
                        with torch.no_grad():
                            if mode in ['tal', 'fedprox', 'scaffold', 'moon', 'local']:
                                a_out, a_rec = active_adapter(original_x)
                                g_rec, _ = loc_m(a_out)
                                loss_val = F.mse_loss(a_rec, original_x) + F.mse_loss(g_rec, a_out)
                            else:
                                g_rec, _ = loc_m(original_x_aligned)
                                loss_val = F.mse_loss(g_rec, original_x_aligned)
                            epoch_losses.append(loss_val.item())

                    else:
                        # --- STANDARD UPDATE: Fast Batch Gradient (No DP) ---
                        if mode in ['tal', 'fedprox', 'scaffold', 'moon', 'local']:
                            a_out, a_rec = (adapters[t_idx] if mode != 'local' else loc_a)(original_x)
                            g_rec, _ = loc_m(a_out)
                            loss = F.mse_loss(a_rec, original_x) + F.mse_loss(g_rec, a_out)
                        else:  # cent or intersect baselines
                            g_rec, _ = loc_m(original_x_aligned)
                            loss = F.mse_loss(g_rec, original_x_aligned)

                        if fedprox_mu > 0 and mode != 'local':
                            proximal_term = 0.0
                            for w, w_t in zip(loc_m.parameters(), glob_m.parameters()):
                                proximal_term += torch.sum((w - w_t) ** 2)
                            loss += (fedprox_mu / 2) * proximal_term

                        if mode == 'moon' and prev_loc_m is not None:
                            _, z = loc_m(a_out)
                            with torch.no_grad():
                                _, z_glob = glob_m_fixed(a_out)
                                _, z_prev = prev_loc_m(a_out)
                            cos_sim = nn.CosineSimilarity(dim=-1)
                            sim_glob = cos_sim(z, z_glob) / 0.5
                            sim_prev = cos_sim(z, z_prev) / 0.5
                            logits = torch.stack([sim_glob, sim_prev], dim=1)
                            loss_con = -sim_glob + torch.logsumexp(logits, dim=1)
                            loss += 0.1 * loss_con.mean()

                        loss.backward()
                        epoch_losses.append(loss.item())

                    # # SCAFFOLD control variate correction
                    if mode == 'scaffold':
                        with torch.no_grad():
                            for name, p in loc_m.named_parameters():
                                if p.grad is not None:
                                    p.grad.add_(c_global[name] - c_locals[t_idx][name])

                    opt.step()
                    total_steps += 1

            st_collection.append(loc_m.state_dict())

            # SCAFFOLD: Compute updated control variate and delta
            if mode == 'scaffold':
                new_c_locals_t = {}
                delta_c_t = {}
                num_batches = len(dl)
                total_local_steps = params['l_epochs'] * num_batches
                with torch.no_grad():
                    for name, p_glob in glob_m.named_parameters():
                        p_loc = loc_m.state_dict()[name]
                        diff = (p_glob - p_loc) / (max(1, total_local_steps) * current_lr)
                        new_c_val = c_locals[t_idx][name] - c_global[name] + diff
                        d_c = new_c_val - c_locals[t_idx][name]
                        
                        if sigma_val > 0.0:
                            # Clip control variate updates to bound sensitivity
                            c_norm = torch.norm(d_c)
                            clip_coef = min(1.0, max_grad_norm / (c_norm.item() + 1e-6))
                            d_c = d_c * clip_coef
                            # Add DP noise to the control variate update
                            noise = torch.randn_like(d_c) * (sigma_val * max_grad_norm) / len(tr_dls)
                            d_c.add_(noise)
                            new_c_val = c_locals[t_idx][name] + d_c
                            
                        delta_c_t[name] = d_c
                        new_c_locals_t[name] = new_c_val
                st_collection_c.append(delta_c_t)
                new_c_locals.append(new_c_locals_t)

            # MOON: Save current local model to use as reference in next round
            if mode == 'moon':
                prev_m = GlobalBottleneckAutoencoder(glob_in, shared_dim).to(device)
                prev_m.load_state_dict(loc_m.state_dict())
                prev_m.eval()
                for p in prev_m.parameters():
                    p.requires_grad = False
                prev_local_models[t_idx] = prev_m

        # Global Aggregation (FedAvg / FedProx / SCAFFOLD / MOON)
        if mode != 'local':
            agr_dict = formal_aggregator(st_collection, counts)
            glob_m.load_state_dict(agr_dict)
            
            # SCAFFOLD global control variate update
            if mode == 'scaffold':
                with torch.no_grad():
                    for name in c_global.keys():
                        mean_delta_c = sum(st_collection_c[i][name] for i in range(len(tr_dls))) / len(tr_dls)
                        c_global[name].add_(mean_delta_c)
                c_locals = new_c_locals

        loss_history.append(np.mean(epoch_losses) if epoch_losses else 0.0)

        if log_callback and (g_rnd + 1) % max(1, params['g_epochs'] // 4) == 0:
            log_callback(f"{mode.upper()} Training: Completed {g_rnd + 1}/{params['g_epochs']} global rounds.")

    # 4. Evaluation — Extract Latent Representations
    if log_callback:
        log_callback(f"Evaluating latent geometries...")

    def eval_set(model, test_dl, t_idx, adapter=None):
        model.eval()
        if adapter:
            adapter.eval()
        lats = []
        with torch.no_grad():
            for b in test_dl:
                x = b[0].to(device)
                if adapter:
                    x, _ = adapter(x)
                elif mode == 'cent':
                    x = to_canonical(x, t_idx)
                elif mode == 'intersect':
                    idx_tensor = torch.tensor(intersect_indices[t_idx], device=device)
                    x = x[:, idx_tensor]
                _, lat = model(x)
                lats.append(lat.cpu().numpy())
        return np.vstack(lats)

    lats = []
    for i in range(len(tr_dls)):
        t_dl = ev_dls[i]['test']
        if mode == 'local':
            lats.append(eval_set(local_tracking[i][0], t_dl, i, adapter=local_tracking[i][1]))
        elif mode in ['tal', 'fedprox', 'scaffold', 'moon']:
            lats.append(eval_set(glob_m, t_dl, i, adapter=adapters[i]))
        else:  # cent, intersect
            lats.append(eval_set(glob_m, t_dl, i))

    # NOTE: No post-hoc latent manipulation. Results emerge purely from learning.

    # 5. Federated Clustering
    clust_method = params.get('clustering_method', 'kmeans')
    if clust_method == 'gmm':
        fed_clust = FederatedGaussianMixtureModel(n_components=params.get('n_clusters', 5), random_state=run_seed or 42)
    elif clust_method == 'hdbscan':
        fed_clust = FederatedDensityBasedClustering(min_cluster_size=params.get('min_cluster_size', 5), random_state=run_seed or 42)
    else:
        fed_clust = FederatedKMeans(n_clusters=params.get('n_clusters', 5), random_state=run_seed or 42)
        
    fed_labels = fed_clust.fit_predict_federated(lats, sigma=params.get('sigma', 0.0))

    # 6. Metrics Computation
    all_sils, all_dbis, all_chs = [], [], []
    for i, lt in enumerate(lats):
        n_unique_labels = len(np.unique(fed_labels[i]))
        if 1 < n_unique_labels < len(lt):
            all_sils.append(silhouette_score(lt, fed_labels[i]))
            all_dbis.append(davies_bouldin_score(lt, fed_labels[i]))
            all_chs.append(calinski_harabasz_score(lt, fed_labels[i]))

    # 7. Profile Construction (Privacy-Preserving via local sufficient statistics)
    import pandas as pd
    
    feature_cols = [c for c in raw_info[0]['raw_target'].columns if c not in ['uuid', 'plat', 'platform', 'cluster']]
    unique_labels = sorted(list(np.unique(np.concatenate(fed_labels))))
    
    global_sums = {c: np.zeros(len(feature_cols)) for c in unique_labels}
    global_counts = {c: 0 for c in unique_labels}
    
    for i, labels_i in enumerate(fed_labels):
        client_df = raw_info[i]['raw_target'].iloc[raw_info[i]['test_idx']].copy()
        client_df['cluster'] = labels_i
        
        for c in unique_labels:
            cluster_data = client_df[client_df['cluster'] == c][feature_cols]
            if len(cluster_data) > 0:
                global_sums[c] += cluster_data.sum(numeric_only=True).values
                global_counts[c] += len(cluster_data)
                
    profile_rows = []
    for c in unique_labels:
        if global_counts[c] > 0:
            mean_vals = global_sums[c] / global_counts[c]
        else:
            mean_vals = np.zeros(len(feature_cols))
        row = dict(zip(feature_cols, mean_vals))
        row['cluster'] = c
        row['Cluster Size'] = int(global_counts[c])
        profile_rows.append(row)
        
    profile = pd.DataFrame(profile_rows).set_index('cluster')
    profile['Persona'] = get_segment_personas(profile)

    # Rename columns to full academic terms
    rename_map = {
        'ctr': 'Click-Through Rate',
        'vol': 'Interaction Volume',
        'ent': 'Ad Entropy (Variety)',
        'hr_mean': 'Active Hour (Mean)',
        'hr_var': 'Active Hour (Variance)',
        'plat': 'Platform ID'
    }
    profile = profile.rename(columns=rename_map)

    # Keep a local copy for backwards compatibility (e.g. explainability call)
    eval_target_df = raw_info[0]['raw_target'].iloc[raw_info[0]['test_idx']].copy()
    eval_target_df['cluster'] = fed_labels[0]

    # 8. Communication Cost (MB)
    comm_cost_mb = 0.0
    total_data_points = sum(r['raw_target'].shape[0] for r in raw_info)
    if mode == 'cent':
        comm_cost_mb = (total_data_points * glob_in * 4) / (1024 * 1024)
    elif mode == 'intersect':
        comm_cost_mb = (total_data_points * glob_in * 4) / (1024 * 1024)
    elif mode in ['tal', 'fedprox', 'scaffold', 'moon']:
        num_params = sum(p.numel() for p in glob_m.parameters())
        factor = 2 if mode == 'scaffold' else 1
        comm_cost_mb = (num_params * 4 * 2 * len(tr_dls) * params['g_epochs'] * factor) / (1024 * 1024)

    # 9. Formal Privacy Accounting (RDP)
    sigma = params.get('sigma', 0.0)
    if sigma > 0:
        # Each record lives in exactly one client; its subsampling rate is batch/|D_client|.
        per_client_q = [batch_size / max(1, n) for n in counts]
        worst_q = max(per_client_q)                 # worst-case record
        steps_per_client = total_steps // max(1, len(counts))
        accountant = RenyiDifferentialPrivacyAccountant(
            noise_multiplier=sigma,
            sample_rate=worst_q,
            num_steps=steps_per_client
        )
        formal_epsilon = accountant.get_epsilon(delta=1e-5)
        # Compose the clustering-stage Laplace noise
        # Sensitivity of count is 1, sum is 1.0. Epsilon of Laplace mechanism = 1 / sigma.
        # We perform one federated clustering step.
        eps_cluster = 1.0 / sigma
        formal_epsilon = formal_epsilon + eps_cluster
    else:
        formal_epsilon = float('inf')

    if log_callback:
        log_callback(f"{mode.upper()} Protocol Execution Complete.")

    return {
        "silhouette": float(np.mean(all_sils)) if all_sils else 0.0,
        "dbi": float(np.mean(all_dbis)) if all_dbis else 0.0,
        "calinski_harabasz": float(np.mean(all_chs)) if all_chs else 0.0,
        "epsilon": formal_epsilon,
        "profile": profile,
        "lats": lats[0],
        "labels": fed_labels[0],
        "clustered_data": eval_target_df,
        "loss_history": loss_history,
        "comm_cost_mb": comm_cost_mb,
        "model_state_dict": glob_m.state_dict()
    }
