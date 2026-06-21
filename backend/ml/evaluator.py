"""
Scientific Evaluation Module for SegFL.

Provides:
  - Multi-metric clustering evaluation (Silhouette, DBI, Calinski-Harabasz)
  - Cross-clustering agreement (NMI, ARI)
  - Multi-seed stability analysis with pairwise NMI/ARI
  - Wilcoxon signed-rank statistical significance testing
"""

import numpy as np
import torch
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    normalized_mutual_info_score, adjusted_rand_score
)
from scipy.stats import wilcoxon


def compute_all_metrics(latents, labels):
    """
    Compute comprehensive internal clustering quality metrics.
    
    Args:
        latents: ndarray (n_samples, n_features) — latent representations
        labels: ndarray (n_samples,) — cluster assignments
    Returns:
        dict with silhouette, dbi, calinski_harabasz scores
    """
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(labels):
        return {
            'silhouette': 0.0,
            'dbi': float('inf'),
            'calinski_harabasz': 0.0
        }
    return {
        'silhouette': float(silhouette_score(latents, labels)),
        'dbi': float(davies_bouldin_score(latents, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(latents, labels))
    }


def compute_nmi_ari(labels_a, labels_b):
    """
    Compute Normalized Mutual Information and Adjusted Rand Index
    between two clustering assignments.
    
    Args:
        labels_a, labels_b: ndarray — cluster assignments (same length)
    Returns:
        dict with nmi and ari scores
    """
    min_len = min(len(labels_a), len(labels_b))
    a, b = labels_a[:min_len], labels_b[:min_len]
    return {
        'nmi': float(normalized_mutual_info_score(a, b)),
        'ari': float(adjusted_rand_score(a, b))
    }


def stability_analysis(raw_df, params, execute_fn, n_seeds=10, log_callback=None):
    """
    Run federated training across multiple random seeds and compute
    stability metrics (mean ± std) plus pairwise clustering agreement.
    
    Args:
        raw_df: Processed DataFrame from processor.py
        params: Training parameters dict
        execute_fn: The execute_federated_training function
        n_seeds: Number of random seeds to evaluate
        log_callback: Optional log function for progress updates
    Returns:
        dict with mean/std of all metrics and pairwise NMI/ARI
    """
    all_silhouettes = []
    all_dbis = []
    all_labels = []
    all_epsilons = []
    all_ch_scores = []

    for i in range(n_seeds):
        seed = 42 + i
        np.random.seed(seed)
        torch.manual_seed(seed)

        if log_callback:
            log_callback(f"Stability Run {i+1}/{n_seeds} (seed={seed})...")

        res = execute_fn(raw_df, {**params, 'run_seed': seed}, log_callback=None)
        all_silhouettes.append(res['silhouette'])
        all_dbis.append(res['dbi'])
        all_labels.append(res['labels'])
        all_epsilons.append(res['epsilon'])
        all_ch_scores.append(res.get('calinski_harabasz', 0.0))

    # Pairwise NMI/ARI across all seed pairs
    nmi_scores, ari_scores = [], []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            scores = compute_nmi_ari(all_labels[i], all_labels[j])
            nmi_scores.append(scores['nmi'])
            ari_scores.append(scores['ari'])

    return {
        'silhouette_mean': float(np.mean(all_silhouettes)),
        'silhouette_std': float(np.std(all_silhouettes)),
        'dbi_mean': float(np.mean(all_dbis)),
        'dbi_std': float(np.std(all_dbis)),
        'ch_mean': float(np.mean(all_ch_scores)),
        'ch_std': float(np.std(all_ch_scores)),
        'nmi_mean': float(np.mean(nmi_scores)) if nmi_scores else 0.0,
        'nmi_std': float(np.std(nmi_scores)) if nmi_scores else 0.0,
        'ari_mean': float(np.mean(ari_scores)) if ari_scores else 0.0,
        'ari_std': float(np.std(ari_scores)) if ari_scores else 0.0,
        'epsilon_mean': float(np.mean(all_epsilons)),
        'all_silhouettes': all_silhouettes,
        'all_dbis': all_dbis,
    }


def wilcoxon_test(scores_a, scores_b, alternative='two-sided'):
    """
    Wilcoxon signed-rank test for paired samples.
    Non-parametric test appropriate for small sample sizes (n ≥ 5).
    
    Args:
        scores_a, scores_b: lists of paired metric scores
        alternative: 'two-sided', 'greater', or 'less'
    Returns:
        dict with test statistic, p-value, and significance flag
    """
    a, b = np.array(scores_a), np.array(scores_b)
    if len(a) < 5 or len(b) < 5:
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
    
    # Remove identical pairs (Wilcoxon requires differences ≠ 0)
    diffs = a - b
    if np.all(diffs == 0):
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
    
    try:
        stat, p = wilcoxon(a, b, alternative=alternative, zero_method='wilcox')
        return {
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < 0.05
        }
    except Exception:
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}


def scalability_analysis(raw_df, params, execute_fn, tenant_counts=[2, 3, 4, 5, 6], log_callback=None):
    """
    Measure system execution time and clustering quality (Silhouette)
    as the number of tenants scales.
    """
    import time
    results = []
    for count in tenant_counts:
        if log_callback:
            log_callback(f"Running scalability test for {count} tenants...")
        start_time = time.time()
        res = execute_fn(raw_df, {**params, 'tenant_limit': count}, log_callback=None)
        elapsed = time.time() - start_time
        results.append({
            'tenants': count,
            'time_seconds': elapsed,
            'silhouette': res['silhouette'],
            'dbi': res['dbi']
        })
    return results


def generalization_test(raw_df, params, execute_fn, log_callback=None):
    """
    Evaluate generalization to an unseen hold-out tenant:
      1. Train global model on all tenants except the last one.
      2. Freeze global model.
      3. Train a new adapter for the hold-out tenant on its local training set.
      4. Evaluate clustering quality (Silhouette) on the hold-out tenant's test set.
    """
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
    from backend.ml.processor import prepare_tenant_datasets
    from backend.ml.segmenter import TAL_Adapter, GlobalBottleneckAE
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if log_callback:
        log_callback("Preparing datasets for generalization test (holding out the last tenant)...")
        
    tr_dls, ev_dls, raw_info = prepare_tenant_datasets(raw_df, batch_size=256, run_seed=params.get('run_seed', 42))
    num_tenants = len(tr_dls)
    if num_tenants < 2:
        if log_callback:
            log_callback("⚠️ Insufficient tenants to perform generalization test (need at least 2).")
        return {'holdout_silhouette': 0.0, 'holdout_dbi': float('inf')}
        
    if log_callback:
        log_callback(f"Training global model on {num_tenants-1} tenants (excluding tenant {num_tenants-1})...")
    res_train = execute_fn(raw_df, {**params, 'tenant_limit': num_tenants - 1}, log_callback=None)
    
    mode = params.get('mode', 'tal')
    uses_tal = mode in ['tal', 'fedprox', 'scaffold', 'moon', 'local']
    
    glob_in_model = 4 if uses_tal else max(r['dim'] for r in raw_info[:-1])
    shared_dim = 4
    
    glob_m = GlobalBottleneckAE(glob_in_model, shared_dim).to(device)
    glob_m.load_state_dict(res_train['model_state_dict'])
    glob_m.eval()
    for p in glob_m.parameters():
        p.requires_grad = False
        
    holdout_idx = num_tenants - 1
    holdout_info = raw_info[holdout_idx]
    holdout_dim = holdout_info['dim']
    
    if log_callback:
        log_callback(f"Initializing and training local adapter for hold-out tenant with feature dim {holdout_dim}...")
        
    adapter = TAL_Adapter(holdout_dim, shared_dim).to(device)
    optimizer = optim.AdamW(adapter.parameters(), lr=0.005, weight_decay=1e-4)
    
    holdout_tr_dl = tr_dls[holdout_idx]
    for epoch in range(5):
        adapter.train()
        for b in holdout_tr_dl:
            x = b[0].to(device)
            optimizer.zero_grad()
            a_out, a_rec = adapter(x)
            g_rec, _ = glob_m(a_out)
            loss = F.mse_loss(a_rec, x) + F.mse_loss(g_rec, a_out)
            loss.backward()
            optimizer.step()
            
    if log_callback:
        log_callback("Evaluating hold-out tenant on test set...")
        
    adapter.eval()
    from sklearn.preprocessing import StandardScaler
    local_df = holdout_info['raw_target']
    active_feats = holdout_info['mask']
    feats_raw = StandardScaler().fit_transform(local_df[active_feats].values)
    
    lats = []
    with torch.no_grad():
        # Predict in chunks to save GPU memory
        for chunk in np.array_split(feats_raw, max(1, len(feats_raw) // 256)):
            x_b = torch.FloatTensor(chunk).to(device)
            a_out_b, _ = adapter(x_b)
            _, lat_b = glob_m(a_out_b)
            lats.append(lat_b.cpu().numpy())
            
    latents = np.vstack(lats)
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    n_clusters = params.get('n_clusters', 5)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = km.fit_predict(latents)
    
    n_unique = len(np.unique(labels))
    if 1 < n_unique < len(latents):
        sil = float(silhouette_score(latents, labels))
        dbi = float(davies_bouldin_score(latents, labels))
    else:
        sil = 0.0
        dbi = float('inf')
        
    if log_callback:
        log_callback(f"Hold-out Tenant Evaluation Complete: Silhouette={sil:.3f}, DBI={dbi:.3f}")
        
    return {
        'holdout_silhouette': sil,
        'holdout_dbi': dbi
    }

