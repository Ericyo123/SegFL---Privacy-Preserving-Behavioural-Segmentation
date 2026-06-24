import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.ml.processor import process_csv
from backend.ml.core_engine import execute_federated_training
from backend.ml.evaluator import scalability_analysis, generalization_test

def main():
    print("Initializing local verification check...")
    # Create small dummy dataframe with platform field
    np.random.seed(42)
    n_samples = 200
    data = {
        'uuid': [f"user_{i}" for i in range(n_samples)],
        'clicked': np.random.randint(0, 2, n_samples),
        'ad_id': np.random.randint(100, 150, n_samples),
        'timestamp': np.random.randint(1450000000, 1460000000, n_samples),
        'platform': np.random.choice([1, 2, 3], n_samples)
    }
    df = pd.DataFrame(data)
    
    # Run extractor to get semantic features
    from backend.ml.processor import _extract_behavioural_heuristics
    feats = _extract_behavioural_heuristics(df)
    
    params = {
        'g_epochs': 2,
        'l_epochs': 2,
        'n_clusters': 3,
        'sigma': 0.1,
    }
    
    print("\n--- Testing SCAFFOLD baseline ---")
    res_scaffold = execute_federated_training(feats, {**params, 'mode': 'scaffold'}, log_callback=print)
    print(f"SCAFFOLD success! Silhouette={res_scaffold['silhouette']:.3f}, Epsilon={res_scaffold['epsilon']:.3f}")
    
    print("\n--- Testing MOON baseline ---")
    res_moon = execute_federated_training(feats, {**params, 'mode': 'moon'}, log_callback=print)
    print(f"MOON success! Silhouette={res_moon['silhouette']:.3f}, Epsilon={res_moon['epsilon']:.3f}")
    
    print("\n--- Testing GMM clustering ---")
    res_gmm = execute_federated_training(feats, {**params, 'mode': 'tal', 'clustering_method': 'gmm'}, log_callback=print)
    print(f"GMM success! Silhouette={res_gmm['silhouette']:.3f}")
    
    print("\n--- Testing HDBSCAN clustering ---")
    res_hdb = execute_federated_training(feats, {**params, 'mode': 'tal', 'clustering_method': 'hdbscan'}, log_callback=print)
    print(f"HDBSCAN success! Silhouette={res_hdb['silhouette']:.3f}")
    
    print("\n--- Testing Scalability Analysis ---")
    scal = scalability_analysis(feats, {**params, 'mode': 'tal'}, execute_federated_training, tenant_counts=[2, 3], log_callback=print)
    print("Scalability results:", scal)
    
    print("\n--- Testing Generalization hold-out test ---")
    gen = generalization_test(feats, {**params, 'mode': 'tal'}, execute_federated_training, log_callback=print)
    print("Generalization results:", gen)
    
    print("\n--- Testing PDF Report Generation ---")
    from backend.ml.report_generator import build_pdf_report
    base_df = pd.DataFrame([{
        'Mode': 'TAL-FL', 
        'Silhouette ↑': '0.800', 
        'DBI ↓': '0.150', 
        'Calinski-Harabasz ↑': '250.0', 
        'NMI (vs Cent) ↑': '0.950', 
        'ARI (vs Cent) ↑': '0.940', 
        'Comm. Cost (MB)': '0.1245'
    }])
    abl_df = pd.DataFrame([{
        'Condition': 'Full SegFL', 
        'Silhouette ↑': '0.800', 
        'DBI ↓': '0.150', 
        'NMI (vs Cent) ↑': '0.950', 
        'ARI (vs Cent) ↑': '0.940', 
        'Privacy (ε)': '1.50', 
        'Comm. Cost (MB)': '0.1245'
    }])
    dp_df = pd.DataFrame([
        {'DP Sigma (σ)': 0.0, 'Privacy Budget (ε)': float('inf'), 'Silhouette': 0.85, 'DBI': 0.12},
        {'DP Sigma (σ)': 0.1, 'Privacy Budget (ε)': 1.5, 'Silhouette': 0.80, 'DBI': 0.15}
    ])
    
    report_data = {
        'results': res_scaffold,
        'base_df': base_df,
        'abl_df': abl_df,
        'dp_df': dp_df,
        'stability_results': {
            'silhouette_mean': 0.79, 'silhouette_std': 0.02,
            'dbi_mean': 0.16, 'dbi_std': 0.01,
            'nmi_mean': 0.94, 'nmi_std': 0.02,
            'ch_mean': 245.0, 'ch_std': 5.0
        },
        'stat_test_results': {'statistic': 4.0, 'p_value': 0.03, 'significant': True},
        'scal_results': scal,
        'gen_results': gen
    }
    
    pdf_path = os.path.join(os.path.dirname(__file__), "test_report.pdf")
    build_pdf_report(report_data, pdf_path)
    print(f"PDF generated successfully at {pdf_path} (exists: {os.path.exists(pdf_path)})")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        
    # Assertions to ensure validity
    assert res_scaffold['silhouette'] >= -1.0 and res_scaffold['silhouette'] <= 1.0, "Silhouette score out of range"
    assert len(res_scaffold['clustered_data']) == len(res_scaffold['labels']), "clustered_data rows must align 1:1 with labels"
    assert res_scaffold['epsilon'] >= 0, "Privacy budget cannot be negative"
    assert res_moon['silhouette'] >= -1.0 and res_moon['silhouette'] <= 1.0, "Silhouette score out of range"
    assert len(res_moon['clustered_data']) == len(res_moon['labels']), "clustered_data rows must align 1:1 with labels"

    print("\nAll verification checks passed successfully!")

if __name__ == '__main__':
    main()
