import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def compute_cluster_explainability(df, labels, feature_cols=None):
    """
    Computes explainability metrics for clustering segments:
      1. Surrogate Global Feature Importances: Train a Random Forest Classifier
         to predict cluster assignments using raw features.
      2. Cluster Enrichment Scores: Calculates how many standard deviations the
         mean of each feature in a cluster deviates from the global dataset average.
    """
    if len(df) == 0:
        return pd.DataFrame(), {}
        
    if feature_cols is None:
        exclude_cols = ['uuid', 'cluster', 'plat', 'platform', 'Cluster Size', 'Persona', 'Platform ID']
        feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
    if not feature_cols:
        return pd.DataFrame(), {}

    # Map column names to clean terms if present
    rename_map = {
        'ctr': 'Click-Through Rate',
        'vol': 'Interaction Volume',
        'ent': 'Ad Entropy (Variety)',
        'hr_mean': 'Active Hour (Mean)',
        'hr_var': 'Active Hour (Variance)'
    }
    
    # 1. Global Surrogate Random Forest
    X = df[feature_cols].fillna(0).values
    y = np.array(labels)
    
    # Check if we have valid classes to classify
    unique_classes = np.unique(y)
    if len(unique_classes) > 1 and len(X) >= len(unique_classes):
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        importances = clf.feature_importances_
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)
        
    display_names = [rename_map.get(col, col) for col in feature_cols]
    
    importance_df = pd.DataFrame({
        'Feature': display_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # 2. Local Cluster Enrichment Scores: (Mean_cluster - Mean_global) / Std_global
    global_means = df[feature_cols].mean()
    global_stds = df[feature_cols].std().replace(0, 1e-5)
    
    enrichment_scores = {}
    for cluster_id in unique_classes:
        cluster_df = df[y == cluster_id]
        if len(cluster_df) > 0:
            cluster_means = cluster_df[feature_cols].mean()
            # Calculate standard score (z-score difference)
            scores = (cluster_means - global_means) / global_stds
            scores_renamed = {rename_map.get(col, col): float(scores[col]) for col in feature_cols}
            enrichment_scores[int(cluster_id)] = scores_renamed
            
    return importance_df, enrichment_scores
