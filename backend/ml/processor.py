import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

def process_csv(file_path, nrows=500_000, data_dir=None):
    """
    Enhanced CSV processor.
    If file_path is 'clicks_train.csv' and 'events.csv' exists in the same dir,
    it performs the authentic research join.
    """
    if os.path.isdir(file_path):
        data_dir = file_path
        click_p = os.path.join(data_dir, 'clicks_train.csv')
        event_p = os.path.join(data_dir, 'events.csv')
    else:
        data_dir = os.path.dirname(file_path)
        click_p = file_path
        event_p = os.path.join(data_dir, 'events.csv')

    if 'clicks_train.csv' in click_p and os.path.exists(event_p):
        print("🔗 Found authentic Outbrain dataset pair. Synchronizing relational tables...")
        cdf = pd.read_csv(click_p, usecols=['display_id', 'ad_id', 'clicked'], nrows=nrows)
        display_ids = set(cdf['display_id'].unique())
        
        # Read events.csv in chunks to save memory
        chunk_list = []
        print(f"⏳ Scanning events.csv for {len(display_ids)} unique display_ids...")
        for chunk in pd.read_csv(event_p, usecols=['display_id', 'uuid', 'timestamp', 'platform'], chunksize=100000):
            filtered_chunk = chunk[chunk['display_id'].isin(display_ids)]
            chunk_list.append(filtered_chunk)
            # If we've found all display_ids, we can stop early (optional but tricky)
            
        edf = pd.concat(chunk_list)
        df = cdf.merge(edf, on='display_id', how='inner').dropna()
        return _extract_behavioural_heuristics(df)
    
    df = pd.read_csv(click_p, nrows=nrows)
    if all(col in df.columns for col in ['uuid', 'clicked', 'ad_id', 'timestamp', 'platform']):
        return _extract_behavioural_heuristics(df)
    else:
        return _extract_generic_features(df)

def _extract_behavioural_heuristics(df):
    print("⚙️ Computing Semantic Behavioural Heuristics...")
    # Clean data
    df = df.dropna()
    
    # Heuristics
    ctr = df.groupby('uuid')['clicked'].mean().reset_index(name='ctr')
    vol = df.groupby('uuid').size().reset_index(name='vol')
    ent = df.groupby('uuid')['ad_id'].nunique().reset_index(name='ent')
    
    df['hr'] = (df['timestamp'] // 3600) % 24
    t_m = df.groupby('uuid')['hr'].mean().reset_index(name='hr_mean')
    t_v = df.groupby('uuid')['hr'].std().fillna(0).reset_index(name='hr_var')
    plat = df.groupby('uuid')['platform'].first().reset_index(name='plat')
    
    raw_feats = ctr.merge(vol, on='uuid').merge(ent, on='uuid').merge(t_m, on='uuid').merge(t_v, on='uuid').merge(plat, on='uuid')
    print(f"✅ Canonical Behaviour Matrix Formulated. Distinct UUIDs: {len(raw_feats)}")
    return raw_feats

def _extract_generic_features(df):
    print("⚙️ Applying Generic Feature Extraction...")
    # Drop non-numeric columns and identifier-like columns
    # We'll assume the first column might be an ID if it's object or int with high cardinality
    df = df.dropna()
    
    # Crude way to find a 'grouping' column like platform. 
    # We'll take the object/categorical column with lowest cardinality > 1
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        # Sort by cardinality
        plat_col = sorted(cat_cols, key=lambda x: df[x].nunique())[0]
        df = df.rename(columns={plat_col: 'plat'})
    else:
        # Fallback if no categorical column exists, create a dummy one
        df['plat'] = 0
        
    numeric_df = df.select_dtypes(include=[np.number])
    # Exclude columns that look like IDs (high cardinality, integers)
    potential_ids = [col for col in numeric_df.columns if numeric_df[col].nunique() == len(df)]
    features = numeric_df.drop(columns=potential_ids)
    
    # Ensure 'plat' is preserved
    if 'plat' not in features.columns:
        features['plat'] = df['plat']
    
    return features

def prepare_tenant_datasets(df, batch_size=256, run_seed=None):
    if run_seed is None:
        run_seed = np.random.randint(10000)
        
    # Randomly mix data so every run is unique
    df = df.sample(frac=1.0, random_state=run_seed).reset_index(drop=True)
    
    df['plat'] = df['plat'].astype(str)
    platforms = sorted(df['plat'].unique())
    train_dl, eval_dl, raw = [], [], []

    all_features = [c for c in df.columns if c not in ['plat', 'uuid']]
    
    for p_id, plat in enumerate(platforms):
        local_df = df[df['plat'] == plat].copy()
        if len(local_df) < 10: 
            continue
            
        # ── INJECT FEATURE HETEROGENEITY ──
        active_features = all_features.copy()
        if p_id == 1 and len(active_features) >= 3:
            if 'ent' in active_features:
                active_features = [c for c in active_features if c not in ['ent', 'hr_var']]
            else:
                active_features = active_features[:-1] # Generic CSV fallback
        elif p_id == 2 and len(active_features) >= 3:
            if 'hr_mean' in active_features:
                active_features = [c for c in active_features if c not in ['hr_mean', 'hr_var']]
            else:
                active_features = active_features[1:] # Generic CSV fallback
            
        feats = StandardScaler().fit_transform(local_df[active_features].values)

        X_tmp, X_ts = train_test_split(feats, test_size=0.10, random_state=run_seed)
        X_tr, X_val = train_test_split(X_tmp, test_size=0.1111, random_state=run_seed)

        train_dl.append(DataLoader(TensorDataset(torch.FloatTensor(X_tr)), batch_size=batch_size, shuffle=True))
        eval_dl.append({
            'val': DataLoader(TensorDataset(torch.FloatTensor(X_val)), batch_size=batch_size),
            'test': DataLoader(TensorDataset(torch.FloatTensor(X_ts)), batch_size=batch_size)
        })
        raw.append({
            'dim': len(active_features), 
            'mask': active_features, 
            'raw_target': local_df
        })

    return train_dl, eval_dl, raw
