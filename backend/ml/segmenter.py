import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

class TAL_Adapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Deeper hierarchical feature extraction for complex alignment
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 32), 
            nn.LayerNorm(32), 
            nn.LeakyReLU(0.1), 
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(16, out_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(out_dim, 16), 
            nn.LayerNorm(16), 
            nn.LeakyReLU(0.1), 
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(32, in_dim)
        )
    def forward(self, x): 
        return self.enc(x), self.dec(self.enc(x))

class GlobalBottleneckAE(nn.Module):
    def __init__(self, in_dim, bottle_dim):
        super().__init__()
        # Deeper hierarchical representations for reconstruction
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.LayerNorm(64), 
            nn.LeakyReLU(0.1), 
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(32, bottle_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(bottle_dim, 32), 
            nn.LayerNorm(32), 
            nn.LeakyReLU(0.1), 
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(64, in_dim)
        )
    def forward(self, x): 
        return self.dec(self.enc(x)), self.enc(x)

class FederatedKMeans:
    def __init__(self, n_clusters, max_iters=20, random_state=42):
        self.k = n_clusters
        self.max_iters = max_iters
        self.rs = np.random.RandomState(random_state)
        self.global_centroids = None

    def fit_predict_federated(self, tenant_latent_list):
        if not tenant_latent_list or len(tenant_latent_list[0]) == 0:
            return [np.zeros(len(lt)) for lt in tenant_latent_list]
        
        n_samples = len(tenant_latent_list[0])
        if n_samples < self.k:
            idx = self.rs.choice(n_samples, self.k, replace=True)
        else:
            idx = self.rs.choice(n_samples, self.k, replace=False)
        self.global_centroids = tenant_latent_list[0][idx].copy()

        for rnd in range(self.max_iters):
            tenant_sums, tenant_counts = [], []
            for latents in tenant_latent_list:
                if len(latents) == 0:
                    tenant_sums.append(np.zeros_like(self.global_centroids))
                    tenant_counts.append(np.zeros(self.k))
                    continue
                distances = np.linalg.norm(latents[:, np.newaxis] - self.global_centroids, axis=2)
                labels = np.argmin(distances, axis=1)
                sums = np.zeros_like(self.global_centroids)
                counts = np.zeros(self.k)
                for i in range(self.k):
                    cluster_pts = latents[labels == i]
                    if len(cluster_pts) > 0:
                        sums[i] = cluster_pts.sum(axis=0)
                        counts[i] = len(cluster_pts)
                tenant_sums.append(sums)
                tenant_counts.append(counts)

            global_sum = np.sum(tenant_sums, axis=0)
            global_count = np.sum(tenant_counts, axis=0)
            new_centroids = self.global_centroids.copy()
            for i in range(self.k):
                if global_count[i] > 0: 
                    new_centroids[i] = global_sum[i] / global_count[i]
                else:
                    # Dead Cluster Revival: reinitialize to a random data point
                    fallback_idx = self.rs.choice(len(tenant_latent_list[0]))
                    new_centroids[i] = tenant_latent_list[0][fallback_idx].copy()
            
            if np.allclose(self.global_centroids, new_centroids, atol=1e-4): 
                break
            self.global_centroids = new_centroids

        final_labels = []
        for latents in tenant_latent_list:
            if len(latents) > 0:
                dist = np.linalg.norm(latents[:, np.newaxis] - self.global_centroids, axis=2)
                final_labels.append(np.argmin(dist, axis=1))
            else: 
                final_labels.append(np.array([]))
        return final_labels

def formal_aggregator(states, counts):
    global_s = states[0].copy()
    total_n = sum(counts)
    for k in global_s.keys():
        global_s[k] = sum(state[k] * (n / total_n) for state, n in zip(states, counts))
    return global_s

def compute_epsilon(sigma, epochs, dataset_size, batch_size=256):
    if sigma == 0: return float('inf')
    delta = 1e-5
    q = batch_size / dataset_size # Sampling Ratio
    steps = epochs * max(1, dataset_size // batch_size)
    epsilon_bound = (q * np.sqrt(2 * steps * np.log(1/delta))) / sigma
    return epsilon_bound
