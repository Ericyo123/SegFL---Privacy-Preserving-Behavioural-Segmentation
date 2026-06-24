import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

class TenantAdapterLayer(nn.Module):
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

class GlobalBottleneckAutoencoder(nn.Module):
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

    def fit_predict_federated(self, tenant_latent_list, sigma=0.0):
        if not tenant_latent_list or len(tenant_latent_list[0]) == 0:
            return [np.zeros(len(lt)) for lt in tenant_latent_list]
        
        # Bounding input sensitivity by clipping latents to unit norm under DP
        if sigma > 0.0:
            clipped_list = []
            for latents in tenant_latent_list:
                if len(latents) == 0:
                    clipped_list.append(latents)
                    continue
                norms = np.linalg.norm(latents, axis=1, keepdims=True)
                clipped = latents / np.maximum(1.0, norms)
                clipped_list.append(clipped)
            tenant_latent_list = clipped_list
        
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
            
            if sigma > 0.0:
                # Bounded sensitivity with clip threshold 1.0 (count sensitivity 1, sum sensitivity 1.0)
                scale_count = sigma
                scale_sum = sigma * 1.0
                global_count += self.rs.laplace(0.0, scale_count, size=global_count.shape)
                global_count = np.clip(global_count, 1e-5, None)
                global_sum += self.rs.laplace(0.0, scale_sum, size=global_sum.shape)
                
            new_centroids = self.global_centroids.copy()
            for i in range(self.k):
                if global_count[i] > 0: 
                    new_centroids[i] = global_sum[i] / global_count[i]
                else:
                    # Dead Cluster Revival: reinitialize to a random data point
                    fallback_idx = self.rs.choice(len(tenant_latent_list[0]))
                    new_centroids[i] = tenant_latent_list[0][fallback_idx].copy()
                    if sigma > 0.0:
                        new_centroids[i] += self.rs.laplace(0.0, sigma, size=new_centroids[i].shape)
            
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
    """Legacy epsilon approximation. Use RenyiDifferentialPrivacyAccountant for formal guarantees."""
    if sigma == 0: return float('inf')
    delta = 1e-5
    q = batch_size / dataset_size
    steps = epochs * max(1, dataset_size // batch_size)
    epsilon_bound = (q * np.sqrt(2 * steps * np.log(1/delta))) / sigma
    return epsilon_bound

class RenyiDifferentialPrivacyAccountant:
    """
    Rényi Differential Privacy Accountant for DP-SGD.
    
    Provides formal (ε,δ)-DP guarantees by:
      1. Computing RDP of the Gaussian mechanism at multiple orders α
      2. Composing linearly across T training steps
      3. Converting to (ε,δ)-DP via optimal order selection
    
    References:
        Mironov, I. (2017). "Rényi Differential Privacy." CSF.
        Abadi, M. et al. (2016). "Deep Learning with Differential Privacy." CCS.
        Balle, B. et al. (2020). "Hypothesis Testing Interpretations and Renyi DP."
    """
    # Candidate RDP orders for optimization (fine-grained near 1, coarser for large α)
    ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(self, noise_multiplier, sample_rate, num_steps):
        """
        Args:
            noise_multiplier: σ — ratio of noise std to clipping norm (C).
            sample_rate: q = batch_size / dataset_size (Poisson subsampling rate).
            num_steps: T — total number of gradient descent steps.
        """
        self.sigma = noise_multiplier
        self.q = sample_rate
        self.T = num_steps

    def _rdp_gaussian(self, alpha):
        """RDP of the Gaussian mechanism at order α: ρ(α) = α / (2σ²)."""
        if self.sigma <= 0:
            return float('inf')
        return alpha / (2.0 * self.sigma ** 2)

    def _rdp_subsampled_gaussian(self, alpha):
        """
        RDP of the subsampled Gaussian mechanism.
        
        Uses the privacy amplification by subsampling bound
        (Theorem 9, Mironov 2017; tightened in Balle et al. 2020):
            ρ_sub(α) ≤ (1/(α-1)) · log(1 + q²·C(α,2)·(exp(ρ(2)) - 1))
        
        Returns the tighter of: subsampled bound vs raw Gaussian bound.
        """
        if self.sigma <= 0:
            return float('inf')
        if self.q == 0:
            return 0.0
        if self.q >= 1.0 or alpha <= 1.0:
            return self._rdp_gaussian(alpha)

        rdp_at_2 = self._rdp_gaussian(2.0)
        
        # Subsampling amplification bound
        binom_coeff = alpha * (alpha - 1) / 2.0
        amplified_term = self.q ** 2 * binom_coeff * (np.exp(rdp_at_2) - 1)

        if amplified_term > 0:
            rdp_sub = np.log1p(amplified_term) / (alpha - 1)
            return min(self._rdp_gaussian(alpha), rdp_sub)

        return self._rdp_gaussian(alpha)

    def get_epsilon(self, delta=1e-5):
        """
        Convert accumulated RDP to (ε,δ)-DP via optimal order selection.
        
        Formula: ε = min_α { T·ρ(α) + log(1/δ) / (α - 1) }
        
        Args:
            delta: Target δ for (ε,δ)-DP. Typically 1/n or 1e-5.
        Returns:
            Formal ε guarantee (float).
        """
        if self.sigma <= 0:
            return float('inf')

        best_eps = float('inf')
        for alpha in self.ORDERS:
            if alpha <= 1.0:
                continue
            rdp_total = self._rdp_subsampled_gaussian(alpha) * self.T
            eps = rdp_total + np.log(1.0 / delta) / (alpha - 1)
            if eps < best_eps:
                best_eps = eps

        return max(0.0, best_eps)


class FederatedGaussianMixtureModel:
    """
    Federated Gaussian Mixture Model clustering using expectation maximization (EM)
    over decentralized clients, assuming diagonal covariance.
    """
    def __init__(self, n_components, max_iters=20, random_state=42):
        self.k = n_components
        self.max_iters = max_iters
        self.rs = np.random.RandomState(random_state)
        self.means = None
        self.covariances = None
        self.weights = None

    def fit_predict_federated(self, tenant_latent_list, sigma=0.0):
        if not tenant_latent_list or len(tenant_latent_list[0]) == 0:
            return [np.zeros(len(lt)) for lt in tenant_latent_list]
        
        # Bounding input sensitivity by clipping latents to unit norm under DP
        if sigma > 0.0:
            clipped_list = []
            for latents in tenant_latent_list:
                if len(latents) == 0:
                    clipped_list.append(latents)
                    continue
                norms = np.linalg.norm(latents, axis=1, keepdims=True)
                clipped = latents / np.maximum(1.0, norms)
                clipped_list.append(clipped)
            tenant_latent_list = clipped_list

        dim = tenant_latent_list[0].shape[1]
        n_samples = len(tenant_latent_list[0])
        
        if n_samples < self.k:
            idx = self.rs.choice(n_samples, self.k, replace=True)
        else:
            idx = self.rs.choice(n_samples, self.k, replace=False)
        self.means = tenant_latent_list[0][idx].copy()
        self.covariances = np.ones((self.k, dim))
        self.weights = np.ones(self.k) / self.k
        
        for rnd in range(self.max_iters):
            local_Ns = []
            local_Ss = []
            local_Vs = []
            
            for latents in tenant_latent_list:
                if len(latents) == 0:
                    local_Ns.append(np.zeros(self.k))
                    local_Ss.append(np.zeros((self.k, dim)))
                    local_Vs.append(np.zeros((self.k, dim)))
                    continue
                
                log_pdfs = []
                for j in range(self.k):
                    mean = self.means[j]
                    cov = np.clip(self.covariances[j], 1e-6, None)
                    diff = latents - mean
                    squared_diff = diff ** 2 / cov
                    log_pdf = -0.5 * (dim * np.log(2 * np.pi) + np.sum(np.log(cov)) + np.sum(squared_diff, axis=1))
                    log_pdfs.append(log_pdf)
                
                log_pdfs = np.column_stack(log_pdfs)
                log_weighted_pdfs = log_pdfs + np.log(np.clip(self.weights, 1e-10, None))
                
                max_log = np.max(log_weighted_pdfs, axis=1, keepdims=True)
                sum_weighted_pdfs = np.sum(np.exp(log_weighted_pdfs - max_log), axis=1, keepdims=True)
                responsibilities = np.exp(log_weighted_pdfs - max_log) / (sum_weighted_pdfs + 1e-10)
                
                N_local = np.sum(responsibilities, axis=0)
                S_local = np.zeros((self.k, dim))
                V_local = np.zeros((self.k, dim))
                
                for j in range(self.k):
                    resp_col = responsibilities[:, j]
                    S_local[j] = np.sum(latents * resp_col[:, np.newaxis], axis=0)
                    diff = latents - self.means[j]
                    V_local[j] = np.sum((diff ** 2) * resp_col[:, np.newaxis], axis=0)
                
                local_Ns.append(N_local)
                local_Ss.append(S_local)
                local_Vs.append(V_local)
            
            global_N = np.sum(local_Ns, axis=0)
            global_S = np.sum(local_Ss, axis=0)
            global_V = np.sum(local_Vs, axis=0)
            
            if sigma > 0.0:
                global_N += self.rs.laplace(0.0, sigma, size=global_N.shape)
                global_N = np.clip(global_N, 1e-5, None)
                global_S += self.rs.laplace(0.0, sigma * 1.0, size=global_S.shape)
                global_V += self.rs.laplace(0.0, sigma * 1.0, size=global_V.shape)
                
            reg_N = np.clip(global_N, 1e-6, None)
            new_means = global_S / reg_N[:, np.newaxis]
            new_covs = global_V / reg_N[:, np.newaxis]
            new_covs = np.clip(new_covs, 1e-4, None)
            
            total_samples = np.sum(global_N)
            new_weights = global_N / (total_samples + 1e-10)
            
            if np.allclose(self.means, new_means, atol=1e-4) and np.allclose(self.covariances, new_covs, atol=1e-4):
                break
                
            self.means = new_means
            self.covariances = new_covs
            self.weights = new_weights
            
        final_labels = []
        for latents in tenant_latent_list:
            if len(latents) == 0:
                final_labels.append(np.array([]))
                continue
            
            log_pdfs = []
            for j in range(self.k):
                mean = self.means[j]
                cov = np.clip(self.covariances[j], 1e-6, None)
                diff = latents - mean
                squared_diff = diff ** 2 / cov
                log_pdf = -0.5 * (dim * np.log(2 * np.pi) + np.sum(np.log(cov)) + np.sum(squared_diff, axis=1))
                log_pdfs.append(log_pdf)
            
            log_pdfs = np.column_stack(log_pdfs)
            log_weighted_pdfs = log_pdfs + np.log(np.clip(self.weights, 1e-10, None))
            final_labels.append(np.argmax(log_weighted_pdfs, axis=1))
            
        return final_labels


class FederatedDensityBasedClustering:
    """
    Federated Density-Based clustering (HDBSCAN) using local centroid summarization.
    """
    def __init__(self, min_cluster_size=5, random_state=42):
        self.min_cluster_size = min_cluster_size
        self.rs = np.random.RandomState(random_state)
        self.exemplars = None
        self.exemplar_labels = None

    def fit_predict_federated(self, tenant_latent_list, sigma=0.0):
        from sklearn.cluster import HDBSCAN
        
        if not tenant_latent_list or len(tenant_latent_list[0]) == 0:
            return [np.zeros(len(lt)) for lt in tenant_latent_list]
        
        # Bounding input sensitivity by clipping latents to unit norm under DP
        if sigma > 0.0:
            clipped_list = []
            for latents in tenant_latent_list:
                if len(latents) == 0:
                    clipped_list.append(latents)
                    continue
                norms = np.linalg.norm(latents, axis=1, keepdims=True)
                clipped = latents / np.maximum(1.0, norms)
                clipped_list.append(clipped)
            tenant_latent_list = clipped_list

        all_exemplars = []
        for latents in tenant_latent_list:
            if len(latents) == 0:
                continue
            n_samples = len(latents)
            k_local = min(20, n_samples)
            if k_local > 1:
                km = KMeans(n_clusters=k_local, random_state=42, n_init='auto')
                km.fit(latents)
                centers = km.cluster_centers_
                if sigma > 0.0:
                    centers += self.rs.laplace(0.0, sigma, size=centers.shape)
                all_exemplars.append(centers)
            elif k_local == 1:
                mean = latents.mean(axis=0, keepdims=True)
                if sigma > 0.0:
                    mean += self.rs.laplace(0.0, sigma, size=mean.shape)
                all_exemplars.append(mean)
                
        if not all_exemplars:
            return [np.zeros(len(lt)) for lt in tenant_latent_list]
            
        self.exemplars = np.vstack(all_exemplars)
        
        mcs = min(self.min_cluster_size, len(self.exemplars) - 1)
        mcs = max(2, mcs)
        
        hdb = HDBSCAN(min_cluster_size=mcs)
        self.exemplar_labels = hdb.fit_predict(self.exemplars)
        
        final_labels = []
        for latents in tenant_latent_list:
            if len(latents) == 0:
                final_labels.append(np.array([]))
                continue
            
            dists = np.linalg.norm(latents[:, np.newaxis] - self.exemplars, axis=2)
            nearest_exemplar_idx = np.argmin(dists, axis=1)
            labels = self.exemplar_labels[nearest_exemplar_idx]
            
            max_label = np.max(self.exemplar_labels)
            noise_label = max(0, max_label + 1)
            labels = np.where(labels == -1, noise_label, labels)
            
            final_labels.append(labels)
            
        return final_labels

