import torch
from torch.utils.data import TensorDataset, DataLoader
import mhnlib.utils as mhn_utils

@torch.no_grad()
def mhn_full_pcha(x, n_c, beta, eta, num_iterations):
    N, d = x.shape

    z = torch.randn(n_c, d, dtype=x.dtype, device=x.device) * x.var(dim=0).sqrt()[None, :] + x.mean(dim=0)[None, :]
    a = torch.zeros(n_c, dtype=x.dtype, device=x.device)
    b = torch.zeros(N, dtype=x.dtype, device=x.device)

    for _ in range(num_iterations):
        # Z -> X : construct archetypes as convex combinations of data
        sp = torch.softmax(beta * (z @ x.T + b[None, :]), dim=1)          # (K, N)
        error_z = sp @ x - z                                               # (K, d)

        g_sp = error_z @ x.T                                               # (K, N)
        g_logits_sp = sp * (g_sp - (sp * g_sp).sum(dim=1, keepdim=True))
        b -= eta * beta * g_logits_sp.mean(dim=0)
        b -= b.mean()

        # Relax Z toward its convex representation
        sp = torch.softmax(beta * (z @ x.T + b[None, :]), dim=1)
        z += eta * (sp @ x - z)

        # X -> Z : reconstruct data from archetypes
        s = torch.softmax(beta * (x @ z.T + a[None, :]), dim=1)           # (N, K)
        error_x = s @ z - x                                                # (N, d)

        g_s = error_x @ z.T                                                # (N, K)
        g_logits_s = s * (g_s - (s * g_s).sum(dim=1, keepdim=True))
        a -= eta * beta * g_logits_s.mean(dim=0)
        a -= a.mean()

    sp = torch.softmax(beta * (z @ x.T + b[None, :]), dim=1)
    z = sp @ x
    s = torch.softmax(beta * (x @ z.T + a[None, :]), dim=1)

    return z, a, b, s, sp

@torch.no_grad()
def mhn_relaxed_pcha(x, n_c, beta, eta, num_iterations, rcond=1e-6):
    N, d = x.shape

    z = torch.randn(n_c, d, dtype=x.dtype, device=x.device) * x.var(dim=0).sqrt()[None, :] + x.mean(dim=0)[None, :]
    a = torch.zeros(n_c, dtype=x.dtype, device=x.device)

    for _ in range(num_iterations):
        p = torch.softmax(beta * (x @ z.T + a[None, :]), dim=1)

        # Relaxed Z update
        z_opt = torch.linalg.pinv(p, rtol=rcond) @ x
        z += eta * (z_opt - z)

        # Recompute assignments
        p = torch.softmax(beta * (x @ z.T + a[None, :]), dim=1)

        # Gradient update for a
        error = p @ z - x
        g_p = error @ z.T
        g_logits = p * (g_p - (p * g_p).sum(dim=1, keepdim=True))
        a -= eta * beta * g_logits.mean(dim=0)
        a -= a.mean()

    p = torch.softmax(beta * (x @ z.T + a[None, :]), dim=1)
    return z, a, p

@torch.no_grad()
def euclidean_mhn_relaxed_pcha(x, n_c, beta, eta, num_iterations, rcond=1e-6):
    N, d = x.shape

    z = torch.randn(n_c, d, dtype=x.dtype, device=x.device) * x.var(dim=0).sqrt()[None, :] + x.mean(dim=0)[None, :]

    for _ in range(num_iterations):
        distances = torch.cdist(x, z).square()                 # (N, K)
        p = torch.softmax(-0.5 * beta * distances, dim=1)

        z_opt = torch.linalg.pinv(p, rtol=rcond) @ x          # (K, d)
        z += eta * (z_opt - z)

    distances = torch.cdist(x, z).square()
    p = torch.softmax(-0.5 * beta * distances, dim=1)

    return z, p
        
    

@torch.no_grad()
def mhn_fit(data, num_clusters, log_beta_0, fit_bias, batch_size, learning_rate, num_epochs, device = None, cov_estimator_num_batches = 5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset =   TensorDataset(torch.as_tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    N = data.shape[-1]
    cov_estimator_num_batches = min(cov_estimator_num_batches, 1)
    cov_estimator_indices = torch.randint(low=0, high=len(dataloader), size=(cov_estimator_num_batches * batch_size,), device=device)
    cov = torch.cov(torch.as_tensor(data[cov_estimator_indices.cpu().numpy()].T)).to(device)

    cov_vals, cov_vecs = torch.linalg.eigh(cov)
    log_biases = torch.randn(num_clusters, device=device) if fit_bias else torch.zeros(num_clusters, device=device)

    clusters = torch.randn(num_clusters, N, device=device)*cov_vals.clamp(min=1e-12).sqrt() @ cov_vecs.T
    beta = torch.exp(torch.as_tensor(log_beta_0, device=device))
    beta_hist = [beta.cpu().item()]
    for epoch in range(num_epochs):
        for (batch,) in dataloader:
            batch = batch.to(device)
            gram_batch = batch @ batch.T
            stab_gram_batch  = mhn_utils.get_symmetric_stability_matrix(gram_batch,torch.ones(len(batch), device=device)/len(batch), return_proj=False )
            stab_gram_batch_s_vals = torch.linalg.svdvals(stab_gram_batch)
            beta_c_batch= 1.0/(stab_gram_batch_s_vals**2).max()
            beta = torch.exp(torch.log(beta) + learning_rate*(torch.log(beta_c_batch)-torch.log(beta)))
            batch_norm_sq = batch.pow(2).sum(dim=-1)
            clusters_norm_sq = clusters.pow(2).sum(dim=-1)
            logits = torch.einsum("ki,si->sk", clusters, batch)
            logits.add(-0.5 * batch_norm_sq.view( -1, 1))
            logits.add(-0.5 * clusters_norm_sq.view(1, -1))
            logits.mul_(beta.view(1, 1))
            logits.add_(log_biases)
            weights = torch.softmax(logits, dim=-1)
            average_weights = weights.mean(dim=0)
            posterior_mean_data = torch.einsum("sk,si->ki", weights, batch)/len(batch)
            posterior_mean_clusters = torch.einsum("k,ki->ki", average_weights, clusters)
            clusters += learning_rate * (posterior_mean_data - posterior_mean_clusters)
            if fit_bias:
                log_biases += learning_rate * (average_weights - torch.softmax(log_biases, dim=0))
        gram_clusters = clusters @ clusters.T
        stab_clusters  = mhn_utils.get_symmetric_stability_matrix(gram_clusters,torch.softmax(log_biases, dim=0), return_proj=False )
        stab_gram_clusters_s_vals = torch.linalg.svdvals(stab_clusters)
        beta_c_clusters= 1.0/(stab_gram_clusters_s_vals**2).max()
        beta = torch.exp(torch.log(beta) + learning_rate*(torch.log(beta_c_clusters)-torch.log(beta)))
        beta_hist.append(beta.cpu().item())
    return clusters.cpu(), log_biases.cpu(), torch.as_tensor(beta_hist).cpu()

@torch.no_grad()
def mhn_fit_diffusion_like(data, num_clusters, fit_bias, batch_size, learning_rate, num_epochs, device = None, cov_estimator_num_batches = 5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset =   TensorDataset(torch.as_tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    N = data.shape[-1]
    cov_estimator_num_batches = min(cov_estimator_num_batches, 1)
    cov_estimator_indices = torch.randint(low=0, high=len(dataloader), size=(cov_estimator_num_batches * batch_size,), device=device)
    cov = torch.cov(torch.as_tensor(data[cov_estimator_indices.cpu().numpy()].T)).to(device)

    cov_vals, cov_vecs = torch.linalg.eigh(cov)
    log_biases = torch.randn(num_clusters, device=device) if fit_bias else torch.zeros(num_clusters, device=device)

    clusters = torch.randn(num_clusters, N, device=device)*cov_vals.clamp(min=1e-12).sqrt() @ cov_vecs.T
    for epoch in range(num_epochs):
        for (batch,) in dataloader:
            batch = batch.to(device)
            batch_norm_sq = batch.pow(2).sum(dim=-1)
            clusters_norm_sq = clusters.pow(2).sum(dim=-1)
            logits = torch.einsum("ki,si->sk", clusters, batch)
            logits.add(-0.5 * batch_norm_sq.view( -1, 1))
            logits.add(-0.5 * clusters_norm_sq.view(1, -1))
            logits.mul_(beta.view(1, 1))
            logits.add_(log_biases)
            weights = torch.softmax(logits, dim=-1)
            average_weights = weights.mean(dim=0)
            posterior_mean_data = torch.einsum("sk,si->ki", weights, batch)/len(batch)
            posterior_mean_clusters = torch.einsum("k,ki->ki", average_weights, clusters)
            clusters += learning_rate * (posterior_mean_data - posterior_mean_clusters)
            if fit_bias:
                log_biases += learning_rate * (average_weights - torch.softmax(log_biases, dim=0))
    return clusters.cpu(), log_biases.cpu()