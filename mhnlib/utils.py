import numpy as np
import torch
import torch.linalg
from einops import einsum
from tqdm.auto import tqdm
from math import log, sqrt
from typing import Optional, Tuple, Union
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix, csr_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA

def shift_and_rms(x : torch.Tensor, eps=1e-12):
    shift = x.mean(dim=0, keepdim=False)
    x_shifted = x - shift[None, :]

    rms = x_shifted.norm(dim=1).square().mean().sqrt()
    rms = rms.clamp_min(eps)

    x_norm = x_shifted / rms

    return shift, rms, x_norm


def jensen_shannon_torch(p, q, eps=1e-12):
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)

    p = p.unsqueeze(-2)
    q = q.unsqueeze(-3)

    m = 0.5 * (p + q)

    kl_pm = torch.where(
        p > 0,
        p * (p.clamp_min(eps).log() - m.clamp_min(eps).log()),
        0.0,
    ).sum(dim=-1)

    kl_qm = torch.where(
        q > 0,
        q * (q.clamp_min(eps).log() - m.clamp_min(eps).log()),
        0.0,
    ).sum(dim=-1)

    jsd = 0.5 * (kl_pm + kl_qm)

    return jsd.clamp_min(0).sqrt()
def jensen_shannon_numpy(p, q, eps=1e-12):
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), eps)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), eps)

    p = np.expand_dims(p, axis=-2)
    q = np.expand_dims(q, axis=-3)

    m = 0.5 * (p + q)

    kl_pm = np.where(
        p > 0,
        p * (np.log(np.maximum(p, eps)) - np.log(np.maximum(m, eps))),
        0.0,
    ).sum(axis=-1)

    kl_qm = np.where(
        q > 0,
        q * (np.log(np.maximum(q, eps)) - np.log(np.maximum(m, eps))),
        0.0,
    ).sum(axis=-1)

    jsd = 0.5 * (kl_pm + kl_qm)

    return np.sqrt(np.maximum(jsd, 0.0))

def group_by_jensen(data, threshold: float, complete_linkage: bool = False):
    """
    Group points by Jensen-Shannon distance.

    complete_linkage=False:
        Connected components under JSD(x_i, x_j) <= threshold.

    complete_linkage=True:
        Complete linkage: every pair within a cluster satisfies
        JSD(x_i, x_j) <= threshold.
    """

    if isinstance(data, torch.Tensor):
        convert_to_torch = True
        device = data.device
        dtype = data.dtype
        x = data.detach().cpu().numpy()
    else:
        convert_to_torch = False
        x = np.asarray(data)

    n, d = x.shape

    jsd_matrix = jensen_shannon_numpy(x, x)

    if complete_linkage:
        if n == 1:
            labels = np.zeros(1, dtype=int)
            n_groups = 1
        else:
            jsd_matrix = 0.5 * (jsd_matrix + jsd_matrix.T)
            np.fill_diagonal(jsd_matrix, 0.0)

            Z = linkage(
                squareform(jsd_matrix, checks=False),
                method="complete",
            )

            labels = fcluster(
                Z,
                t=threshold,
                criterion="distance",
            ) - 1

            n_groups = labels.max() + 1

    else:
        indices = np.where(jsd_matrix <= threshold)

        graph = coo_matrix(
            (np.ones_like(indices[0]), indices),
            shape=(n, n),
        ).tocsr()

        n_groups, labels = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )

    counts = np.bincount(labels, minlength=n_groups)

    x_unique = np.empty((n_groups, d), dtype=x.dtype)

    for k in range(d):
        x_unique[:, k] = np.bincount(
            labels,
            weights=x[:, k],
            minlength=n_groups,
        )

    x_unique /= counts[:, None]

    if convert_to_torch:
        x_unique = torch.as_tensor(
            x_unique, device=device, dtype=dtype
        )
        counts = torch.as_tensor(counts, device=device)
        labels = torch.as_tensor(labels, device=device)

    return x_unique, counts, labels
def group_by_similarity(
    data,
    dot_scale: float,
    complete_linkage: bool = False,
):
    """
    Group points by cosine similarity.

    complete_linkage=False:
        Connected components under cosine(x_i, x_j) >= dot_scale.

    complete_linkage=True:
        Complete linkage: every pair within a cluster satisfies
        cosine(x_i, x_j) >= dot_scale.
    """

    if isinstance(data, torch.Tensor):
        convert_to_torch = True
        device = data.device
        dtype = data.dtype
        x = data.detach().cpu().numpy()
    else:
        convert_to_torch = False
        x = np.asarray(data)

    n, d = x.shape

    dot_product = np.dot(x, x.T)
    dot_product /= np.linalg.norm(x, axis=1)[:, None]
    dot_product /= np.linalg.norm(x, axis=1)[None, :]

    dot_product = np.clip(dot_product, -1.0, 1.0)

    if complete_linkage:
        if n == 1:
            labels = np.zeros(1, dtype=int)
            n_groups = 1
        else:
            distance_matrix = 1.0 - dot_product
            np.fill_diagonal(distance_matrix, 0.0)

            Z = linkage(
                squareform(distance_matrix, checks=False),
                method="complete",
            )

            labels = fcluster(
                Z,
                t=1.0 - dot_scale,
                criterion="distance",
            ) - 1

            n_groups = labels.max() + 1

    else:
        indices = np.where(dot_product >= dot_scale)

        graph = coo_matrix(
            (np.ones_like(indices[0]), indices),
            shape=(n, n),
        ).tocsr()

        n_groups, labels = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )

    counts = np.bincount(labels, minlength=n_groups)

    x_unique = np.empty((n_groups, d), dtype=x.dtype)

    for k in range(d):
        x_unique[:, k] = np.bincount(
            labels,
            weights=x[:, k],
            minlength=n_groups,
        )

    x_unique /= counts[:, None]

    if convert_to_torch:
        x_unique = torch.as_tensor(
            x_unique, device=device, dtype=dtype
        )
        counts = torch.as_tensor(counts, device=device)
        labels = torch.as_tensor(labels, device=device)

    return x_unique, counts, labels
def group_by_distance(
    data,
    eps: float,
    complete_linkage: bool):
    """
    Group points under ||x_i - x_j|| <= eps.

    complete_linkage=False:
        Connected components.

    complete_linkage=True:
        Complete linkage: every pair within a cluster satisfies
        ||x_i - x_j|| <= eps.
    """

    if isinstance(data, torch.Tensor):
        convert_to_torch = True
        device = data.device
        dtype = data.dtype
        x = data.detach().cpu().numpy()
    else:
        convert_to_torch = False
        x = np.asarray(data)

    n, d = x.shape

    if complete_linkage:
        if n == 1:
            labels = np.zeros(1, dtype=int)
            n_groups = 1
        else:
            Z = linkage(
                pdist(x, metric="euclidean"),
                method="complete",
            )

            labels = fcluster(
                Z,
                t=eps,
                criterion="distance",
            ) - 1

            n_groups = labels.max() + 1

    else:
        tree = KDTree(x)

        graph = tree.sparse_distance_matrix(
            tree,
            max_distance=eps,
            output_type="coo_matrix",
        )

        n_groups, labels = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )

    counts = np.bincount(labels, minlength=n_groups)

    x_unique = np.empty((n_groups, d), dtype=x.dtype)

    for k in range(d):
        x_unique[:, k] = np.bincount(
            labels,
            weights=x[:, k],
            minlength=n_groups,
        )

    x_unique /= counts[:, None]

    if convert_to_torch:
        x_unique = torch.as_tensor(
            x_unique, device=device, dtype=dtype
        )
        counts = torch.as_tensor(counts, device=device)
        labels = torch.as_tensor(labels, device=device)

    return x_unique, counts, labels


def get_gram_matrix(patterns):
    if patterns.ndim < 2:
        raise ValueError(f"patterns must have at least 2 dimensions, got {patterns.shape}")
    return torch.einsum("...hi,...ki->...hk", patterns, patterns)
def get_entropy(w):
    h = -w * w.log()
    h[~torch.isfinite(h)] = 0.0
    h = h.sum(dim=-1)
    return h
def get_fisher_matrix(w):
    if w.ndim == 1:
        return torch.diag(w) - torch.outer(w, w)
    elif w.ndim == 2:
        return torch.diag_embed(w) - w[:, :, None] * w[:, None, :]
    else:
        raise ValueError(f"w must have shape (K,) or (B, K), got {w.shape}")
def get_dual_stability_matrix(gram, w):
    fisher = get_fisher_matrix(w)
    return torch.einsum("...hk,...kl->...hl", fisher, gram)
def get_dual_symmetric_stability_matrix(gram, w, return_proj=False):
    fisher = get_fisher_matrix(w)
    fisher_vals, fisher_vecs = torch.linalg.eigh(fisher)
    fisher_sqrt_vals = torch.sqrt(fisher_vals.clamp_min(0.0))
    fisher_sqrt = torch.einsum("...hk,...k,...lk->...hl", fisher_vecs, fisher_sqrt_vals, fisher_vecs)
    stab = torch.einsum("...hk,...kl,...lm->...hm", fisher_sqrt, gram, fisher_sqrt)
    stab = 0.5 * (stab + stab.transpose(-1, -2))
    if not return_proj:
        return stab
    else:
        return stab, fisher_sqrt

def prepare_initial_conditions(
    patterns: torch.Tensor,
    biases: torch.Tensor,
    betas: Union[torch.Tensor, float],
    x0: torch.Tensor,
):
    patterns = torch.as_tensor(patterns)
    device = patterns.device
    dtype = patterns.dtype
    biases = torch.as_tensor(biases, dtype=dtype, device=device)
    x0 = torch.as_tensor(x0, dtype=dtype, device=device)
    betas = torch.as_tensor(betas, dtype=dtype, device=device)

    # ------------------------------------------------------------
    # Patterns: (K, N) or (P, K, N)
    # ------------------------------------------------------------
    if patterns.ndim not in (2, 3):
        raise ValueError(
            f"patterns must have shape (K, N) or (P, K, N), "
            f"got {patterns.shape}"
        )

    if patterns.ndim == 2:
        patterns = patterns.unsqueeze(0)

    P, K, N = patterns.shape

    # ------------------------------------------------------------
    # Biases: (K,) or (P, K)
    # ------------------------------------------------------------
    if biases.ndim not in (1, 2):
        raise ValueError(
            f"biases must have shape (K,) or (P, K), got {biases.shape}"
        )

    if biases.shape[-1] != K:
        raise ValueError(
            f"last dimension of biases must match K={K}, "
            f"got {biases.shape}"
        )

    if biases.ndim == 1:
        biases = biases.unsqueeze(0)

    if biases.shape[0] not in (1, P):
        raise ValueError(
            f"batch dimension of biases must be 1 or match "
            f"patterns batch P={P}, got {biases.shape}"
        )

    # ------------------------------------------------------------
    # Initial conditions:
    # (N,), (N_ic, N), or (P, N_ic, N)
    # ------------------------------------------------------------
    if x0.ndim not in (1, 2, 3):
        raise ValueError(
            f"x0 must have shape (N,), (N_ic, N), or "
            f"(P, N_ic, N), got {x0.shape}"
        )

    if x0.shape[-1] != N:
        raise ValueError(
            f"last dimension of x0 must match N={N}, got {x0.shape}"
        )

    if x0.ndim == 1:
        x0 = x0.reshape(1, 1, N)
    elif x0.ndim == 2:
        x0 = x0.unsqueeze(0)

    if x0.shape[0] not in (1, P):
        raise ValueError(
            f"batch dimension of x0 must be 1 or match "
            f"patterns batch P={P}, got {x0.shape}"
        )

    N_ic = x0.shape[1]

    # ------------------------------------------------------------
    # Beta sweep: (B,)
    # ------------------------------------------------------------
    if betas.ndim == 0:
        betas = betas.reshape(1)
    elif betas.ndim != 1:
        raise ValueError(
            f"betas must be scalar or one-dimensional, got {betas.shape}"
        )

    # Broadcast shared quantities over pattern batches
    biases = biases.expand(P, K)
    x0 = x0.expand(P, N_ic, N)

    return patterns, biases, betas, x0

def prepare_initial_conditions_dual(
    grams: torch.Tensor,
    biases: torch.Tensor,
    betas: Union[torch.Tensor, float],
    w0: torch.Tensor,
):
    grams = torch.as_tensor(grams)
    device = grams.device
    dtype = grams.dtype

    biases = torch.as_tensor(biases, dtype=dtype, device=device)
    w0 = torch.as_tensor(w0, dtype=dtype, device=device)
    betas = torch.as_tensor(betas, dtype=dtype, device=device)

    # Grams: (K, K) or (P, K, K)
    if grams.ndim not in (2, 3):
        raise ValueError(
            f"grams must have shape (K, K) or (P, K, K), "
            f"got {grams.shape}"
        )

    if grams.ndim == 2:
        grams = grams.unsqueeze(0)

    P, K, K_alt = grams.shape

    if K_alt != K:
        raise ValueError(
            f"grams must be square in their last two dimensions, "
            f"got {grams.shape}"
        )

    # Biases: (K,) or (P, K)
    if biases.ndim not in (1, 2):
        raise ValueError(
            f"biases must have shape (K,) or (P, K), got {biases.shape}"
        )

    if biases.shape[-1] != K:
        raise ValueError(
            f"last dimension of biases must match K={K}, "
            f"got {biases.shape}"
        )

    if biases.ndim == 1:
        biases = biases.unsqueeze(0)

    if biases.shape[0] not in (1, P):
        raise ValueError(
            f"batch dimension of biases must be 1 or match "
            f"Gram batch P={P}, got {biases.shape}"
        )

    # Initial conditions:
    # (K,), (N_ic, K), or (P, N_ic, K)
    if w0.ndim not in (1, 2, 3):
        raise ValueError(
            f"w0 must have shape (K,), (N_ic, K), or "
            f"(P, N_ic, K), got {w0.shape}"
        )

    if w0.shape[-1] != K:
        raise ValueError(
            f"last dimension of w0 must match K={K}, got {w0.shape}"
        )

    if w0.ndim == 1:
        w0 = w0.reshape(1, 1, K)
    elif w0.ndim == 2:
        w0 = w0.unsqueeze(0)

    if w0.shape[0] not in (1, P):
        raise ValueError(
            f"batch dimension of w0 must be 1 or match "
            f"Gram batch P={P}, got {w0.shape}"
        )

    N_ic = w0.shape[1]

    # Beta sweep: (B,)
    if betas.ndim == 0:
        betas = betas.reshape(1)
    elif betas.ndim != 1:
        raise ValueError(
            f"betas must be scalar or one-dimensional, got {betas.shape}"
        )

    biases = biases.expand(P, K)
    w0 = w0.expand(P, N_ic, K)

    return grams, biases, betas, w0
    
def deterministic_dynamics(
    patterns: torch.Tensor,
    biases: torch.Tensor,
    betas: Union[torch.Tensor, float],
    x0: torch.Tensor,
    num_iterations: int,
    return_probs: bool = False,
    verbose: bool = False,
    dt : Optional[float] = None,
):
    """
    Parameters:
        patterns: (K, N) or (N_p, K, N)
        biases:   (K,) or (N_p, K)
        betas:    scalar or (B,)
        x0:       (N,), (N_ic, N), or (N_p, N_ic, N)

    Broadcasting:
        Singleton pattern-batch axes are broadcast to the common batch size.

    Returns:
        x:     (B, N_p, N_ic, N)
        probs: (B, N_p, N_ic, K), if return_probs=True

    When return_probs=True and num_iterations >= 1, the returned quantities
    satisfy x = P^T probs up to numerical precision.
    """
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")

    patterns, biases, betas, x0 = prepare_initial_conditions(patterns, biases, betas, x0)
    P, K, N = patterns.shape
    N_ic = x0.shape[-2]
    B = betas.numel()
    x = x0.unsqueeze(0).expand(B, P, N_ic, N).clone()

    betas_view = betas.reshape(B, 1, 1, 1)
    biases_view = biases.reshape(1, P, 1, K)

    iterator = tqdm(
        range(num_iterations),
        desc="Computing fixed points",
        disable=not verbose,
    )

    probs = None

    for _ in iterator:
        projected = torch.einsum(
            "bpjn,pkn->bpjk",
            x,
            patterns,
        )
        logits = betas_view * (projected + biases_view)
        probs = torch.softmax(logits, dim=-1)
        softmax_mean = torch.einsum(
                "bpjk,pkn->bpjn",
                probs,
                patterns)

        if dt:
            x += dt*(softmax_mean - x)
        else:
            x = softmax_mean

    if return_probs:
        return x, probs

    return x

def dual_deterministic_dynamics(
    grams: torch.Tensor,
    biases: torch.Tensor,
    betas: Union[torch.Tensor, float],
    w0: torch.Tensor,
    num_iterations: int,
    verbose: bool = False,
    target_device = None,
    dt : Optional[float] = None,
):
    """
    Perform Picard iterations of the dual map

        w_{t+1} = softmax(beta * (G w_t + b)).

    For G = P P^T and x0 = P^T w0, this is equivalent to
    deterministic_dynamics, with x_t = P^T w_t.

    Shapes:
        grams:  (K, K) or (N_G, K, K)
        biases: (K,) or (N_G, K)
        betas:  scalar or (B,)
        w0:     (K,), (N_ic, K), or (N_G, N_ic, K)

    Returns:
        w: (B, N_G, N_ic, K)
    """
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")

    grams, biases, betas, w0 = prepare_initial_conditions_dual(grams, biases, betas, w0)
    P,K,K = grams.shape
    B = betas.numel()
    N_ic = w0.shape[-2]
    w = w0.unsqueeze(0).expand(B, P, N_ic, K).clone()

    betas_view = betas.reshape(B, 1, 1, 1)
    biases_view = biases.reshape(1, P, 1, K)

    iterator = tqdm(
        range(num_iterations),
        desc="Computing dual fixed points",
        disable=not verbose,
    )

    grams_T = grams.transpose(-1, -2).contiguous()

    for _ in iterator:
        # Explicitly compute (G w)_k = sum_h G_{k h} w_h.
        gram_field = w @ grams_T

        logits = betas_view * (gram_field + biases_view)
        new_w = torch.softmax(logits, dim=-1)
    
        if dt:
            w = w + dt * (new_w - w)
        else:
            w = new_w
        #gram_field = torch.einsum(
        #    "pkh,bpjh->bpjk",
        #    grams,
        #    w,
        #)
        #logits = betas_view * (gram_field + biases_view)
        #new_w = torch.softmax(logits, dim=-1)
        #if dt:
        #    w += dt*(new_w-w)
        #else:
        #    w = new_w
    if target_device is not None:
        w = w.to(target_device)
    return w

def deterministic_dynamics_cont_annealing(
    patterns: torch.Tensor,
    biases: torch.Tensor,
    x0: torch.Tensor,
    beta_start : float,
    beta_end : float,
    beta_logspace : bool,
    dt : float,
    num_iterations: int,
    num_snapshots: int,
    return_probs: bool = False,
    verbose: bool = False,
):

    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1.")
    
    if beta_logspace:
        betas = torch.logspace(log(beta_start), log(beta_end), steps=num_iterations)
    else:
        betas = torch.linspace(beta_start, beta_end, steps=num_iterations)
    patterns, biases, betas, x0 = prepare_initial_conditions(patterns, biases, betas, x0)
    P, K, N = patterns.shape
    N_ic = x0.shape[-2]
    x = x0.expand(P, N_ic, N).clone()

    x_coll = []
    if return_probs:
        probs_coll = []

    iterator = tqdm(
        enumerate(betas),
        total=len(betas),
        desc="Annealing",
        disable=not verbose,
    )
    
    biases_view = biases.view(P, 1, K)
    
    for beta_idx, beta in iterator:
        logits = torch.einsum("ski,sci->sck", patterns, x)
        logits = beta * (logits + biases_view)
        weights = torch.softmax(logits, dim=-1)
        means = torch.einsum("sck,ski->sci", weights, patterns)
        x += dt*(means - x)
        if beta_idx % (num_iterations // num_snapshots) == 0:
            x_coll.append(x.clone())
            if return_probs:
                probs_coll.append(weights.clone())
    if return_probs:
        return torch.stack(x_coll, dim=0), torch.stack(probs_coll, dim=0)
    else:
        return torch.stack(x_coll, dim=0)
        
def deterministic_dynamics_annealing(
    patterns: torch.Tensor,
    biases: torch.Tensor,
    betas: Union[torch.Tensor, float],
    x0: torch.Tensor,
    logit_noise_std: float,
    num_iterations: int,
    return_probs: bool = False,
    verbose: bool = False,
    dt : Optional[float] = None
):
    """
    Annealed deterministic dynamics.

    At each beta, compute the deterministic fixed point

        x = P^T softmax(beta * (P x + b))

    Then, before moving to the next beta, perturb the logits by Gaussian noise:

        logits -> logits + logit_noise_std * eta

    and use the corresponding softmax barycenter as the next initial condition.

    Shapes:
        patterns: (K, N) or (N_p, K, N)
        biases:   (K,) or (N_p, K)
        betas:    scalar or (B,)
        x0:       (N,) or (N_ic, N) or (N_p, N_ic, N)

    Returns:
        betas: (B,)
        xs:    (B, N_p, N_ic, N)
    """

    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1.")
    patterns, biases, betas, x0 = prepare_initial_conditions(patterns, biases, betas, x0)
    P, K, N = patterns.shape
    N_ic = x0.shape[-2]
    x_ic = x0.expand(P, N_ic, N).clone()

    x_coll = []
    if return_probs:
        probs_coll = []

    iterator = tqdm(
        enumerate(betas),
        total=len(betas),
        desc="Annealing",
        disable=not verbose,
    )

    for beta_idx, beta in iterator:
        res = deterministic_dynamics(
            patterns,
            biases,
            beta,
            x_ic, 
            num_iterations,
            return_probs=return_probs,
            verbose=False,
            dt=dt
        )
        # Remove beta axis, since beta is scalar here.
        if return_probs:
            x, probs = res
            x = x.squeeze(0)
            probs = probs.squeeze(0)
            x_coll.append(x.clone())
            probs_coll.append(probs.clone())
        else:
            x = res.squeeze(0)
            x_coll.append(x.clone())
        

        # No need to perturb after the final beta.
        if beta_idx < len(betas) - 1:
            if logit_noise_std > 0:
                logits = torch.einsum("sci,ski->sck", x, patterns)
                logits = beta * (logits + biases.view(P, 1, K))
                logits = logits + logit_noise_std * torch.randn_like(logits)

                perturbed_probs = torch.softmax(logits, dim=-1)
                x_ic = torch.einsum("sck,ski->sci", perturbed_probs, patterns)
            else:
                x_ic = x.clone()
    if return_probs:
        return torch.stack(x_coll, dim=0), torch.stack(probs_coll, dim=0)
    else:
        return torch.stack(x_coll, dim=0)

def dual_deterministic_dynamics_annealing(
    grams: torch.Tensor,
    biases: torch.Tensor,
    betas: Union[torch.Tensor, float],
    w0: torch.Tensor,
    logit_noise_std: float,
    num_iterations: int,
    verbose: bool = False,
    dt : Optional[float] = None
):
    """
    Annealed deterministic dynamics for the dual map.

    At each beta, compute the deterministic fixed point

        w = softmax(beta * (G w + b)).

    Then, before moving to the next beta, perturb the converged logits by
    Gaussian noise,

        logits -> logits + logit_noise_std * eta,

    and use the corresponding softmax weights as the next initial condition.

    Shapes:
        grams:  (K, K) or (N_G, K, K)
        biases: (K,) or (N_G, K)
        betas:  scalar or (B,)
        w0:     (K,), (N_ic, K), or (N_G, N_ic, K)

    Broadcasting:
        Singleton Gram-batch axes are broadcast to the common batch size.

    Returns:
        ws: (B, N_G, N_ic, K)
    """
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")

    if logit_noise_std < 0:
        raise ValueError("logit_noise_std must be non-negative")

    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")
    
    grams, biases, betas, w0 = prepare_initial_conditions_dual(grams, biases, betas, w0)
    P,K,K = grams.shape
    B = betas.numel()
    N_ic = w0.shape[-2]
    w_ic = w0.expand(P, N_ic, K).clone()

    w_coll = []

    iterator = tqdm(
        enumerate(betas),
        total=betas.numel(),
        desc="Dual annealing",
        disable=not verbose,
    )

    for beta_idx, beta in iterator:
        # dual_deterministic_dynamics introduces a singleton beta axis.
        w = dual_deterministic_dynamics(
            grams=grams,
            biases=biases,
            betas=beta,
            w0=w_ic,
            num_iterations=num_iterations,
            verbose=False,
            dt=dt
        ).squeeze(0)

        w_coll.append(w.clone())

        # No need to construct the next initial condition after the final beta.
        if beta_idx < betas.numel() - 1:
            if logit_noise_std > 0:
                gram_field = torch.einsum(
                    "pkh,pjh->pjk",
                    grams,
                    w,
                )
                logits = beta * (
                    gram_field + biases.reshape(P, 1, K)
                )
                logits = logits + logit_noise_std * torch.randn_like(logits)
                w_ic = torch.softmax(logits, dim=-1)
            else:
                w_ic = w.clone()

    return torch.stack(w_coll, dim=0)