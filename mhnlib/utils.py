import torch
import torch.linalg
from einops import einsum
from tqdm.auto import tqdm
from math import log, sqrt
from typing import Optional, Tuple, Union

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
def get_stability_matrix(gram, w):
    fisher = get_fisher_matrix(w)
    return torch.einsum("...hk,...kl->...hl", fisher, gram)
def get_symmetric_stability_matrix(gram, w, return_proj=False):
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

def deterministic_dynamics(
    patterns: torch.Tensor,
    biases: torch.Tensor,
    betas: Union[torch.Tensor, float],
    x0: torch.Tensor,
    num_iterations: int,
    return_probs: bool = False,
    verbose: bool = False,
):
    """
    Compute deterministic fixed points of

        x = P^T softmax(beta * (P x + b))

    Shapes:
        patterns: (K, N) or (N_p, K, N)
        biases:   (K,) or (N_p, K)
        betas:    scalar or (B,)
        x0:       (N,) or (N_ic, N) or (N_p, N_ic, N)

    Broadcasting:
        unbatched patterns/biases/x0 are broadcast over the pattern-batch axis.

    Returns:
        x:     (B, N_p, N_ic, N)
        probs: (B, N_p, N_ic, K) if return_probs=True
    """

    if patterns.ndim not in (2, 3):
        raise ValueError(
            f"patterns must have shape (K, N) or (N_p, K, N), got {patterns.shape}"
        )

    if biases.ndim not in (1, 2):
        raise ValueError(
            f"biases must have shape (K,) or (N_p, K), got {biases.shape}"
        )

    if x0.ndim not in (1, 2, 3):
        raise ValueError(
            f"x0 must have shape (N,), (N_ic, N), or (N_p, N_ic, N), got {x0.shape}"
        )

    device = patterns.device
    dtype = patterns.dtype

    biases = torch.as_tensor(biases, dtype=dtype, device=device)
    x0 = torch.as_tensor(x0, dtype=dtype, device=device)
    betas = torch.as_tensor(betas, dtype=dtype, device=device)

    K, N = patterns.shape[-2], patterns.shape[-1]

    if biases.shape[-1] != K:
        raise ValueError(
            f"last dimension of biases must match number of patterns K={K}, got {biases.shape}"
        )

    if x0.shape[-1] != N:
        raise ValueError(
            f"last dimension of x0 must match pattern dimension N={N}, got {x0.shape}"
        )

    if betas.ndim == 0:
        betas = betas.view(1)
    elif betas.ndim != 1:
        raise ValueError(f"betas must be scalar or 1D tensor, got {betas.shape}")

    if patterns.ndim == 2:
        patterns = patterns.unsqueeze(0)      # (1, K, N)

    if biases.ndim == 1:
        biases = biases.unsqueeze(0)          # (1, K)

    if x0.ndim == 1:
        x0 = x0.view(1, 1, N)                 # (1, 1, N)
    elif x0.ndim == 2:
        x0 = x0.unsqueeze(0)                  # (1, N_ic, N)

    N_p_patterns = patterns.shape[0]
    N_p_biases = biases.shape[0]
    N_p_x0 = x0.shape[0]

    N_p = max(N_p_patterns, N_p_biases, N_p_x0)

    for name, n in [
        ("patterns", N_p_patterns),
        ("biases", N_p_biases),
        ("x0", N_p_x0),
    ]:
        if n not in (1, N_p):
            raise ValueError(
                f"{name} has incompatible batch dimension {n}; expected 1 or {N_p}"
            )

    patterns = patterns.expand(N_p, K, N)
    biases = biases.expand(N_p, K)
    x0 = x0.expand(N_p, x0.shape[1], N)

    B = betas.shape[0]
    N_ic = x0.shape[1]

    x = x0.unsqueeze(0).expand(B, N_p, N_ic, N).clone()

    betas_view = betas.view(B, 1, 1, 1)
    biases_view = biases.view(1, N_p, 1, K)

    iterator = tqdm(
        range(num_iterations),
        desc="Computing fixed points",
        disable=not verbose,
    )

    for _ in iterator:
        logits = torch.einsum("bsci,ski->bsck", x, patterns)
        logits = betas_view * (logits + biases_view)
        probs = torch.softmax(logits, dim=-1)
        x = torch.einsum("bsck,ski->bsci", probs, patterns)

    if return_probs:
        logits = torch.einsum("bsci,ski->bsck", x, patterns)
        logits = betas_view * (logits + biases_view)
        probs = torch.softmax(logits, dim=-1)
        return x, probs

    return x

def deterministic_dynamics_annealing(
    patterns: torch.Tensor,
    biases: torch.Tensor,
    betas: Union[torch.Tensor, float],
    x0: torch.Tensor,
    logit_noise_std: float,
    num_iterations: int,
    return_probs: bool = False,
    verbose: bool = False,
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

    if patterns.ndim not in (2, 3):
        raise ValueError(
            f"patterns must have shape (K, N) or (N_p, K, N), got {patterns.shape}"
        )

    if biases.ndim not in (1, 2):
        raise ValueError(
            f"biases must have shape (K,) or (N_p, K), got {biases.shape}"
        )

    if x0.ndim not in (1, 2, 3):
        raise ValueError(
            f"x0 must have shape (N,), (N_ic, N), or (N_p, N_ic, N), got {x0.shape}"
        )

    device = patterns.device
    dtype = patterns.dtype

    biases = torch.as_tensor(biases, dtype=dtype, device=device)
    x0 = torch.as_tensor(x0, dtype=dtype, device=device)
    betas = torch.as_tensor(betas, dtype=dtype, device=device)

    if betas.ndim == 0:
        betas = betas.view(1)
    elif betas.ndim != 1:
        raise ValueError(f"betas must be scalar or 1D tensor, got {betas.shape}")

    K, N = patterns.shape[-2], patterns.shape[-1]

    if biases.shape[-1] != K:
        raise ValueError(
            f"last dimension of biases must match K={K}, got {biases.shape}"
        )

    if x0.shape[-1] != N:
        raise ValueError(
            f"last dimension of x0 must match N={N}, got {x0.shape}"
        )

    if patterns.ndim == 2:
        patterns = patterns.unsqueeze(0)      # (1, K, N)

    if biases.ndim == 1:
        biases = biases.unsqueeze(0)          # (1, K)

    if x0.ndim == 1:
        x0 = x0.view(1, 1, N)                 # (1, 1, N)
    elif x0.ndim == 2:
        x0 = x0.unsqueeze(0)                  # (1, N_ic, N)

    N_p_patterns = patterns.shape[0]
    N_p_biases = biases.shape[0]
    N_p_x0 = x0.shape[0]

    N_p = max(N_p_patterns, N_p_biases, N_p_x0)

    for name, n in [
        ("patterns", N_p_patterns),
        ("biases", N_p_biases),
        ("x0", N_p_x0),
    ]:
        if n not in (1, N_p):
            raise ValueError(
                f"{name} has incompatible batch dimension {n}; expected 1 or {N_p}"
            )

    patterns = patterns.expand(N_p, K, N)
    biases = biases.expand(N_p, K)
    x_ic = x0.expand(N_p, x0.shape[1], N).clone()

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
                logits = beta * (logits + biases.view(N_p, 1, K))
                logits = logits + logit_noise_std * torch.randn_like(logits)

                perturbed_probs = torch.softmax(logits, dim=-1)
                x_ic = torch.einsum("sck,ski->sci", perturbed_probs, patterns)
            else:
                x_ic = x.clone()
    if return_probs:
        return torch.stack(x_coll, dim=0), torch.stack(probs_coll, dim=0)
    else:
        return torch.stack(x_coll, dim=0)

def stochastic_dynamics(
    patterns: torch.Tensor,
    biases: torch.Tensor,
    betas: Union[torch.Tensor, float],
    temp0 : float,
    use_beta_as_inverse_temp: bool,
    x0: torch.Tensor,
    dt : float,
    num_iterations: int,
    return_probs: bool = False,
    verbose: bool = False,
):
    """
    Run the dynamics fixed points of

        dx/dt = P^T softmax(beta * (P x + b)) + sqrt(2 * temp) * eta
    
    with eta a white noise. The temperature is a single scalar. 

    Two possible use cases:
    1) If use_beta_as_inverse_temp=False, temp is a scalar independent of beta, and the dynamics is a noisy version of the deterministic dynamics at each beta.
    2) If use_beta_as_inverse_temp=True, temp = 1 / beta for each beta, so the noise strength decreases as beta increases.

    Shapes:
        patterns: (K, N) or (N_p, K, N)
        biases:   (K,) or (N_p, K)
        betas:    scalar or (B,)
        x0:       (N,) or (N_ic, N) or (N_p, N_ic, N)

    Broadcasting:
        unbatched patterns/biases/x0 are broadcast over the pattern-batch axis.

    Returns:
        x:     (B, N_p, N_ic, N)
        probs: (B, N_p, N_ic, K) if return_probs=True
    """

    
    if patterns.ndim not in (2, 3):
        raise ValueError(
            f"patterns must have shape (K, N) or (N_p, K, N), got {patterns.shape}"
        )

    if biases.ndim not in (1, 2):
        raise ValueError(
            f"biases must have shape (K,) or (N_p, K), got {biases.shape}"
        )

    if x0.ndim not in (1, 2, 3):
        raise ValueError(
            f"x0 must have shape (N,), (N_ic, N), or (N_p, N_ic, N), got {x0.shape}"
        )

    device = patterns.device
    dtype = patterns.dtype

    biases = torch.as_tensor(biases, dtype=dtype, device=device)
    x0 = torch.as_tensor(x0, dtype=dtype, device=device)
    betas = torch.as_tensor(betas, dtype=dtype, device=device)

    K, N = patterns.shape[-2], patterns.shape[-1]

    if biases.shape[-1] != K:
        raise ValueError(
            f"last dimension of biases must match number of patterns K={K}, got {biases.shape}"
        )

    if x0.shape[-1] != N:
        raise ValueError(
            f"last dimension of x0 must match pattern dimension N={N}, got {x0.shape}"
        )

    if betas.ndim == 0:
        betas = betas.view(1)
    elif betas.ndim != 1:
        raise ValueError(f"betas must be scalar or 1D tensor, got {betas.shape}")

    if patterns.ndim == 2:
        patterns = patterns.unsqueeze(0)      # (1, K, N)

    if biases.ndim == 1:
        biases = biases.unsqueeze(0)          # (1, K)

    if x0.ndim == 1:
        x0 = x0.view(1, 1, N)                 # (1, 1, N)
    elif x0.ndim == 2:
        x0 = x0.unsqueeze(0)                  # (1, N_ic, N)

    N_p_patterns = patterns.shape[0]
    N_p_biases = biases.shape[0]
    N_p_x0 = x0.shape[0]

    N_p = max(N_p_patterns, N_p_biases, N_p_x0)

    for name, n in [
        ("patterns", N_p_patterns),
        ("biases", N_p_biases),
        ("x0", N_p_x0),
    ]:
        if n not in (1, N_p):
            raise ValueError(
                f"{name} has incompatible batch dimension {n}; expected 1 or {N_p}"
            )
    patterns = patterns.expand(N_p, K, N)
    biases = biases.expand(N_p, K)
    x0 = x0.expand(N_p, x0.shape[1], N)

    B = betas.shape[0]
    N_ic = x0.shape[1]

    x = x0.unsqueeze(0).expand(B, N_p, N_ic, N).clone()

    betas_view = betas.view(B, 1, 1, 1)
    biases_view = biases.view(1, N_p, 1, K)

    if use_beta_as_inverse_temp:
        noise_strength  = torch.sqrt(2 * dt / betas_view)
    else:
        noise_strength = torch.sqrt(torch.as_tensor(2 * temp0 * dt, dtype=dtype, device=device)).view(1, 1, 1, 1)

    iterator = tqdm(
        range(num_iterations),
        desc="Computing fixed points",
        disable=not verbose,
    )

    for _ in iterator:
        logits = torch.einsum("bsci,ski->bsck", x, patterns)
        logits = betas_view * (logits + biases_view)
        probs = torch.softmax(logits, dim=-1)
        x +=  dt*(torch.einsum("bsck,ski->bsci", probs, patterns)-x) + noise_strength * torch.randn_like(x)

    if return_probs:
        logits = torch.einsum("bsci,ski->bsck", x, patterns)
        logits = betas_view * (logits + biases_view)
        probs = torch.softmax(logits, dim=-1)
        return x, probs

    return x

#@torch.no_grad()
#def get_jacobian_gram(gram, w, beta):
#    C_w = torch.diag(w) - torch.outer(w, w)
#    J_w = torch.eye(gram.shape[0]) - beta * C_w @ gram
#    return J_w
#
#
#@torch.no_grad()
#def get_symmetric_stability_matrix_gram(gram, w, return_proj=False):
#    w = w / w.sum()
#
#    u = torch.sqrt(w)
#    D = torch.diag(u)
#
#    P = torch.eye(len(w), device=w.device, dtype=w.dtype) - torch.outer(u, u)
#
#    A = P @ D
#    M = A @ gram @ A.T
#    M = 0.5 * (M + M.T)  # numerical symmetrization
#
#    if return_proj:
#        return M, A
#    return M
#
#
#@torch.no_grad()
#def get_entropies(w):
#    h = -w * w.log()
#    h[~torch.isfinite(h)] = 0.0
#    h = h.sum(dim=-1)
#    return h
#
#@torch.no_grad()
#def get_symmetric_stability_matrices_gram(
#    gram,
#    w_s,
#    return_proj=False,
#    return_fisher=False,
#    eps=1e-12,
#):
#    """
#    Batched symmetric stability matrices for
#
#        w_next = softmax(beta * gram @ w)
#
#    For each fixed point w_b, returns
#
#        M_b = C_b^{1/2} G_b C_b^{1/2}
#
#    where
#
#        C_b = diag(w_b) - w_b w_b^T.
#
#    Then
#
#        beta_c = 1 / lambda_max(M_b).
#    """
#
#    if w_s.ndim == 1:
#        w_s = w_s[None, :]
#
#    if w_s.ndim != 2:
#        raise ValueError(f"w_s must have shape (B, K) or (K,), got {w_s.shape}")
#
#    # Important: normalize probabilities
#    w_s = w_s / w_s.sum(dim=-1, keepdim=True)
#
#    B, K = w_s.shape
#
#    if gram.ndim == 2:
#        if gram.shape != (K, K):
#            raise ValueError(f"gram has shape {gram.shape}, expected {(K, K)}")
#        gram_b = gram[None, :, :].expand(B, K, K)
#
#    elif gram.ndim == 3:
#        if gram.shape[-2:] != (K, K):
#            raise ValueError(f"gram has shape {gram.shape}, expected (..., {K}, {K})")
#        if gram.shape[0] == 1:
#            gram_b = gram.expand(B, K, K)
#        elif gram.shape[0] == B:
#            gram_b = gram
#        else:
#            raise ValueError(
#                f"batched gram has batch {gram.shape[0]}, but w_s has batch {B}"
#            )
#    else:
#        raise ValueError(f"gram must have shape (K, K) or (B, K, K), got {gram.shape}")
#
#    # Optional numerical symmetrization of gram
#    gram_b = 0.5 * (gram_b + gram_b.transpose(-1, -2))
#
#    # Fisher / softmax covariance
#    C = torch.diag_embed(w_s) - w_s[:, :, None] * w_s[:, None, :]
#
#    # Symmetric eigendecomposition
#    evals, evecs = torch.linalg.eigh(C)
#
#    # Fisher square root
#    evals_pos = evals.clamp_min(0.0)
#    sqrt_evals = torch.sqrt(evals_pos)
#
#    C_sqrt = (evecs * sqrt_evals[:, None, :]) @ evecs.transpose(-1, -2)
#
#    # Symmetric stability matrix
#    stab_m = C_sqrt @ gram_b @ C_sqrt
#    stab_m = 0.5 * (stab_m + stab_m.transpose(-1, -2))
#
#    if not return_proj and not return_fisher:
#        return stab_m
#
#    outputs = [stab_m]
#
#    if return_proj:
#        mask = evals > eps
#        proj_m = (evecs * mask[:, None, :].to(evecs.dtype)) @ evecs.transpose(-1, -2)
#        outputs.append(proj_m)
#
#    if return_fisher:
#        invsqrt_evals = torch.zeros_like(evals)
#        mask = evals > eps
#        invsqrt_evals[mask] = torch.rsqrt(evals[mask])
#
#        C_invsqrt = (evecs * invsqrt_evals[:, None, :]) @ evecs.transpose(-1, -2)
#
#        info = {
#            "fisher": C,
#            "fisher_evals": evals,
#            "fisher_evecs": evecs,
#            "fisher_sqrt": C_sqrt,
#            "fisher_invsqrt": C_invsqrt,
#        }
#        outputs.append(info)
#
#    return tuple(outputs)