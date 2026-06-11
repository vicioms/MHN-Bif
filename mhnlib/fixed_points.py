import torch

#C(w) = diag(w) - w w^T
# J(w) = I - beta*C(w) @ G
@torch.no_grad()
def get_jacobian_gram(gram, w, beta):
    C_w = torch.diag(w) - torch.outer(w, w)
    J_w = torch.eye(gram.shape[0]) - beta * C_w @ gram
    return J_w


@torch.no_grad()
def get_symmetric_stability_matrix_gram(gram, w, return_proj=False):
    w = w / w.sum()

    u = torch.sqrt(w)
    D = torch.diag(u)
    P = torch.eye(len(w), device=w.device, dtype=w.dtype) - torch.outer(u, u)

    A = P @ D
    M = A @ gram @ A.T

    if return_proj:
        return M, A
    return M

@torch.no_grad()
def get_entropies(w):
    h = -w * w.log()
    h[~torch.isfinite(h)] = 0.0
    h = h.sum(dim=-1)
    return h

import torch


@torch.no_grad()
def get_symmetric_stability_matrices_gram(
    gram,
    w_s,
    return_proj=False,
    return_fisher=False,
    eps=1e-12,
):
    """
    Batched version for many fixed points w_s.

    Parameters
    ----------
    gram : Tensor
        Shape (K, K) or (B, K, K).
    w_s : Tensor
        Shape (B, K) or (K,).
    return_proj : bool
        If True, also returns the tangent-space projectors.
    return_fisher : bool
        If True, also returns Fisher matrices and their square roots.
    eps : float
        Eigenvalue cutoff for the Fisher tangent space.

    Returns
    -------
    stab_m : Tensor
        Shape (B, K, K), with

            S_b = C_b^{1/2} G C_b^{1/2}

        where

            C_b = diag(w_b) - w_b w_b^T.

    proj_m : Tensor, optional
        Shape (B, K, K). Projector onto the numerically non-null Fisher subspace.

    info : dict, optional
        Returned if return_fisher=True.
    """

    if w_s.ndim == 1:
        w_s = w_s[None, :]

    if w_s.ndim != 2:
        raise ValueError(f"w_s must have shape (B, K) or (K,), got {w_s.shape}")

    B, K = w_s.shape

    if gram.ndim == 2:
        if gram.shape != (K, K):
            raise ValueError(f"gram has shape {gram.shape}, expected {(K, K)}")
        gram_b = gram[None, :, :].expand(B, K, K)

    elif gram.ndim == 3:
        if gram.shape[-2:] != (K, K):
            raise ValueError(f"gram has shape {gram.shape}, expected (..., {K}, {K})")
        if gram.shape[0] == 1:
            gram_b = gram.expand(B, K, K)
        elif gram.shape[0] == B:
            gram_b = gram
        else:
            raise ValueError(
                f"batched gram has batch {gram.shape[0]}, but w_s has batch {B}"
            )
    else:
        raise ValueError(f"gram must have shape (K, K) or (B, K, K), got {gram.shape}")

    # Fisher / softmax covariance
    C = torch.diag_embed(w_s) - w_s[:, :, None] * w_s[:, None, :]

    # Symmetric eigendecomposition, batched
    evals, evecs = torch.linalg.eigh(C)

    # Fisher square root
    evals_pos = evals.clamp_min(0.0)
    sqrt_evals = torch.sqrt(evals_pos)

    C_sqrt = (evecs * sqrt_evals[:, None, :]) @ evecs.transpose(-1, -2)

    # Symmetric stability matrix
    stab_m = C_sqrt @ gram_b @ C_sqrt

    if not return_proj and not return_fisher:
        return stab_m

    outputs = [stab_m]

    if return_proj:
        mask = evals > eps
        proj_m = (evecs * mask[:, None, :].to(evecs.dtype)) @ evecs.transpose(-1, -2)
        outputs.append(proj_m)

    if return_fisher:
        invsqrt_evals = torch.zeros_like(evals)
        mask = evals > eps
        invsqrt_evals[mask] = torch.rsqrt(evals[mask])

        C_invsqrt = (evecs * invsqrt_evals[:, None, :]) @ evecs.transpose(-1, -2)

        info = {
            "fisher": C,
            "fisher_evals": evals,
            "fisher_evecs": evecs,
            "fisher_sqrt": C_sqrt,
            "fisher_invsqrt": C_invsqrt,
        }
        outputs.append(info)

    return tuple(outputs)
    