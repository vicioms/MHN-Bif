import torch

#C(w) = diag(w) - w w^T
# J(w) = I - beta*C(w) @ G
@torch.no_grad()
def get_jacobian_gram(gram, w, beta):
    C_w = torch.diag(w) - torch.outer(w, w)
    J_w = torch.eye(gram.shape[0]) - beta * C_w @ gram
    return J_w
@torch.no_grad()
def get_symmetric_stability_matrix_gram(gram, w, return_proj = False):
    u = torch.sqrt(w)
    D_u = torch.diag(u)
    proj = torch.eye(gram.shape[0]) - torch.outer(u, u)
    full_proj = D_u @ proj
    symm_stability_matrix = full_proj  @ gram @ full_proj.T
    if return_proj:
        return symm_stability_matrix, full_proj
    else:
        return symm_stability_matrix

@torch.no_grad()
def get_entropies(w):
    h = -w * w.log()
    h[~torch.isfinite(h)] = 0.0
    h = h.sum(dim=-1)
    return h