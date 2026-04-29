import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def run_dynamics(beta0, betas, patterns, dt, num_steps, num_runs=1, verbose=False):
    device = patterns.device
    dtype = patterns.dtype

    betas = betas.to(device=device, dtype=dtype)
    K, N = patterns.shape
    B = betas.numel()

    x = torch.randn((num_runs, B, N), device=device, dtype=dtype) / beta0**0.5

    P = patterns.contiguous()
    P_T = P.T.contiguous()
    beta_view = betas[None, :, None]


    for _ in tqdm(range(num_steps), disable=not verbose):
        logits = torch.matmul(x, P_T)
        logits.mul_(beta_view)

        weights = torch.softmax(logits, dim=-1)
        force = torch.matmul(weights, P)

        force.sub_(x)
        force.mul_(dt)
        x.add_(force)

    logits = torch.matmul(x, P_T)
    logits.mul_(beta_view)
    weights = torch.softmax(logits, dim=-1)

    return x, weights


@torch.no_grad()
def weights_fixed_points(betas, gram_matrix, num_steps, num_runs=1,verbose=False):
    device = gram_matrix.device
    dtype = gram_matrix.dtype

    B = betas.numel()
    K = gram_matrix.shape[0]
    w = torch.rand((num_runs, B, K), device=device, dtype=dtype)
    w = w / w.sum(dim=-1, keepdim=True)
    beta_view = betas[None, :, None] 
    for _ in tqdm(range(num_steps), disable=not verbose):
        logits = torch.matmul(w, gram_matrix)
        logits.mul_(beta_view)
        w = torch.softmax(logits, dim=-1)
    if num_runs == 1:
        w = w.squeeze(0)
    return w