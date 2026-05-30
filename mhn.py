import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from sklearn.cluster import KMeans

class MHN(nn.Module):
    def __init__(self, input_dim, num_patterns, learn_biases=True, log_beta_mean = 0.0, log_beta_std=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_patterns = num_patterns
        self.patterns = nn.Parameter(
            torch.randn(num_patterns, input_dim),
            requires_grad=True
        )
        self.biases = nn.Parameter(
            torch.zeros(num_patterns),
            requires_grad=learn_biases
        )
        self.register_buffer("log_beta_mean", torch.tensor(log_beta_mean, requires_grad=False))
        self.register_buffer("log_beta_std", torch.tensor(log_beta_std, requires_grad=False))
    
    @torch.no_grad()
    def center_biases(self):
        self.biases -= self.biases.mean()
    def forward(self, x, betas = None, num_betas = None, teacher_student_mode = False):
        N, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"{x} has last dimension {D}, "f"but expected {self.input_dim}.")
        if betas is None:
            if num_betas is None:
                num_betas = 1
            betas = torch.exp(torch.randn(num_betas, device=x.device, dtype=x.dtype)*self.log_beta_std + self.log_beta_std)
        num_betas = betas.shape[0]
        dot_products = x @ self.patterns.T # (N,K)
        patterns_pot = 0.5*torch.sum(self.patterns**2, dim=1)
        logits_model = self.biases[None,:] - patterns_pot[None,:]*betas[:,None]
        log_Z_model = torch.logsumexp(logits_model, dim=1) #(num_betas,)
        if teacher_student_mode:
            with torch.no_grad():
                teacher_logits = betas[:, None] * 0.5 * (x**2).sum(dim=1)[None, :]  # (B, N)
                teacher_probs = torch.softmax(teacher_logits, dim=1)

                idx = torch.multinomial(
                    teacher_probs,
                    num_samples=N,
                    replacement=True)  # (B, N)

                centers = x[idx]  # (B, N, D)
                noise = torch.randn_like(centers) / torch.sqrt(betas)[:, None, None]
                x_teacher = centers + noise

            dot_products_teacher = torch.einsum(
                "bnd,kd->bnk",
                x_teacher,
            self.patterns)

            logits_data = (
                self.biases[None, None, :]
                + betas[:, None, None] * dot_products_teacher)
        else:
            logits_data = (
            self.biases[None, None, :]
            + betas[:, None, None] * dot_products[None, :, :])
        log_Z_data = torch.logsumexp(logits_data, dim=2) # (num_betas, N)
        return log_Z_model, log_Z_data

@torch.no_grad()
def get_critical_beta(patterns, biases):
    w0 = torch.softmax(biases, dim=0)  # (K,)
    u = torch.sqrt(w0)

    K = patterns.shape[0]
    I = torch.eye(K, device=patterns.device, dtype=patterns.dtype)

    proj = I - torch.outer(u, u)

    W_half = torch.diag(torch.sqrt(w0))
    half_stab_matrix = proj @ W_half @ self.patterns

    stability_matrix = half_stab_matrix @ half_stab_matrix.T

    vals = torch.linalg.eigvalsh(stability_matrix)
    beta = 1.0 / vals.max()

    return beta

@torch.no_grad()
def get_entropies(w):
    h = -w * w.log()
    h[~torch.isfinite(h)] = 0.0
    h = h.sum(dim=-1)
    return h

@torch.no_grad()
def denoise_dynamics(samples, beta, patterns, biases, dt, num_steps, verbose=False):
    device = samples.device
    dtype = samples.dtype

    beta = beta.to(device=device, dtype=dtype)
    K, N = patterns.shape

    x = samples.clone()

    P = patterns.contiguous()
    P_T = P.T.contiguous()

    for _ in tqdm(range(num_steps), disable=not verbose):
        logits = torch.matmul(x, P_T)
        logits.mul_(beta)

        weights = torch.softmax(logits + biases[None, :], dim=-1)
        force = torch.matmul(weights, P)

        force.sub_(x)
        force.mul_(dt)
        x.add_(force)

    return x


@torch.no_grad()
def annealing_uniform_init(
    betas,
    patterns,
    dt,
    num_steps,
    num_runs=1,
    biases=None,
    verbose=False,
):
    device = patterns.device
    dtype = patterns.dtype

    betas = betas.to(device=device, dtype=dtype)
    K, N = patterns.shape
    B = betas.numel()

    if biases is None:
        biases = torch.zeros(K, device=device, dtype=dtype)
    else:
        biases = biases.to(device=device, dtype=dtype)

    x_fp = torch.zeros((B, num_runs, N), device=device, dtype=dtype)
    x_fp[0] = patterns.mean(dim=0)

    P = patterns.contiguous()
    P_T = P.T.contiguous()

    for beta_idx, beta in tqdm(enumerate(betas), total=B, disable=not verbose):
        if beta_idx == 0:
            x = x_fp[0]
        else:
            x = x_fp[beta_idx - 1].clone()

        for _ in range(num_steps):
            logits = beta * (x @ P_T) + biases[None, :]
            weights = torch.softmax(logits, dim=-1)
            force = weights @ P - x
            x += dt * force
            x += torch.sqrt(dt*2/beta)*torch.randn_like(x)
        x_fp[beta_idx] = x

    return x_fp.permute(1, 0, 2)

    

    
@torch.no_grad()
def dynamics_gaussian_init(beta0, betas, patterns, dt, num_steps, num_runs=1, verbose=False):
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
def dynamics_gaussian_init_with_biases(beta0, betas, patterns, biases, dt, num_steps, num_runs=1, verbose=False):
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

        weights = torch.softmax(logits + biases[None, :], dim=-1)
        force = torch.matmul(weights, P)

        force.sub_(x)
        force.mul_(dt)
        x.add_(force)

    logits = torch.matmul(x, P_T)
    logits.mul_(beta_view)
    weights = torch.softmax(logits + biases[None, :], dim=-1)

    return x, weights

@torch.no_grad()
def wfp_gram_uniform_init(betas, gram_matrix, num_steps, verbose=False):
    device = gram_matrix.device
    dtype = gram_matrix.dtype

    B = betas.numel()
    K = gram_matrix.shape[0]
    w = torch.ones((B, K), device=device, dtype=dtype)
    w = w / w.sum(dim=-1, keepdim=True)
    beta_view = betas[:, None] 
    for _ in tqdm(range(num_steps), disable=not verbose):
        logits = torch.matmul(w, gram_matrix)
        logits.mul_(beta_view)
        w = torch.softmax(logits, dim=-1)
    return w


@torch.no_grad()
def wfp_gram_dirichlet_init(betas, gram_matrix, num_steps,dirichlet_concentration_param = 1, num_runs=1,verbose=False):
    device = gram_matrix.device
    dtype = gram_matrix.dtype
    B = betas.numel()
    K = gram_matrix.shape[0]

    if device.type == 'mps':
        w = torch.distributions.Dirichlet(dirichlet_concentration_param * torch.ones(K, device='cpu', dtype=dtype)).sample((num_runs, B)).to(device=device)
    else:
        w = torch.distributions.Dirichlet(dirichlet_concentration_param * torch.ones(K, device=device, dtype=dtype)).sample((num_runs, B))    
    beta_view = betas[None, :, None] 
    for _ in tqdm(range(num_steps), disable=not verbose):
        logits = torch.matmul(w, gram_matrix)
        logits.mul_(beta_view)
        w = torch.softmax(logits, dim=-1)
    if num_runs == 1:
        w = w.squeeze(0)
    return w


