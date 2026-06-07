import torch
from torch import nn
from tqdm.auto import tqdm

def deterministic_dual_dynamics_update(w, gram, betas):
    S, B, K = w.shape
    logits = torch.matmul(w, gram) # (S, B, K)
    logits.mul_(betas.view(1,B,1))
    return torch.softmax(logits, dim=2)

def deterministic_dual_dynamics_with_biases(w, gram, betas, biases):
    S, B, K = w.shape
    logits = torch.matmul(w, gram) # (S, B, K)
    logits.mul_(betas.view(1,B,1))
    logits.add_(biases[None, None, :])
    return torch.softmax(logits, dim=2)

class SpatialStochasticDynamics(nn.Module):
    def __init__(self, patterns, sigma,  requires_grad=False, dtype =torch.float32):
        super().__init__()
        if patterns.ndim < 3:
            raise ValueError(f"patterns has shape {patterns.shape}, but expected (K, C, (D_1, ..., D_n)).")
        self.register_buffer("patterns", torch.as_tensor(patterns, dtype=dtype).requires_grad_(requires_grad))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=dtype).requires_grad_(requires_grad))
    def forward(self, x):
        num_dims = self.patterns.ndim - 2
        if x.ndim < 1 + num_dims:
            raise ValueError(f"x has shape {x.shape}, but expected at least (C, (D_1, ..., D_n)).")
        x_spatial_shape = x.shape[-num_dims:]
        if x.shape[-num_dims:] != self.patterns.shape[-num_dims:]:
            raise ValueError(f"x has last {num_dims} dimensions {x.shape[-num_dims:]}, but expected {self.patterns.shape[-num_dims:]}.")
        if x.shape[-num_dims-1] != self.patterns.shape[-num_dims-1]:
            raise ValueError(f"x has channel/feature dimension {x.shape[-num_dims-1]}, but expected {self.patterns.shape[-num_dims-1]}.")
        if x.ndim == num_dims + 1:
            x = x.unsqueeze(0)
        
class StochasticDynamics(nn.Module):
    def __init__(self, patterns, biases, requires_grad=False, dtype =torch.float32):
        super().__init__()
        if patterns.ndim != 2:
            raise ValueError(f"patterns has shape {patterns.shape}, but expected (K, D).")
        if biases.ndim != 1:
            raise ValueError(f"biases has shape {biases.shape}, but expected (K,).")
        if biases.shape[0] != patterns.shape[0]:
            raise ValueError(f"biases has shape {biases.shape}, but expected ({patterns.shape[0]},).")
        self.register_buffer("patterns", torch.as_tensor(patterns, dtype=dtype).requires_grad_(requires_grad))
        self.register_buffer("biases", torch.as_tensor(biases, dtype=dtype).requires_grad_(requires_grad))
    
    def _forward(self, x, betas):
        dot_products = torch.einsum("bnd,kd->bnk", x, self.patterns) # (B, N, K)
        logits = self.biases[None, None, :] + betas[:, None, None] * dot_products
        probs = torch.softmax(logits, dim=2)
        return probs
    def forward(self, x, betas):
        if x.ndim == 2:
            B, D = x.shape
            S = 1
            x = x.view(1, B, D)
        elif x.ndim == 3:
            S, B, D = x.shape
        else:
            raise ValueError(f"x has shape {x.shape}, but expected (B, D) or (S, B, D).")
        if D != self.patterns.shape[1]:
            raise ValueError(f"x has last dimension {D}, but expected {self.patterns.shape[1]}.")
        if betas.ndim != 1:
            raise ValueError(f"betas has shape {betas.shape}, but expected (B,).")
        if betas.shape[0] != B:
            raise ValueError(f"betas has shape {betas.shape}, but expected ({B},).")
        probs = self._forward(x, betas)
        return probs
    def integrate(self, x0, betas, dt, num_iterations, verbose = False):
        if x0.ndim == 2:
            S, D = x0.shape
            x0 = x0.view(S, 1, D)
        elif x0.ndim == 3:
            S, B, D = x0.shape
        else:
            raise ValueError(f"x0 has shape {x0.shape}, but expected (B, D) or (S, B, D).")
        if D != self.patterns.shape[1]:
            raise ValueError(f"x0 has last dimension {D}, but expected {self.patterns.shape[1]}.")
        if betas.ndim != 1:
            raise ValueError(f"betas has shape {betas.shape}, but expected (B,).")
        if betas.shape[0] != B:
            raise ValueError(f"betas has shape {betas.shape}, but expected ({B},).")
        x = x0
        if verbose:
            iterator = tqdm(range(num_iterations))
        else:
            iterator = range(num_iterations)  
        for step in iterator:
            white_noise = torch.randn(S, B, D, device=x0.device, dtype=x0.dtype)*(2*dt/torch.sqrt(betas))[None, :, None]
            probs = self._forward(x, betas)
            x += (torch.einsum("bnk,kd->bnd", probs, self.patterns)-x)*dt + white_noise
        return x
class DualDeterministicDynamics(nn.Module):
    def __init__(self, patterns, biases=None, requires_grad=False, dtype =torch.float32):
        super().__init__()
        if patterns.ndim != 2:
            raise ValueError(f"patterns has shape {patterns.shape}, but expected (K, D).")
        self.register_buffer("patterns", torch.as_tensor(patterns, dtype=dtype).requires_grad_(requires_grad))
        if biases is None:
            self.register_buffer("biases", torch.zeros(patterns.shape[0], dtype=dtype, requires_grad=False))
            self.uses_biases = False
        else:
            if biases.ndim != 1:
                raise ValueError(f"biases has shape {biases.shape}, but expected (K,).")
            if biases.shape[0] != patterns.shape[0]:
                raise ValueError(f"biases has shape {biases.shape}, but expected ({patterns.shape[0]},).")
            self.register_buffer("biases", torch.as_tensor(biases, dtype=dtype).requires_grad_(requires_grad))
            self.uses_biases = True
        self.register_buffer("gram", self.patterns @ self.patterns.T)
    
    @torch.no_grad()
    def integrate(self, w0, betas, num_iterations, verbose = False):
        if w0.ndim == 1:
            K = w0.shape[0]
            w0 = w0.view(1, K)
            S = 1
        elif w0.ndim == 2:
            S, K = w0.shape
        B = betas.shape[0]
        if K != self.patterns.shape[0]:
            raise ValueError(f"w has last dimension {K}, but expected {self.patterns.shape[0]}.")
        gram = self.patterns @ self.patterns.T
        w = torch.repeat_interleave(w0[:,None,:], repeats=B, dim=1) # (S, B, K)
        if verbose:
            iterator = tqdm(range(num_iterations))
        else:
            iterator = range(num_iterations)    
        if self.uses_biases:
            for step in iterator:
                w.copy_(deterministic_dual_dynamics_with_biases(w, gram, betas, self.biases))
        else:
            for step in iterator:
                w.copy_(deterministic_dual_dynamics_update(w, gram, betas))
        return w