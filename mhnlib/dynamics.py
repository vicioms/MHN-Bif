import torch
from torch import nn
from tqdm.auto import tqdm


def deterministic_dual_dynamics_update(w, gram, betas):
    S, B, K = w.shape
    logits = torch.matmul(w, gram)  # (S, B, K)
    logits.mul_(betas.view(1, B, 1))
    return torch.softmax(logits, dim=2)


def deterministic_dual_dynamics_with_biases(w, gram, betas, biases):
    S, B, K = w.shape
    logits = torch.matmul(w, gram)  # (S, B, K)
    logits.mul_(betas.view(1, B, 1))
    logits.add_(biases[None, None, :])
    return torch.softmax(logits, dim=2)


class DualDeterministicDynamics(nn.Module):
    def __init__(self, patterns, biases=None, requires_grad=False, dtype=torch.float32):
        super().__init__()

        patterns = torch.as_tensor(patterns, dtype=dtype)

        if patterns.ndim != 2:
            raise ValueError(f"patterns has shape {patterns.shape}, but expected (K, D).")

        K, D = patterns.shape

        if requires_grad:
            self.patterns = nn.Parameter(patterns)
        else:
            self.register_buffer("patterns", patterns)

        if biases is None:
            biases = torch.zeros(K, dtype=dtype)
            self.register_buffer("biases", biases)
            self.uses_biases = False
        else:
            biases = torch.as_tensor(biases, dtype=dtype)

            if biases.ndim != 1:
                raise ValueError(f"biases has shape {biases.shape}, but expected (K,).")
            if biases.shape[0] != K:
                raise ValueError(f"biases has shape {biases.shape}, but expected ({K},).")

            if requires_grad:
                self.biases = nn.Parameter(biases)
            else:
                self.register_buffer("biases", biases)

            self.uses_biases = True

    @property
    def gram(self):
        return self.patterns @ self.patterns.T

    def _prepare_betas(self, betas):
        betas = torch.as_tensor(
            betas,
            dtype=self.patterns.dtype,
            device=self.patterns.device,
        )

        if betas.ndim == 0:
            betas = betas.view(1)
        elif betas.ndim != 1:
            raise ValueError(f"betas has shape {betas.shape}, but expected scalar or (B,).")

        return betas

    def _prepare_w0(self, w0, betas):
        betas = self._prepare_betas(betas)

        w0 = torch.as_tensor(
            w0,
            dtype=self.patterns.dtype,
            device=self.patterns.device,
        )

        K_expected = self.patterns.shape[0]
        B = betas.shape[0]

        if w0.ndim == 1:
            K = w0.shape[0]
            if K != K_expected:
                raise ValueError(f"w0 has last dimension {K}, but expected {K_expected}.")
            w0 = w0.view(1, 1, K)  # (S=1, B=1, K)

        elif w0.ndim == 2:
            S, K = w0.shape
            if K != K_expected:
                raise ValueError(f"w0 has last dimension {K}, but expected {K_expected}.")
            w0 = w0[:, None, :]  # (S, B=1, K)

        elif w0.ndim == 3:
            S, B0, K = w0.shape
            if K != K_expected:
                raise ValueError(f"w0 has last dimension {K}, but expected {K_expected}.")
        else:
            raise ValueError(f"w0 has shape {w0.shape}, expected (K,), (S, K), or (S, B, K).")

        S, B0, K = w0.shape

        if B0 == 1 and B > 1:
            w0 = w0.expand(S, B, K)
        elif B == 1 and B0 > 1:
            betas = betas.expand(B0)
        elif B0 != B:
            raise ValueError(
                f"w0 has beta axis size {B0}, but betas has length {B}."
            )

        return w0.clone(), betas

    @torch.no_grad()
    def integrate(self, w0, betas, num_iterations, verbose=False, normalize=True):
        w, betas = self._prepare_w0(w0, betas)

        if normalize:
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        iterator = tqdm(range(num_iterations)) if verbose else range(num_iterations)

        gram = self.gram

        for _ in iterator:
            if self.uses_biases:
                w = deterministic_dual_dynamics_with_biases(
                    w=w,
                    gram=gram,
                    betas=betas,
                    biases=self.biases,
                )
            else:
                w = deterministic_dual_dynamics_update(
                    w=w,
                    gram=gram,
                    betas=betas,
                )

        return w


#class SpatialStochasticDynamics(nn.Module):
#    def __init__(self, patterns, sigma,  requires_grad=False, dtype =torch.float32):
#        super().__init__()
#        if patterns.ndim < 3:
#            raise ValueError(f"patterns has shape {patterns.shape}, but expected (K, C, (D_1, ..., D_n)).")
#        self.register_buffer("patterns", torch.as_tensor(patterns, dtype=dtype).requires_grad_(requires_grad))
#        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=dtype).requires_grad_(requires_grad))
#    def forward(self, x):
#        num_dims = self.patterns.ndim - 2
#        if x.ndim < 1 + num_dims:
#            raise ValueError(f"x has shape {x.shape}, but expected at least (C, (D_1, ..., D_n)).")
#        x_spatial_shape = x.shape[-num_dims:]
#        if x.shape[-num_dims:] != self.patterns.shape[-num_dims:]:
#            raise ValueError(f"x has last {num_dims} dimensions {x.shape[-num_dims:]}, but expected {self.patterns.shape[-num_dims:]}.")
#        if x.shape[-num_dims-1] != self.patterns.shape[-num_dims-1]:
#            raise ValueError(f"x has channel/feature dimension {x.shape[-num_dims-1]}, but expected {self.patterns.shape[-num_dims-1]}.")
#        if x.ndim == num_dims + 1:
#            x = x.unsqueeze(0)


class Dynamics(nn.Module):
    def __init__(self, patterns, biases, is_stochastic=False, requires_grad=False, dtype=torch.float32):
        super().__init__()

        patterns = torch.as_tensor(patterns, dtype=dtype)
        biases = torch.as_tensor(biases, dtype=dtype)

        if patterns.ndim != 2:
            raise ValueError(f"patterns has shape {patterns.shape}, but expected (K, D).")
        if biases.ndim != 1:
            raise ValueError(f"biases has shape {biases.shape}, but expected (K,).")
        if biases.shape[0] != patterns.shape[0]:
            raise ValueError(f"biases has shape {biases.shape}, but expected ({patterns.shape[0]},).")

        if requires_grad:
            self.patterns = nn.Parameter(patterns)
            self.biases = nn.Parameter(biases)
        else:
            self.register_buffer("patterns", patterns)
            self.register_buffer("biases", biases)

        self.is_stochastic = is_stochastic

    def _prepare_betas(self, betas):
        betas = torch.as_tensor(
            betas,
            dtype=self.patterns.dtype,
            device=self.patterns.device,
        )

        if betas.ndim == 0:
            betas = betas.view(1)
        elif betas.ndim != 1:
            raise ValueError(f"betas has shape {betas.shape}, but expected scalar or (B,).")

        return betas

    def _prepare_x_and_betas(self, x, betas, name="x"):
        betas = self._prepare_betas(betas)

        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.patterns.dtype, device=self.patterns.device)
        else:
            x = x.to(dtype=self.patterns.dtype, device=self.patterns.device)

        D_expected = self.patterns.shape[1]

        if x.ndim == 1:
            D = x.shape[0]
            if D != D_expected:
                raise ValueError(f"{name} has last dimension {D}, expected {D_expected}.")
            x = x.view(1, 1, D)

        elif x.ndim == 2:
            S, D = x.shape
            if D != D_expected:
                raise ValueError(f"{name} has last dimension {D}, expected {D_expected}.")
            x = x[:, None, :]  # (S, 1, D)

        elif x.ndim == 3:
            S, Bx, D = x.shape
            if D != D_expected:
                raise ValueError(f"{name} has last dimension {D}, expected {D_expected}.")

        else:
            raise ValueError(f"{name} has shape {x.shape}, expected (D,), (S, D), or (S, B, D).")

        S, Bx, D = x.shape
        Bb = betas.shape[0]

        if Bx == 1 and Bb > 1:
            x = x.expand(S, Bb, D)
        elif Bb == 1 and Bx > 1:
            betas = betas.expand(Bx)
        elif Bx != Bb:
            raise ValueError(
                f"Incompatible beta axis: {name} has beta axis size {Bx}, "
                f"but betas has shape {betas.shape}."
            )

        return x, betas

    def _forward(self, x, betas):
        # x:     (S, B, D)
        # betas: (B,)
        # output probs: (S, B, K)

        dot_products = torch.einsum("sbd,kd->sbk", x, self.patterns)

        logits = self.biases[None, None, :] + betas[None, :, None] * dot_products

        probs = torch.softmax(logits, dim=-1)

        return probs

    def forward(self, x, betas):
        x, betas = self._prepare_x_and_betas(x, betas, name="x")
        return self._forward(x, betas)

    @torch.no_grad()
    def integrate(self, x0, betas, dt, num_iterations, verbose=False):
        x, betas = self._prepare_x_and_betas(x0, betas, name="x0")
        x = x.clone()

        if self.is_stochastic:
            if torch.any(betas <= 0):
                raise ValueError("For stochastic dynamics, all betas must be positive because noise scale is sqrt(2 dt / beta).")

        iterator = tqdm(range(num_iterations)) if verbose else range(num_iterations)

        for _ in iterator:
            probs = self._forward(x, betas)

            drift = torch.einsum("sbk,kd->sbd", probs, self.patterns) - x

            x = x + dt * drift

            if self.is_stochastic:
                noise_scale = torch.sqrt(2.0 * dt / betas)[None, :, None]
                x = x + noise_scale * torch.randn_like(x)

        return x


    @torch.no_grad()
    def integrate_fixed_point(self, x0, betas, num_iterations, verbose=False):
        x, betas = self._prepare_x_and_betas(x0, betas, name="x0")
        x = x.clone()

        iterator = tqdm(range(num_iterations)) if verbose else range(num_iterations)

        for _ in iterator:
            probs = self._forward(x, betas)
            x =  torch.einsum("sbk,kd->sbd", probs, self.patterns)
        return x