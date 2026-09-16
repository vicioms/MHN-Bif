import torch
from math import sqrt
from tqdm.auto import tqdm
class Dynamics:
    def __init__(self, patterns, biases, low_memory=False):
        super().__init__()
        if patterns.ndim != 2:
            raise ValueError("Patterns must be a 2D tensor.")
        if biases.ndim != 1:
            raise ValueError("Biases must be a 1D tensor.")

        self.K, self.N = patterns.shape

        if biases.shape[0] != self.K:
            raise ValueError("Biases must have shape (K,).")

        self.patterns = patterns
        if low_memory:
            self.patterns_T = patterns.T
        else:
            self.patterns_T = patterns.T.contiguous()
        self.biases = biases

    def get_logits(self, x : torch.Tensor):
        return x @ self.patterns_T + self.biases

    def get_update(self, x : torch.Tensor, betas : torch.Tensor):
        p = torch.softmax(betas * self.get_logits(x), dim=-1)
        return p @ self.patterns

    def _setup(self, x0 : torch.Tensor, betas : torch.Tensor):
        if betas.ndim != 1:
            raise ValueError("Betas must be a 1D tensor.")

        B = betas.shape[0]

        if x0.ndim == 1:
            if x0.shape[0] != self.N:
                raise ValueError("For 1D x0, shape must be (N,).")
            x = x0[None, None, :].repeat(B, 1, 1)

        elif x0.ndim == 2:
            if x0.shape[-1] != self.N:
                raise ValueError("For 2D x0, shape must be (M, N).")
            x = x0[None, :, :].repeat(B, 1, 1)

        elif x0.ndim == 3:
            if x0.shape[-1] != self.N:
                raise ValueError("For 3D x0, shape must be (B, M, N).")
            if x0.shape[0] != B:
                raise ValueError("For 3D x0, first dimension must equal len(betas).")
            x = x0.clone()

        else:
            raise ValueError("x0 must have shape (N,), (M, N), or (B, M, N).")

        betas = betas[:, None, None]

        return x, betas

    def _converged(self, x_new : torch.Tensor, x : torch.Tensor, epsilon : float):
        return torch.all(torch.linalg.vector_norm(x_new - x, dim=-1) < epsilon)

    def discrete_num_iters(self, x0 : torch.Tensor, betas : torch.Tensor, num_iters : int, verbose : bool = False):
        x, betas = self._setup(x0, betas)

        if verbose:
            pbar = tqdm(range(num_iters), desc="Discrete Dynamics")
        else:
            pbar = range(num_iters)
        for _ in pbar:
            x = self.get_update(x, betas)

        return x

    def discrete(self, x0 : torch.Tensor, betas : torch.Tensor, epsilon : float, max_iters : int, verbose : bool = False):
        x, betas = self._setup(x0, betas)

        if verbose:
            pbar = tqdm(range(max_iters), desc="Discrete Dynamics")
        else:
            pbar = range(max_iters)

        for _ in pbar:
            x_new = self.get_update(x, betas)
            if self._converged(x_new, x, epsilon):
                x = x_new
                break
            x = x_new

        return x

    def continuous(self, x0 : torch.Tensor, betas : torch.Tensor, dt : float, epsilon : float, max_iters : int, verbose : bool = False):
        x, betas = self._setup(x0, betas)

        if verbose:
            pbar = tqdm(range(max_iters), desc="Continuous Dynamics")
        else:
            pbar = range(max_iters)

        for _ in pbar:
            x_new = x + dt * (self.get_update(x, betas) - x)
            if self._converged(x_new, x, epsilon):
                x = x_new
                break
            x = x_new

        return x

    def continuous_num_iters(self, x0 : torch.Tensor, betas : torch.Tensor, dt : float, num_iters : int, verbose : bool = False):
        x, betas = self._setup(x0, betas)

        if verbose:
            pbar = tqdm(range(num_iters), desc="Continuous Dynamics")
        else:
            pbar = range(num_iters)

        for _ in pbar:
            x = x + dt * (self.get_update(x, betas) - x)

        return x

    def stochastic_single_temp_num_iters(self, x0 : torch.Tensor, betas : torch.Tensor, temp : float, dt : float, num_iters : int, verbose : bool = False):
        x, betas = self._setup(x0, betas)
        noise_scale = sqrt(2 * dt * temp)
        if verbose:
            pbar = tqdm(range(num_iters), desc="Stochastic Dynamics")
        else:
            pbar = range(num_iters)

        for _ in pbar:
            x = x + dt * (self.get_update(x, betas) - x) + noise_scale * torch.randn_like(x)

        return x

    def stochastic_multiple_temp_num_iters(self, x0 : torch.Tensor, betas : torch.Tensor, temps : torch.Tensor, dt : float, num_iters : int, verbose : bool = False):
        x, betas = self._setup(x0, betas)
        if temps.ndim != 1:
            raise ValueError("Temps must be a 1D tensor.")
        if temps.shape[0] != betas.shape[0]:
            raise ValueError("Temps must have the same length as betas.")
        temps = temps[:, None, None]
        noise_scale = torch.sqrt(2 * dt * temps)
        if verbose:
            pbar = tqdm(range(num_iters), desc="Stochastic Dynamics")
        else:
            pbar = range(num_iters)

        for _ in pbar:
            x = x + dt * (self.get_update(x, betas) - x) + noise_scale * torch.randn_like(x)

        return x


    
class DualDynamics:
    def __init__(self, patterns : torch.Tensor, biases : torch.Tensor):
        super().__init__()
        if patterns.ndim != 2:
            raise ValueError("Patterns must be a 2D tensor.")
        if biases.ndim != 1:
            raise ValueError("Biases must be a 1D tensor.")

        self.K, self.N = patterns.shape

        if biases.shape[0] != self.K:
            raise ValueError("Biases must have shape (K,).")

        self.patterns = patterns
        self.biases = biases

        if self.K > 2 * self.N:
            self.gram = None
        else:
            self.gram = patterns @ patterns.T

    def get_dual_logits(self, p : torch.Tensor):
        if self.gram is None:
            logits = (p @ self.patterns) @ self.patterns.T
        else:
            logits = p @ self.gram

        return logits + self.biases

    def _setup(self, p0 : torch.Tensor, betas : torch.Tensor):
        if betas.ndim != 1:
            raise ValueError("Betas must be a 1D tensor.")

        B = betas.shape[0]

        if p0.ndim == 1:
            if p0.shape[0] != self.K:
                raise ValueError("For 1D p0, shape must be (K,).")
            p = p0[None, None, :].repeat(B, 1, 1)

        elif p0.ndim == 2:
            if p0.shape[-1] != self.K:
                raise ValueError("For 2D p0, shape must be (M, K).")
            p = p0[None, :, :].repeat(B, 1, 1)

        elif p0.ndim == 3:
            if p0.shape[-1] != self.K:
                raise ValueError("For 3D p0, shape must be (B, M, K).")
            if p0.shape[0] != B:
                raise ValueError("For 3D p0, first dimension must equal len(betas).")
            p = p0.clone()

        else:
            raise ValueError("p0 must have shape (K,), (M, K), or (B, M, K).")

        betas = betas[:, None, None]

        return p, betas

    def _converged(self, p_new : torch.Tensor, p : torch.Tensor, epsilon : float):
        return torch.all(torch.linalg.vector_norm(p_new - p, dim=-1) < epsilon)

    def discrete_num_iters(self, p0 : torch.Tensor, betas : torch.Tensor, num_iters : int, verbose : bool = False):
        p, betas = self._setup(p0, betas)

        if verbose:
            pbar = tqdm(range(num_iters), desc="Discrete Dynamics")
        else:
            pbar = range(num_iters)

        for _ in pbar:
            p = torch.softmax(betas * self.get_dual_logits(p), dim=-1)

        return p

    def discrete(self, p0 : torch.Tensor, betas : torch.Tensor, epsilon : float, max_iters : int, verbose : bool = False):
        p, betas = self._setup(p0, betas)

        if verbose:
            pbar = tqdm(range(max_iters), desc="Discrete Dynamics")
        else:
            pbar = range(max_iters)

        for _ in pbar:
            p_new = torch.softmax(betas * self.get_dual_logits(p), dim=-1)
            if self._converged(p_new, p, epsilon):
                p = p_new
                break
            p = p_new

        return p

    def continuous(self, p0 : torch.Tensor, betas : torch.Tensor, dt : float, epsilon : float, max_iters : int, verbose : bool = False):
        p, betas = self._setup(p0, betas)

        if verbose:
            pbar = tqdm(range(max_iters), desc="Continuous Dynamics")
        else:
            pbar = range(max_iters)

        for _ in pbar:
            p_new = p + dt * (torch.softmax(betas * self.get_dual_logits(p), dim=-1) - p)
            if self._converged(p_new, p, epsilon):
                p = p_new
                break
            p = p_new

        return p

    def continuous_num_iters(self, p0 : torch.Tensor, betas : torch.Tensor, dt : float, num_iters : int, verbose : bool = False):
        p, betas = self._setup(p0, betas)

        if verbose:
            pbar = tqdm(range(num_iters), desc="Continuous Dynamics")
        else:
            pbar = range(num_iters)

        for _ in pbar:
            p = p + dt * (torch.softmax(betas * self.get_dual_logits(p), dim=-1) - p)

        return p

    def natural_gradient_num_iters(self, p0 : torch.Tensor, betas : torch.Tensor, dt : float, num_iters : int, verbose : bool = False):

        p, betas = self._setup(p0, betas)
    
        if verbose:
            pbar = tqdm(range(num_iters), desc="Natural Gradient Dynamics")
        else:
            pbar = range(num_iters)

        for _ in pbar:
            term = self.get_dual_logits(p)
            term = term - (torch.log(p) + 1.0) / betas
            p = p * torch.exp(dt * term)
            p = p / p.sum(dim=-1, keepdim=True)
    
        return p

    def natural_gradient_entmax(self, alpha: float, p0: torch.Tensor, betas: torch.Tensor,
                            dt_base: float, total_time: float, max_iters: int, verbose: bool = False):
        if not (1.0 < alpha <= 2.0):
            raise ValueError("alpha must satisfy 1 < alpha <= 2")
    
        p, betas = self._setup(p0, betas)
        t = torch.zeros_like(p[..., :1])
        
        if verbose:
            pbar = tqdm(range(max_iters), desc="Natural Gradient Entmax Dynamics")
        else:
            pbar = range(max_iters)

        for _ in pbar:
            s = p.pow(2.0 - alpha)
    
            term = self.get_dual_logits(p)
            term = term - p.pow(alpha - 1.0) / (betas * (alpha - 1.0))
    
            mean_term = (s * term).sum(dim=-1, keepdim=True) / s.sum(dim=-1, keepdim=True)
            update = s * (term - mean_term)
    
            negative = update < 0
            dt_max = torch.where(negative, -p / update, torch.inf).amin(dim=-1, keepdim=True)
    
            dt_eff = torch.minimum(torch.full_like(dt_max, dt_base), 0.9 * dt_max)
            dt_eff = torch.minimum(dt_eff, total_time - t)
            dt_eff = dt_eff.clamp_min(0.0)
    
            p = p + dt_eff * update
            t = t + dt_eff

            if torch.all(t >= total_time):
                break
    
        return p




    