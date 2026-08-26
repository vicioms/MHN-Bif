import torch
import torch.nn as nn
import torch.nn.functional as F

class HopfieldLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_memories,
        beta,
    ):
        super().__init__()

        self.W_Q = nn.Linear(input_dim, hidden_dim, bias=False)

        self.keys = nn.Parameter(
            torch.randn(num_memories, hidden_dim)
            / hidden_dim**0.5
        )

        self.values = nn.Parameter(
            torch.randn(num_memories, output_dim)
            / output_dim**0.5
        )

        self.register_buffer("beta", torch.tensor(float(beta)))

        
    def forward(self, x):
        q = self.W_Q(x)

        logits = self.beta * (q @ self.keys.T)
        weights = torch.softmax(logits, dim=-1)

        return weights @ self.values

class MHNLayer(nn.Module):
    @staticmethod
    def inv_sigmoid(x, eps=1e-6):
        x = torch.clamp(x, eps, 1.0 - eps)
        return torch.logit(x)
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_memories,
        beta,
        dt=1.0,
        entropic_bias=False,
        learn_dt=False,
        euclidean=False,
    ):
        super().__init__()

        self.W_Q = nn.Linear(
            input_dim,
            hidden_dim,
            bias=False
        )

        if entropic_bias:
            self.entropic_bias = nn.Parameter(
                torch.zeros(num_memories)
            )
        else:
            self.register_buffer(
                "entropic_bias",
                torch.zeros(num_memories)
            )

        # dt = sigmoid(alpha), therefore dt is constrained to (0, 1)
        alpha = self.inv_sigmoid(
            torch.tensor(float(dt))
        )

        if learn_dt:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

        # Learned keys
        self.keys = nn.Parameter(
            torch.randn(num_memories, hidden_dim)
            / hidden_dim**0.5
        )

        # Learned values
        self.values = nn.Parameter(
            torch.randn(num_memories, input_dim)
            / input_dim**0.5
        )

        self.register_buffer(
            "beta",
            torch.tensor(float(beta))
        )

        self.euclidean = euclidean

    @property
    @torch.no_grad()
    def stab_matrix(self):
        # V_{\mu i} softmax_\mu(beta K_\mu a Q_ai x_i) - x_i
        # grad is:
        # \beta \sum_\mu V_{\mu i}p_\mu(x) (K_\mu a - \sum_\nu p_\nu(x) K_\nu a) Q_aj - \delta_{ij}
        # for uniform weight (at beta = 0)
        # - \delta_{ij} + (\beta / M) \sum_\mu V_{\mu i} (K_{\mu a} - K_a) Q_{aj}
        mean_keys = self.keys.mean(dim=0, keepdim=True)
        key_diff = self.keys - mean_keys
        return self.values.T @ key_diff @ self.W_Q.weight

    def forward(self, x):
        q = self.W_Q(x)

        logits = q @ self.keys.T
        if self.euclidean:
            logits = -0.5 * (self.keys.pow(2).sum(dim=1, keepdim=False))
        logits = self.beta * logits + self.entropic_bias
        weights = torch.softmax(logits, dim=-1)

        retrieved = weights @ self.values

        dt = torch.sigmoid(self.alpha)

        return x + dt * (retrieved - x)



#class MixtureLayer(nn.Module):
#    def __init__(
#        self,
#        input_dim,
#        hidden_dim,
#        output_dim,
#        n_clusters,
#        fit_cluster_biases=True,
#        fit_lin_in=True,
#        fit_lin_out=True,
#        is_residual=False,
#        eps=1e-8,
#    ):
#        super().__init__()
#
#        self.input_dim = input_dim
#        self.hidden_dim = hidden_dim
#        self.output_dim = output_dim
#        self.n_clusters = n_clusters
#        self.fit_cluster_biases = fit_cluster_biases
#        self.fit_lin_in = fit_lin_in
#        self.fit_lin_out = fit_lin_out
#        self.is_residual = is_residual
#        self.eps = eps
#
#        if not fit_lin_in and hidden_dim != input_dim:
#            raise ValueError("If fit_lin_in is False, hidden_dim must equal input_dim")
#
#        if not fit_lin_out and output_dim != hidden_dim:
#            raise ValueError("If fit_lin_out is False, output_dim must equal hidden_dim")
#
#        if is_residual and input_dim != output_dim:
#            raise ValueError("For residual connections, input_dim must equal output_dim")
#
#        self.hidden_clusters = nn.Parameter(torch.randn(n_clusters, hidden_dim) / hidden_dim**0.5)
#
#        if fit_cluster_biases:
#            self.hidden_biases = nn.Parameter(torch.zeros(n_clusters))
#        else:
#            self.register_buffer("hidden_biases", torch.zeros(n_clusters))
#
#        self.hidden_mean = nn.Parameter(torch.zeros(hidden_dim))
#        self.hidden_log_var = nn.Parameter(torch.zeros(hidden_dim))
#
#        self.lin_in = nn.Linear(input_dim, hidden_dim) if fit_lin_in else nn.Identity()
#        self.lin_out = nn.Linear(hidden_dim, output_dim) if fit_lin_out else nn.Identity()
#
#        if is_residual and fit_lin_out:
#            self.log_modulation = nn.Parameter(torch.tensor(0.0))
#        else:
#            self.register_buffer("log_modulation", torch.tensor(0.0))
#
#    def forward(self, x):
#        if x.dim() == 1:
#            x = x.unsqueeze(0)
#
#        batch_size, dim = x.shape
#
#        if dim != self.input_dim:
#            raise ValueError(f"Expected input dimension {self.input_dim}, got {dim}")
#
#        h = self.lin_in(x)
#
#        logits = torch.einsum("bh,kh->bk", h, self.hidden_clusters)
#        logits = logits + self.hidden_biases.view(1, -1)
#
#        weights = torch.softmax(logits, dim=1)
#
#        var = F.softplus(self.hidden_log_var).clamp_min(self.eps)
#
#        linear_force = -(h - self.hidden_mean.view(1, -1)) / var.view(1, -1)
#
#        cluster_force = torch.einsum("bk,kh->bh", weights, self.hidden_clusters)
#
#        force = cluster_force + linear_force
#
#        out_force = self.log_modulation.exp() * self.lin_out(force)
#
#        if self.is_residual:
#            return x + out_force
#        else:
#            return out_force
#
#
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#
#
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#
#
#class MemoryLayer(nn.Module):
#    def __init__(
#        self,
#        input_dim: int,
#        num_memories: int,
#        beta: float,
#        is_causal: bool = True,
#        modulation: float = 0.1,
#        decay: float = 0.95,
#        normalize: bool = True,
#    ):
#        super().__init__()
#
#        self.input_dim = input_dim
#        self.num_memories = num_memories
#        self.beta = beta
#        self.is_causal = is_causal
#        self.decay = decay
#        self.normalize = normalize
#
#        # memory keys: m_k
#        self.memory_keys = nn.Linear(input_dim, num_memories, bias=False)
#
#        # prompt-token -> memory bias contribution
#        self.context_proj = nn.Linear(input_dim, num_memories, bias=False)
#
#        # memory values: v_k
#        self.memory_values = nn.Parameter(
#            torch.randn(num_memories, input_dim) / input_dim**0.5
#        )
#
#        self.log_modulation = nn.Parameter(torch.tensor(float(modulation)).log())
#
#    def causal_bias(self, x):
#        """
#        x: (B, T, input_dim)
#
#        returns:
#            bias: (B, T, num_memories)
#
#        b_t = decay * b_{t-1} + P x_t
#        """
#
#        raw = self.context_proj(x)  # (B, T, K)
#
#        if not self.is_causal:
#            # Non-causal global prompt field shared by all positions.
#            global_bias = raw.sum(dim=1, keepdim=True)  # (B, 1, K)
#            return global_bias.expand_as(raw)
#
#        B, T, K = raw.shape
#        b = torch.zeros(B, K, device=x.device, dtype=x.dtype)
#        out = []
#
#        for t in range(T):
#            b = self.decay * b + raw[:, t]
#            out.append(b)
#
#        return torch.stack(out, dim=1)  # (B, T, K)
#
#    def forward(self, x):
#        """
#        x: (B, T, input_dim)
#
#        returns:
#            out:     (B, T, input_dim)
#            weights: (B, T, num_memories)
#            bias:    (B, T, num_memories)
#        """
#
#        if x.ndim != 3:
#            raise ValueError(f"x must be (B, T, input_dim), got {x.shape}")
#
#        # similarity of current token/state to learned memory keys
#        if self.normalize:
#            x_hat = F.normalize(x, dim=-1)
#            key_hat = F.normalize(self.memory_keys.weight, dim=-1)  # (K, input_dim)
#            memory_scores = torch.einsum("btd,kd->btk", x_hat, key_hat)
#        else:
#            memory_scores = self.memory_keys(x)  # (B, T, K)
#
#        # recurrent prompt-induced bias over memories
#        bias = self.causal_bias(x)  # (B, T, K)
#
#        logits = self.beta * memory_scores + bias
#
#        weights = torch.softmax(logits, dim=-1)  # (B, T, K)
#
#        retrieved = torch.einsum("btk,kd->btd", weights, self.memory_values)
#
#        eps = self.log_modulation.exp()
#
#        out = x + eps * (retrieved - x)
#
#        return out, weights, bias
#
#        
#
#class CausalHopfieldLayer(nn.Module):
#    def __init__(
#        self,
#        x_dim,
#        seq_length,
#        h_dim,
#        att_dim,
#        beta,
#        tau_h=1.0,
#        tau_m=1.0,
#        bias=False,
#    ):
#        super().__init__()
#
#        self.x_dim = x_dim
#        self.seq_length = seq_length
#        self.h_dim = h_dim
#        self.att_dim = att_dim
#
#        self.beta = beta
#        self.tau_h = tau_h
#        self.tau_m = tau_m
#
#        self.lin_q = nn.Linear(h_dim, att_dim, bias=bias)
#        self.lin_k = nn.Linear(x_dim, att_dim, bias=bias)
#        self.lin_v = nn.Linear(x_dim, h_dim, bias=bias)
#        self.lin_out = nn.Linear(h_dim, x_dim, bias=bias)
#
#        self.x_grid = None
#        self.t_grid = None
#        self.valid_mask = None
#
#    def set_context(self, x_grid, t_grid, valid_mask=None):
#        device = next(self.parameters()).device
#        dtype = next(self.parameters()).dtype
#
#        self.x_grid = x_grid.to(device=device, dtype=dtype)
#        self.t_grid = t_grid.to(device=device, dtype=dtype)
#
#        if valid_mask is None:
#            self.valid_mask = None
#        else:
#            self.valid_mask = valid_mask.to(device=device, dtype=torch.bool)
#
#    def forward(self, t, h):
#        """
#        t: scalar tensor
#        h: (B, h_dim)
#
#        returns:
#            dh: (B, h_dim)
#        """
#
#        if self.x_grid is None or self.t_grid is None:
#            raise RuntimeError("Call set_context(x_grid, t_grid, valid_mask) before forward.")
#
#        x = self.x_grid       # (B, T, x_dim)
#        ts = self.t_grid      # (T,)
#
#        B, T, _ = x.shape
#
#        time_mask = ts <= t   # (T,)
#
#        if self.valid_mask is not None:
#            mask = time_mask[None, :] & self.valid_mask   # (B, T)
#        else:
#            mask = time_mask[None, :].expand(B, T)        # (B, T)
#
#        q = self.lin_q(h)     # (B, att_dim)
#        k = self.lin_k(x)     # (B, T, att_dim)
#        v = self.lin_v(x)     # (B, T, h_dim)
#
#        scores = self.beta * torch.einsum("bd,btd->bt", q, k)
#
#        # Recency bias. Equivalent to -(t - s) / tau_m up to a softmax constant.
#        scores = scores + ts[None, :] / self.tau_m
#
#        scores = scores.masked_fill(~mask, -torch.inf)
#
#        alpha = torch.softmax(scores, dim=1)          # (B, T)
#
#        y = torch.einsum("bt,bth->bh", alpha, v)      # (B, h_dim)
#
#        dh = -h / self.tau_h + y
#
#        return dh
#
#    def readout(self, h):
#        return self.lin_out(h)
#