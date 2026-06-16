import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_clusters,
        fit_cluster_biases=True,
        fit_lin_in=True,
        fit_lin_out=True,
        is_residual=False,
        eps=1e-8,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_clusters = n_clusters
        self.fit_cluster_biases = fit_cluster_biases
        self.fit_lin_in = fit_lin_in
        self.fit_lin_out = fit_lin_out
        self.is_residual = is_residual
        self.eps = eps

        if not fit_lin_in and hidden_dim != input_dim:
            raise ValueError("If fit_lin_in is False, hidden_dim must equal input_dim")

        if not fit_lin_out and output_dim != hidden_dim:
            raise ValueError("If fit_lin_out is False, output_dim must equal hidden_dim")

        if is_residual and input_dim != output_dim:
            raise ValueError("For residual connections, input_dim must equal output_dim")

        self.hidden_clusters = nn.Parameter(torch.randn(n_clusters, hidden_dim) / hidden_dim**0.5)

        if fit_cluster_biases:
            self.hidden_biases = nn.Parameter(torch.zeros(n_clusters))
        else:
            self.register_buffer("hidden_biases", torch.zeros(n_clusters))

        self.hidden_mean = nn.Parameter(torch.zeros(hidden_dim))
        self.hidden_log_var = nn.Parameter(torch.zeros(hidden_dim))

        self.lin_in = nn.Linear(input_dim, hidden_dim) if fit_lin_in else nn.Identity()
        self.lin_out = nn.Linear(hidden_dim, output_dim) if fit_lin_out else nn.Identity()

        if is_residual and fit_lin_out:
            self.log_modulation = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("log_modulation", torch.tensor(0.0))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size, dim = x.shape

        if dim != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {dim}")

        h = self.lin_in(x)

        logits = torch.einsum("bh,kh->bk", h, self.hidden_clusters)
        logits = logits + self.hidden_biases.view(1, -1)

        weights = torch.softmax(logits, dim=1)

        var = F.softplus(self.hidden_log_var).clamp_min(self.eps)

        linear_force = -(h - self.hidden_mean.view(1, -1)) / var.view(1, -1)

        cluster_force = torch.einsum("bk,kh->bh", weights, self.hidden_clusters)

        force = cluster_force + linear_force

        out_force = self.log_modulation.exp() * self.lin_out(force)

        if self.is_residual:
            return x + out_force
        else:
            return out_force