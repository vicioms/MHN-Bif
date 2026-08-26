import torch
import torch.nn as nn
from einops import rearrange

class DenoiserBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        beta: float,
        dt: float,
        num_steps: int,
        euclidean: bool = True,
    ):
        super().__init__()

        if beta <= 0:
            raise ValueError("beta must be > 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.euclidean = euclidean
        self.num_steps = num_steps

        self.register_buffer(
            "beta",
            torch.tensor(beta, dtype=torch.float32),
        )

        self.register_buffer(
            "dt",
            torch.tensor(dt, dtype=torch.float32),
        )

        self.patterns = nn.Parameter(
            torch.randn(input_dim, hidden_dim) / input_dim**0.5
        )

        self.biases = nn.Parameter(
            torch.zeros(hidden_dim)
        )

    def denoise_step(self, x):
        logits = self.beta * (x @ self.patterns)

        if self.euclidean:
            logits -= (
                0.5
                * self.beta
                * torch.sum(self.patterns**2, dim=0)
            )

        # beta-independent bias
        logits += self.biases

        weights = torch.softmax(logits, dim=-1)
        retrieved = weights @ self.patterns.T

        return x + self.dt * (retrieved - x)

    def forward(self, x):
        for _ in range(self.num_steps):
            x = self.denoise_step(x)

        return x
