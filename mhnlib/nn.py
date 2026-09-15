import torch
import torch.nn as nn
import torch.nn.functional as F



class SpatialMHN2d(nn.Module):
    """
    Spatial Modern Hopfield / neuron-field layer.

    Input:
        x: (B, H, W, C)

    Dynamics:
        dx_c/dt =
            sum_mu w_mu(x) * xi_{mu,c}
            - x_c
            + kappa_c * Laplacian(x_c)

    with:
        w_mu(x) = softmax_mu[
            beta * (xi_mu . x + bias_mu)
        ]

    Parameters:
        patterns:  (M, C)
        biases:    (M,)
        kappa:     (C,)
    """

    def __init__(
        self,
        memory_dim,
        memory_size,
        beta=1.0,
        kappa=0.1,
        dt=0.1,
        dx=1.0,
        learn_beta=False,
        learn_kappa=True,
        learn_dt=True,
        boundary="circular",
    ):
        super().__init__()

        self.memory_dim = memory_dim
        self.memory_size = memory_size
        self.dx = dx
        self.boundary = boundary

        # --------------------------------------------------
        # Memories
        #
        # xi_{mu,c} ~ N(0, 1/C)
        # => ||xi_mu|| ~ O(1)
        # --------------------------------------------------

        self.patterns = nn.Parameter(
            torch.randn(memory_size, memory_dim)
            * memory_dim**(-0.5)
        )

        self.biases = nn.Parameter(
            torch.zeros(memory_size)
        )

        # --------------------------------------------------
        # beta > 0
        # --------------------------------------------------

        log_beta = torch.tensor(
            float(beta)
        ).log()

        if learn_beta:
            self.log_beta = nn.Parameter(log_beta)
        else:
            self.register_buffer(
                "log_beta",
                log_beta
            )

        # --------------------------------------------------
        # channel-dependent kappa_c > 0
        # shape: (C,)
        # --------------------------------------------------

        log_kappa = torch.full(
            (memory_dim,),
            float(kappa)
        ).log()

        if learn_kappa:
            self.log_kappa = nn.Parameter(log_kappa)
        else:
            self.register_buffer(
                "log_kappa",
                log_kappa
            )

        # --------------------------------------------------
        # dt > 0
        # --------------------------------------------------

        log_dt = torch.tensor(
            float(dt)
        ).log()

        if learn_dt:
            self.log_dt = nn.Parameter(log_dt)
        else:
            self.register_buffer(
                "log_dt",
                log_dt
            )

        # --------------------------------------------------
        # 2D Laplacian
        # --------------------------------------------------

        laplacian = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0],
        ])

        self.register_buffer(
            "laplacian_kernel",
            laplacian[None, None]
        )

    # ======================================================
    # Positive parameters
    # ======================================================

    @property
    def beta(self):
        return self.log_beta.exp()

    @property
    def kappa(self):
        return self.log_kappa.exp()

    @property
    def dt(self):
        return self.log_dt.exp()

    # ======================================================
    # Spatial operator
    # ======================================================

    def laplacian(self, x):
        """
        x:
            (B, H, W, C)

        returns:
            (B, H, W, C)
        """

        # B,H,W,C -> B,C,H,W
        y = x.permute(0, 3, 1, 2)

        C = y.shape[1]

        kernel = self.laplacian_kernel.expand(
            C, 1, 3, 3
        )

        y = F.pad(
            y,
            (1, 1, 1, 1),
            mode=self.boundary
        )

        y = F.conv2d(
            y,
            kernel,
            groups=C
        )

        y = y / (self.dx ** 2)

        # B,C,H,W -> B,H,W,C
        return y.permute(0, 2, 3, 1)

    # ======================================================
    # Hopfield part
    # ======================================================

    def weights(self, x):
        """
        Latent phase field.

        x:
            (B,H,W,C)

        returns:
            w: (B,H,W,M)
        """

        logits = (
            x @ self.patterns.T
            + self.biases
        )

        return torch.softmax(
            self.beta * logits,
            dim=-1
        )

    def memory_field(self, x):
        """
        m_c(x) = sum_mu w_mu xi_{mu,c}

        returns:
            (B,H,W,C)
        """

        w = self.weights(x)

        return w @ self.patterns

    # ======================================================
    # Neuron-field dynamics
    # ======================================================

    def vector_field(self, x):
        """
        dx/dt =
            m(x) - x + kappa_c Laplacian(x)
        """

        memory = self.memory_field(x)
        lapl = self.laplacian(x)

        # (C,) -> (1,1,1,C)
        kappa = self.kappa.view(
            1, 1, 1, -1
        )

        return (
            memory
            - x
            + kappa * lapl
        )

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        x,
        num_steps=1,
        return_weights=False,
    ):
        """
        Euler integration.

        x:
            (B,H,W,C)
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected (B,H,W,C), got {x.shape}"
            )

        if x.shape[-1] != self.memory_dim:
            raise ValueError(
                f"Expected C={self.memory_dim}, "
                f"got C={x.shape[-1]}"
            )

        for _ in range(num_steps):
            x = (
                x
                + self.dt * self.vector_field(x)
            )

        if return_weights:
            return x, self.weights(x)

        return x