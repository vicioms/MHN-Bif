import torch

class EntmaxBisectFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, alpha, dim, n_iter):

        alpha = float(alpha)
        dim = int(dim)
        n_iter = int(n_iter)

        if not (1.0 < alpha <= 2.0):
            raise ValueError("alpha must satisfy 1 < alpha <= 2")

        # Do threshold calculation in fp32 for fp16/bfloat16 inputs
        work_dtype = (
            torch.float32
            if z.dtype in (torch.float16, torch.bfloat16)
            else z.dtype
        )

        x = z.to(work_dtype)

        # Entmax is invariant under z -> z + const
        x = x - x.amax(dim=dim, keepdim=True)

        power = 1.0 / (alpha - 1.0)

        # Since max(x)=0:
        #
        # tau_hi = 0       -> total mass = 0
        # tau_lo = -1/(a-1) -> largest component alone has mass >= 1
        #
        tau_hi = torch.zeros_like(
            x.amax(dim=dim, keepdim=True)
        )

        tau_lo = torch.full_like(
            tau_hi,
            -1.0 / (alpha - 1.0)
        )

        for _ in range(n_iter):

            tau = 0.5 * (tau_lo + tau_hi)

            p = (
                (alpha - 1.0) * (x - tau)
            ).clamp_min(0).pow(power)

            mass = p.sum(dim=dim, keepdim=True)

            # mass decreases monotonically with tau
            too_much = mass > 1.0

            tau_lo = torch.where(
                too_much,
                tau,
                tau_lo
            )

            tau_hi = torch.where(
                too_much,
                tau_hi,
                tau
            )

        tau = 0.5 * (tau_lo + tau_hi)

        p = (
            (alpha - 1.0) * (x - tau)
        ).clamp_min(0).pow(power)

        p = p.to(z.dtype)

        ctx.save_for_backward(p)
        ctx.alpha = alpha
        ctx.dim = dim

        return p

    @staticmethod
    def backward(ctx, grad_output):

        (p,) = ctx.saved_tensors

        alpha = ctx.alpha
        dim = ctx.dim

        work_dtype = (
            torch.float32
            if p.dtype in (torch.float16, torch.bfloat16)
            else p.dtype
        )

        p = p.to(work_dtype)
        g = grad_output.to(work_dtype)

        # s_mu = p_mu^(2-alpha)
        # explicitly zero outside support
        s = torch.where(
            p > 0,
            p.pow(2.0 - alpha),
            torch.zeros_like(p),
        )

        Z = s.sum(dim=dim, keepdim=True)

        mean_g = (
            (s * g).sum(dim=dim, keepdim=True)
            / Z.clamp_min(torch.finfo(work_dtype).tiny)
        )

        grad_z = s * (g - mean_g)

        return grad_z.to(grad_output.dtype), None, None, None


def entmax(z, alpha=1.5, dim=-1, n_iter=32):
    return EntmaxBisectFn.apply(
        z,
        alpha,
        dim,
        n_iter,
    )


def sparsemax(x, dim=-1):
    """alpha = 2."""
    # Translation invariance improves numerical stability
    x = x - x.amax(dim=dim, keepdim=True)

    xs, _ = torch.sort(x, dim=dim, descending=True)

    K = x.shape[dim]
    rho = torch.arange(
        1, K + 1,
        device=x.device,
        dtype=x.dtype,
    )

    shape = [1] * x.ndim
    shape[dim] = K
    rho = rho.view(shape)

    cumsum = xs.cumsum(dim=dim)

    # Candidate thresholds
    tau = (cumsum - 1.0) / rho

    # Active-set condition
    support = xs > tau
    k = support.sum(dim=dim, keepdim=True)

    # Select tau corresponding to the actual support size
    tau_star = torch.gather(
        tau,
        dim,
        k - 1,
    )

    return torch.clamp(x - tau_star, min=0.0)


def entmax15(x, dim=-1):
    """alpha = 3/2."""

    # Since p_i = [ (z_i - tau)/2 ]_+^2,
    # work with x = z/2 so p_i = [x_i - tau']_+^2.
    x = x / 2.0

    # Translation invariance
    x = x - x.amax(dim=dim, keepdim=True)

    xs, _ = torch.sort(x, dim=dim, descending=True)

    K = x.shape[dim]
    rho = torch.arange(
        1, K + 1,
        device=x.device,
        dtype=x.dtype,
    )

    shape = [1] * x.ndim
    shape[dim] = K
    rho = rho.view(shape)

    mean = xs.cumsum(dim=dim) / rho
    mean_sq = xs.square().cumsum(dim=dim) / rho

    # sum_{i<=k} (x_i - mean_k)^2
    ss = rho * (mean_sq - mean.square())

    # From sum_i (x_i - tau)^2 = 1
    delta = (1.0 - ss) / rho
    delta = torch.clamp(delta, min=0.0)

    tau = mean - torch.sqrt(delta)

    # Candidate active sets must satisfy x_(k) > tau_k
    support = xs > tau
    k = support.sum(dim=dim, keepdim=True)

    tau_star = torch.gather(
        tau,
        dim,
        k - 1,
    )

    return torch.clamp(x - tau_star, min=0.0).square()