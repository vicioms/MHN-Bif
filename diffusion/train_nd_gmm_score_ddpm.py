import os
import json
import math
import argparse
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    num_train_samples: int = 50_000

    dim: int = 16
    K: int = 8
    centroid_scale: float = 3.0

    min_component_std: float = 0.05
    max_component_std: float = 0.25

    epochs: int = 200
    batch_size: int = 512
    lr: float = 2e-4

    T: int = 200
    hidden_dim: int = 256
    time_dim: int = 64

    save_dir: str = "runs/nd_gmm_score_ddpm"
    save_every_epochs: int = 20
    plot_every_epochs: int = 20

    num_workers: int = 0
    seed: int = 123


# ============================================================
# N-dimensional GMM dataset
# ============================================================

def make_random_gmm_params(
    K,
    dim,
    centroid_scale,
    min_component_std,
    max_component_std,
    seed,
    device,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Random centroids on approximately a sphere of radius centroid_scale * sqrt(dim).
    centers = torch.randn(K, dim, generator=generator, device=device)
    centers = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    centers = centers * centroid_scale * math.sqrt(dim)

    # Component-dependent isotropic stds.
    component_stds = min_component_std + (
        max_component_std - min_component_std
    ) * torch.rand(K, generator=generator, device=device)

    # Uniform mixture weights.
    weights = torch.ones(K, device=device) / K

    return centers, component_stds, weights


def sample_gmm_dataset(
    num_samples,
    centers,
    component_stds,
    weights,
    seed,
    device,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    K, dim = centers.shape

    labels = torch.multinomial(
        weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    ).to(device)

    noise = torch.randn(
        num_samples,
        dim,
        generator=generator,
        device=device,
    )

    x = centers[labels] + component_stds[labels].view(-1, 1) * noise

    return x, labels


class TensorDatasetND(Dataset):
    def __init__(self, x, labels=None):
        self.x = x
        self.labels = labels

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return self.x[idx]
        return self.x[idx], self.labels[idx]


# ============================================================
# Diffusion schedule
# ============================================================

class DiffusionSchedule:
    def __init__(self, T, device):
        self.T = T
        self.device = device

        self.betas = torch.linspace(1e-4, 2e-2, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        a = self.sqrt_alpha_bars[t].view(-1, 1)
        s = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)

        x_t = a * x0 + s * noise

        return x_t, noise


# ============================================================
# Time embedding
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2

        freqs = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=t.device)
            / max(half - 1, 1)
        )

        args = t.float().view(-1, 1) * freqs.view(1, -1)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


# ============================================================
# Score network
# ============================================================

class ScoreNet(nn.Module):
    def __init__(self, x_dim, time_dim=64, hidden_dim=256):
        super().__init__()

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(x_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x, t):
        temb = self.time_emb(t)
        h = torch.cat([x, temb], dim=-1)
        return self.net(h)


# ============================================================
# Exact noised N-dimensional GMM score
# ============================================================

@torch.no_grad()
def exact_gmm_score_t(
    x,
    t,
    centers,
    component_stds,
    weights,
    schedule,
):
    """
    Exact score of p_t(x) for an N-dimensional GMM with isotropic components.

    Original component:

        x0 | k ~ N(mu_k, sigma_k^2 I)

    Noised component:

        x_t | k ~ N(
            sqrt(alpha_bar_t) mu_k,
            [alpha_bar_t sigma_k^2 + 1 - alpha_bar_t] I
        )
    """
    alpha_bar_t = schedule.alpha_bars[t]

    means_t = torch.sqrt(alpha_bar_t) * centers

    vars_t = alpha_bar_t * component_stds.pow(2) + (1.0 - alpha_bar_t)

    diff = x[:, None, :] - means_t[None, :, :]

    sqdist = diff.pow(2).sum(dim=-1)

    dim = x.shape[-1]

    log_probs = (
        torch.log(weights.view(1, -1))
        - 0.5 * dim * torch.log(2.0 * torch.pi * vars_t.view(1, -1))
        - 0.5 * sqdist / vars_t.view(1, -1)
    )

    resp = torch.softmax(log_probs, dim=-1)

    component_scores = -diff / vars_t.view(1, -1, 1)

    score = (resp[:, :, None] * component_scores).sum(dim=1)

    return score


# ============================================================
# Sampling
# ============================================================

@torch.no_grad()
def p_sample(model, x_t, t, schedule):
    B = x_t.shape[0]
    device = x_t.device

    t_batch = torch.full(
        (B,),
        t,
        device=device,
        dtype=torch.long,
    )

    beta_t = schedule.betas[t]
    alpha_t = schedule.alphas[t]

    score_pred = model(x_t, t_batch)

    mean = (x_t + beta_t * score_pred) / torch.sqrt(alpha_t)

    if t == 0:
        return mean

    noise = torch.randn_like(x_t)
    sigma_t = torch.sqrt(beta_t)

    return mean + sigma_t * noise


@torch.no_grad()
def sample_ddpm(model, schedule, num_samples, dim, device):
    model.eval()

    x = torch.randn(num_samples, dim, device=device)

    for t in reversed(range(schedule.T)):
        x = p_sample(model, x, t, schedule)

    return x.detach().cpu()


# ============================================================
# Saving
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    losses,
    cfg,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "losses": losses,
        "config": asdict(cfg),
    }

    torch.save(payload, path)


def save_dataset(
    path,
    x_train,
    labels,
    centers,
    component_stds,
    weights,
    cfg,
):
    payload = {
        "x_train": x_train.cpu(),
        "labels": labels.cpu(),
        "centers": centers.cpu(),
        "component_stds": component_stds.cpu(),
        "weights": weights.cpu(),
        "config": asdict(cfg),
    }

    torch.save(payload, path)


# ============================================================
# Diagnostics
# ============================================================

@torch.no_grad()
def plot_2d_projection_samples(
    path,
    model,
    schedule,
    x_train,
    centers,
    cfg,
    epoch,
    device,
    num_samples=3000,
):
    """
    For dim > 2, plot the first two coordinates.
    This is crude but useful for quickly checking if samples have the right scale.
    """
    model.eval()

    idx = torch.randint(0, x_train.shape[0], (num_samples,))
    real = x_train[idx].cpu()

    fake = sample_ddpm(
        model=model,
        schedule=schedule,
        num_samples=num_samples,
        dim=cfg.dim,
        device=device,
    )

    plt.figure(figsize=(6, 6))

    plt.scatter(
        real[:, 0],
        real[:, 1],
        s=4,
        alpha=0.35,
        label="training data",
    )

    plt.scatter(
        fake[:, 0],
        fake[:, 1],
        s=4,
        alpha=0.35,
        label="score-DDPM samples",
    )

    plt.scatter(
        centers.cpu()[:, 0],
        centers.cpu()[:, 1],
        s=80,
        marker="x",
        label="GMM centers",
    )

    plt.axis("equal")
    plt.title(f"First two coordinates, epoch {epoch}")
    plt.legend()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def plot_pca_projection_samples(
    path,
    model,
    schedule,
    x_train,
    centers,
    cfg,
    epoch,
    device,
    num_samples=3000,
):
    """
    PCA projection using the training data covariance.
    This is better than first-two-coordinates when dim is large.
    """
    model.eval()

    idx = torch.randint(0, x_train.shape[0], (min(num_samples, x_train.shape[0]),))
    real = x_train[idx].cpu()

    fake = sample_ddpm(
        model=model,
        schedule=schedule,
        num_samples=real.shape[0],
        dim=cfg.dim,
        device=device,
    )

    x_mean = x_train.cpu().mean(dim=0, keepdim=True)
    X = x_train.cpu() - x_mean

    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    W = Vh[:2].T

    real_2d = (real - x_mean) @ W
    fake_2d = (fake - x_mean) @ W
    centers_2d = (centers.cpu() - x_mean) @ W

    plt.figure(figsize=(6, 6))

    plt.scatter(
        real_2d[:, 0],
        real_2d[:, 1],
        s=4,
        alpha=0.35,
        label="training data",
    )

    plt.scatter(
        fake_2d[:, 0],
        fake_2d[:, 1],
        s=4,
        alpha=0.35,
        label="score-DDPM samples",
    )

    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        s=80,
        marker="x",
        label="GMM centers",
    )

    plt.axis("equal")
    plt.title(f"PCA projection, epoch {epoch}")
    plt.legend()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def estimate_score_mse(
    model,
    schedule,
    centers,
    component_stds,
    weights,
    cfg,
    device,
    num_eval=4096,
):
    """
    Compare learned score to exact GMM score at random noisy points.
    """
    model.eval()

    x0, _ = sample_gmm_dataset(
        num_samples=num_eval,
        centers=centers,
        component_stds=component_stds,
        weights=weights,
        seed=cfg.seed + 999,
        device=device,
    )

    t = torch.randint(0, cfg.T, (num_eval,), device=device)
    x_t, noise = schedule.q_sample(x0, t)

    score_pred = model(x_t, t)

    # Exact score must be computed per time. For simplicity, loop over unique t.
    score_exact = torch.empty_like(score_pred)

    for tt in t.unique():
        mask = t == tt
        score_exact[mask] = exact_gmm_score_t(
            x=x_t[mask],
            t=int(tt.item()),
            centers=centers,
            component_stds=component_stds,
            weights=weights,
            schedule=schedule,
        )

    mse = (score_pred - score_exact).pow(2).sum(dim=-1).mean().item()

    return mse


def plot_loss(path, losses):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("optimization step")
    plt.ylabel("weighted score matching loss")
    plt.title("Training loss")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Training
# ============================================================

def train(cfg):
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)

    checkpoint_dir = os.path.join(cfg.save_dir, "checkpoints")
    plot_dir = os.path.join(cfg.save_dir, "plots")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Training N-dimensional GMM with dim={cfg.dim}, K={cfg.K}")

    centers, component_stds, weights = make_random_gmm_params(
        K=cfg.K,
        dim=cfg.dim,
        centroid_scale=cfg.centroid_scale,
        min_component_std=cfg.min_component_std,
        max_component_std=cfg.max_component_std,
        seed=cfg.seed,
        device=device,
    )

    x_train, labels = sample_gmm_dataset(
        num_samples=cfg.num_train_samples,
        centers=centers,
        component_stds=component_stds,
        weights=weights,
        seed=cfg.seed + 1,
        device=device,
    )

    save_dataset(
        path=os.path.join(cfg.save_dir, "training_dataset.pt"),
        x_train=x_train,
        labels=labels,
        centers=centers,
        component_stds=component_stds,
        weights=weights,
        cfg=cfg,
    )

    dataset = TensorDatasetND(
        x=x_train.detach().cpu(),
        labels=labels.detach().cpu(),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )

    schedule = DiffusionSchedule(
        T=cfg.T,
        device=device,
    )

    model = ScoreNet(
        x_dim=cfg.dim,
        time_dim=cfg.time_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    losses = []
    score_mses = []
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch in loader:
            x0, _ = batch
            x0 = x0.to(device)

            B = x0.shape[0]

            t = torch.randint(
                low=0,
                high=cfg.T,
                size=(B,),
                device=device,
            )

            x_t, noise = schedule.q_sample(x0, t)

            score_pred = model(x_t, t)

            sigma_t = schedule.sqrt_one_minus_alpha_bars[t].view(-1, 1)

            score_target = -noise / sigma_t

            # Weighted score matching:
            #
            #     sigma_t^2 || s_theta(x_t,t) + eps / sigma_t ||^2
            #
            # equivalent to epsilon prediction after eps_theta = -sigma_t s_theta.
            weight = sigma_t**2

            loss = (weight * (score_pred - score_target).pow(2)).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        mean_epoch_loss = epoch_loss / max(num_batches, 1)

        print(
            f"epoch {epoch:5d}/{cfg.epochs} | "
            f"step {global_step:8d} | "
            f"loss {mean_epoch_loss:.6f}"
        )

        should_save = (
            epoch % cfg.save_every_epochs == 0
            or epoch == 1
            or epoch == cfg.epochs
        )

        should_plot = (
            epoch % cfg.plot_every_epochs == 0
            or epoch == 1
            or epoch == cfg.epochs
        )

        if should_save:
            score_mse = estimate_score_mse(
                model=model,
                schedule=schedule,
                centers=centers,
                component_stds=component_stds,
                weights=weights,
                cfg=cfg,
                device=device,
            )

            score_mses.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "score_mse": score_mse,
                }
            )

            print(f"exact score MSE: {score_mse:.6f}")

            save_checkpoint(
                path=os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:05d}.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                losses=losses,
                cfg=cfg,
            )

            with open(os.path.join(cfg.save_dir, "score_mses.json"), "w") as f:
                json.dump(score_mses, f, indent=2)

        if should_plot:
            plot_loss(
                path=os.path.join(plot_dir, "loss.png"),
                losses=losses,
            )

            plot_2d_projection_samples(
                path=os.path.join(plot_dir, f"first2_epoch_{epoch:05d}.png"),
                model=model,
                schedule=schedule,
                x_train=x_train.detach().cpu(),
                centers=centers.detach().cpu(),
                cfg=cfg,
                epoch=epoch,
                device=device,
            )

            plot_pca_projection_samples(
                path=os.path.join(plot_dir, f"pca_epoch_{epoch:05d}.png"),
                model=model,
                schedule=schedule,
                x_train=x_train.detach().cpu(),
                centers=centers.detach().cpu(),
                cfg=cfg,
                epoch=epoch,
                device=device,
            )

    save_checkpoint(
        path=os.path.join(cfg.save_dir, "final_model.pt"),
        model=model,
        optimizer=optimizer,
        epoch=cfg.epochs,
        global_step=global_step,
        losses=losses,
        cfg=cfg,
    )

    print(f"Saved run to: {cfg.save_dir}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_train_samples", type=int, default=50_000)

    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--centroid_scale", type=float, default=3.0)

    parser.add_argument("--min_component_std", type=float, default=0.05)
    parser.add_argument("--max_component_std", type=float, default=0.25)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=64)

    parser.add_argument("--save_dir", type=str, default="runs/nd_gmm_score_ddpm")
    parser.add_argument("--save_every_epochs", type=int, default=20)
    parser.add_argument("--plot_every_epochs", type=int, default=20)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    cfg = Config(
        num_train_samples=args.num_train_samples,
        dim=args.dim,
        K=args.K,
        centroid_scale=args.centroid_scale,
        min_component_std=args.min_component_std,
        max_component_std=args.max_component_std,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        T=args.T,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        save_dir=args.save_dir,
        save_every_epochs=args.save_every_epochs,
        plot_every_epochs=args.plot_every_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)