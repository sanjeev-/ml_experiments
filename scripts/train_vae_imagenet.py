"""
Train a small convolutional VAE on ImageNet using Modal.

Usage:
    modal run scripts/train_vae_imagenet.py
    modal run scripts/train_vae_imagenet.py --epochs 10 --latent-dim 256
"""

import modal
import time

# ─── Modal setup ────────────────────────────────────────────────────────────

app = modal.App("vae-imagenet")

vol = modal.Volume.from_name("vae-imagenet-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "datasets>=2.16.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "rich>=13.0.0",
        "wandb>=0.16.0",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
    })
)


# ─── VAE Model ──────────────────────────────────────────────────────────────

def build_model_code():
    """Returns model classes — called inside the Modal function."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )

        def forward(self, x):
            return x + self.net(x)

    class Encoder(nn.Module):
        def __init__(self, in_channels=3, ch=64, ch_mults=(1, 2, 4, 8), latent_dim=128):
            super().__init__()
            layers = [nn.Conv2d(in_channels, ch, 3, padding=1)]

            in_ch = ch
            for mult in ch_mults:
                out_ch = ch * mult
                layers += [
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    ResBlock(out_ch),
                ]
                in_ch = out_ch

            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())
            self.net = nn.Sequential(*layers)
            self.fc_mu = nn.Linear(in_ch, latent_dim)
            self.fc_logvar = nn.Linear(in_ch, latent_dim)

        def forward(self, x):
            h = self.net(x)
            return self.fc_mu(h), self.fc_logvar(h)

    class Decoder(nn.Module):
        def __init__(self, latent_dim=128, ch=64, ch_mults=(8, 4, 2, 1), out_channels=3, init_size=8):
            super().__init__()
            first_ch = ch * ch_mults[0]
            self.init_size = init_size
            self.fc = nn.Linear(latent_dim, first_ch * init_size * init_size)

            layers = []
            in_ch = first_ch
            for mult in ch_mults:
                out_ch = ch * mult
                layers += [
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                    ResBlock(out_ch),
                ]
                in_ch = out_ch

            layers.append(nn.Conv2d(in_ch, out_channels, 3, padding=1))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)

        def forward(self, z):
            h = self.fc(z)
            h = h.view(h.size(0), -1, self.init_size, self.init_size)
            return self.net(h)

    class VAE(nn.Module):
        def __init__(self, latent_dim=128, ch=64, ch_mults=(1, 2, 4, 8)):
            super().__init__()
            self.encoder = Encoder(3, ch, ch_mults, latent_dim)
            self.decoder = Decoder(latent_dim, ch, tuple(reversed(ch_mults)), 3)

        def reparameterize(self, mu, logvar):
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar

        def encode(self, x):
            mu, logvar = self.encoder(x)
            return self.reparameterize(mu, logvar)

        def decode(self, z):
            return self.decoder(z)

    return VAE


# ─── Training function ──────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/outputs": vol, "/cache": hf_cache},
    timeout=3600 * 6,
    secrets=[
        modal.Secret.from_name("huggingface-secret", required=False),
        modal.Secret.from_name("wandb-api-key", required=False),
    ],
)
def train(
    epochs: int = 5,
    latent_dim: int = 128,
    batch_size: int = 64,
    lr: float = 1e-4,
    kl_weight: float = 1e-4,
    image_size: int = 128,
    log_every: int = 50,
    save_every_epoch: int = 1,
    max_train_samples: int = 200_000,
):
    import os
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from datasets import load_dataset
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table
    import numpy as np
    import wandb

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Device:[/bold] {device}")
    if device.type == "cuda":
        console.print(f"[bold]GPU:[/bold] {torch.cuda.get_device_name()}")
        console.print(f"[bold]VRAM:[/bold] {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Output dirs ──
    run_name = f"vae_imagenet_z{latent_dim}_{int(time.time())}"
    out_dir = Path(f"/outputs/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # ── Dataset ──
    console.print("[bold yellow]Loading ImageNet...[/bold yellow]")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    def collate(batch):
        images = []
        for item in batch:
            img = item["image"].convert("RGB")
            images.append(transform(img))
        return torch.stack(images)

    # We'll iterate manually since it's streaming
    console.print("[green]Dataset ready (streaming mode)[/green]")

    # ── Model ──
    VAE = build_model_code()
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters())
    console.print(f"[bold]Model params:[/bold] {num_params:,}")

    # ── W&B ──
    wandb.init(
        project="vae-imagenet",
        name=run_name,
        config={
            "epochs": epochs,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "lr": lr,
            "kl_weight": kl_weight,
            "image_size": image_size,
            "max_train_samples": max_train_samples,
            "num_params": num_params,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        },
    )
    wandb.watch(model, log="gradients", log_freq=100)

    # ── Training loop ──
    console.print(f"\n[bold yellow]Training for {epochs} epochs[/bold yellow]")
    console.print(f"  batch_size={batch_size}, lr={lr}, kl_weight={kl_weight}")
    console.print(f"  image_size={image_size}, latent_dim={latent_dim}")
    console.print(f"  max_train_samples={max_train_samples:,}\n")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        samples_seen = 0

        batch_buf = []
        epoch_start = time.time()

        for sample in ds:
            img = sample["image"].convert("RGB")
            batch_buf.append(transform(img))

            if len(batch_buf) < batch_size:
                continue

            # Form batch
            x = torch.stack(batch_buf[:batch_size]).to(device)
            batch_buf = batch_buf[batch_size:]

            # Forward
            recon, mu, logvar = model(x)

            # Resize recon to match input if needed
            if recon.shape[-2:] != x.shape[-2:]:
                recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)

            # Losses
            recon_loss = F.mse_loss(recon, x, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_total_loss += loss.item()
            num_batches += 1
            samples_seen += batch_size
            global_step += 1

            # Log to W&B every step
            wandb.log({
                "train/recon_loss": recon_loss.item(),
                "train/kl_loss": kl_loss.item(),
                "train/total_loss": loss.item(),
                "train/samples_seen": samples_seen,
                "train/epoch": epoch,
            }, step=global_step)

            if global_step % log_every == 0:
                console.print(
                    f"  [step {global_step}] "
                    f"recon={recon_loss.item():.4f}  "
                    f"kl={kl_loss.item():.4f}  "
                    f"total={loss.item():.4f}  "
                    f"samples={samples_seen:,}"
                )

            if samples_seen >= max_train_samples:
                break

        epoch_time = time.time() - epoch_start
        avg_recon = epoch_recon_loss / max(num_batches, 1)
        avg_kl = epoch_kl_loss / max(num_batches, 1)
        avg_total = epoch_total_loss / max(num_batches, 1)

        # Epoch summary
        table = Table(title=f"Epoch {epoch + 1}/{epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Recon Loss", f"{avg_recon:.5f}")
        table.add_row("KL Loss", f"{avg_kl:.5f}")
        table.add_row("Total Loss", f"{avg_total:.5f}")
        table.add_row("Samples", f"{samples_seen:,}")
        table.add_row("Batches", f"{num_batches:,}")
        table.add_row("Time", f"{epoch_time:.1f}s")
        table.add_row("Throughput", f"{samples_seen / epoch_time:.0f} img/s")
        console.print(table)

        # Log epoch summary to W&B
        wandb.log({
            "epoch/recon_loss": avg_recon,
            "epoch/kl_loss": avg_kl,
            "epoch/total_loss": avg_total,
            "epoch/throughput": samples_seen / epoch_time,
            "epoch/time_s": epoch_time,
            "epoch/epoch": epoch + 1,
        }, step=global_step)

        # Save reconstructions
        model.eval()
        with torch.no_grad():
            sample_recon, _, _ = model(x[:8])
            if sample_recon.shape[-2:] != x.shape[-2:]:
                sample_recon = F.interpolate(sample_recon, size=x.shape[-2:], mode="bilinear", align_corners=False)

            from torchvision.utils import save_image
            comparison = torch.cat([x[:8], sample_recon], dim=0)
            save_image(
                comparison,
                out_dir / "samples" / f"epoch_{epoch + 1:03d}.png",
                nrow=8,
                padding=2,
            )

            # Random samples from prior
            z = torch.randn(8, latent_dim, device=device)
            gen = model.decode(z)
            if gen.shape[-2:] != (image_size, image_size):
                gen = F.interpolate(gen, size=(image_size, image_size), mode="bilinear", align_corners=False)
            save_image(
                gen,
                out_dir / "samples" / f"generated_epoch_{epoch + 1:03d}.png",
                nrow=8,
                padding=2,
            )

            # Log images to W&B
            wandb.log({
                "images/reconstructions": wandb.Image(
                    str(out_dir / "samples" / f"epoch_{epoch + 1:03d}.png"),
                    caption=f"Top: input, Bottom: reconstruction (epoch {epoch + 1})",
                ),
                "images/generated": wandb.Image(
                    str(out_dir / "samples" / f"generated_epoch_{epoch + 1:03d}.png"),
                    caption=f"Random samples from prior (epoch {epoch + 1})",
                ),
            }, step=global_step)

        console.print(f"  [green]Saved samples to {out_dir}/samples/[/green]")

        # Save checkpoint
        if (epoch + 1) % save_every_epoch == 0 or avg_total < best_loss:
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_total,
                "config": {
                    "latent_dim": latent_dim,
                    "batch_size": batch_size,
                    "lr": lr,
                    "kl_weight": kl_weight,
                    "image_size": image_size,
                },
            }
            ckpt_path = out_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt"
            torch.save(ckpt, ckpt_path)
            console.print(f"  [green]Saved checkpoint: {ckpt_path}[/green]")

            if avg_total < best_loss:
                best_loss = avg_total
                torch.save(ckpt, out_dir / "checkpoints" / "best.pt")
                console.print(f"  [bold green]New best model (loss={best_loss:.5f})[/bold green]")

        vol.commit()

    wandb.finish()

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Outputs: /outputs/{run_name}")
    console.print(f"  Best loss: {best_loss:.5f}")

    return {
        "run_name": run_name,
        "best_loss": best_loss,
        "global_step": global_step,
        "epochs": epochs,
    }


# ─── CLI entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    epochs: int = 5,
    latent_dim: int = 128,
    batch_size: int = 64,
    lr: float = 1e-4,
    kl_weight: float = 1e-4,
    image_size: int = 128,
    max_train_samples: int = 200_000,
):
    print(f"Launching VAE training on Modal (A10G GPU)")
    print(f"  epochs={epochs}, latent_dim={latent_dim}, batch_size={batch_size}")
    print(f"  image_size={image_size}, max_train_samples={max_train_samples:,}")
    print()

    result = train.remote(
        epochs=epochs,
        latent_dim=latent_dim,
        batch_size=batch_size,
        lr=lr,
        kl_weight=kl_weight,
        image_size=image_size,
        max_train_samples=max_train_samples,
    )

    print(f"\nResults: {result}")
    print(f"\nTo view outputs:")
    print(f"  modal volume ls vae-imagenet-outputs/{result['run_name']}/samples/")
    print(f"  modal volume get vae-imagenet-outputs/{result['run_name']}/samples/ ./samples/")
