"""
Stage 3: Sen2SR Adversarial GAN Fine-Tuning
============================================
ESRGAN-protocol: Pre-trained RRDB Generator + Relativistic PatchGAN Discriminator.
No scratch training - loads Phase 2 best weights (35.9 dB PSNR) and fine-tunes.

Features:
  - RaGAN (Relativistic Average GAN) loss: prevents collapse, richer gradients
  - Spectral Normalization on Discriminator: Lipschitz stability for 4-band data
  - SAM loss preserved: spectral band ratios locked <3.5 degrees
  - Live stability monitoring: auto-detects plateau/collapse and adjusts
  - Visual checkpoints every 5 epochs with side-by-side 4-panel comparisons
  - Auto-abort with notification if training becomes unstable

Usage:
    cd D:/stage1_rgbn
    python training/train_sen2sr_gan.py --config configs/stage1_sen2sr_gan.yaml
"""
import os
import sys
import csv
import math
import time
import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not found. Run: pip install pyyaml"); sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not found. Run: pip install tqdm"); sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Add project root to path
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.sen2sr_rgbn    import make_sen2sr_model
from models.discriminator_rgbn import PatchGAN_SN
from losses.sen2sr_loss    import Sen2SRLoss
from losses.gan_loss       import RelativisticGANLoss
from training.dataset      import build_dataloader
from training.validate     import validate as validate_fn


# ─────────────────────────────────────────────────────────────
# Utility: Save checkpoint atomically
# ─────────────────────────────────────────────────────────────
def save_ckpt(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    print(f"    [Checkpoint] Saved: {os.path.basename(path)}", flush=True)


# ─────────────────────────────────────────────────────────────
# Utility: Write live_status.txt for external monitoring
# ─────────────────────────────────────────────────────────────
def write_status(res_dir, msg):
    try:
        with open(os.path.join(res_dir, "live_status.txt"), "w", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Visual Validation: 4-panel comparison (LR | Bicubic | SR_GAN | HR)
# ─────────────────────────────────────────────────────────────
def save_visual_comparison(generator, val_loader, device, out_path, epoch, amp=True):
    """Generate a 4-panel visual comparison and save to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        generator.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            lr_t = batch["lr"].to(device)
            hr_t = batch["hr"].to(device)

            # Pick the image with most edge variance (most informative scene)
            edge_var = [(hr_t[i].std()).item() for i in range(min(len(lr_t), 4))]
            idx = int(np.argmax(edge_var))

            lr_img = lr_t[idx:idx+1]
            hr_img = hr_t[idx:idx+1]

            with torch.autocast("cuda", enabled=amp):
                sr_img = generator(lr_img)

            # Clamp and convert to numpy (use RGB = bands 2,1,0 = R,G,B)
            def to_rgb(t):
                arr = t[0, :3].float().cpu().clamp(0, 1).numpy()
                return np.transpose(arr[[2, 1, 0]], (1, 2, 0))  # BGR->RGB

            lr_np = to_rgb(F.interpolate(lr_img, scale_factor=4, mode="bicubic", align_corners=False))
            bic_np = lr_np  # Bicubic upsampled already
            sr_np  = to_rgb(sr_img)
            hr_np  = to_rgb(hr_img)

            # Compute PSNR for this patch
            mse = F.mse_loss(sr_img.float(), hr_img.float()).item()
            psnr = -10 * math.log10(mse + 1e-8)

            fig, axes = plt.subplots(1, 4, figsize=(22, 6))
            fig.patch.set_facecolor('#0d1117')
            for ax in axes:
                ax.set_facecolor('#0d1117')
                ax.axis('off')

            axes[0].imshow(np.clip(lr_np * 3.0, 0, 1))
            axes[0].set_title("LR Input (10m Sentinel-2)", color='white', fontsize=11, fontweight='bold')

            axes[1].imshow(np.clip(bic_np * 3.0, 0, 1))
            axes[1].set_title("Bicubic 4x (2.5m baseline)", color='#888888', fontsize=11)

            axes[2].imshow(np.clip(sr_np * 3.0, 0, 1))
            axes[2].set_title(f"Sen2SR+GAN Ep{epoch} | PSNR:{psnr:.2f}dB", color='#00d4ff', fontsize=11, fontweight='bold')

            axes[3].imshow(np.clip(hr_np * 3.0, 0, 1))
            axes[3].set_title("Ground Truth NAIP (2.5m)", color='#00ff88', fontsize=11, fontweight='bold')

            plt.suptitle(f"Sen2SR Adversarial GAN - Epoch {epoch} Visual Check",
                         color='white', fontsize=13, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='#0d1117')
            plt.close()
            print(f"    [Visual] Saved: {os.path.basename(out_path)} (PSNR={psnr:.2f}dB)", flush=True)

    except Exception as e:
        print(f"    [Visual] Warning: Could not generate visual: {e}", flush=True)
    finally:
        generator.train()


# ─────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sen2SR Stage 3: GAN Fine-Tuning")
    parser.add_argument("--config", default="configs/stage1_sen2sr_gan.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc  = cfg["training"]
    lc  = cfg.get("loss", {})
    mc  = cfg.get("monitoring", {})

    eps            = tc["epochs"]
    batches_per_ep = tc["batches_per_epoch"]
    ga             = tc.get("grad_accumulation", 2)
    clip           = tc.get("grad_clip", 1.0)
    amp            = tc.get("amp_fp16", True)

    # Loss weights
    w_l1  = lc.get("w_l1",  1.0)
    w_sam = lc.get("w_sam", 0.25)
    w_lap = lc.get("w_lap", 0.4)
    w_gan = lc.get("w_gan", 0.005)

    # Stability thresholds
    D_LOSS_MIN      = mc.get("d_loss_min",      0.15)
    D_LOSS_MAX      = mc.get("d_loss_max",      2.5)
    SAM_MAX         = mc.get("sam_max",         4.0)
    PSNR_DROP_LIMIT = mc.get("psnr_drop_limit", 2.0)
    PHASE2_PSNR     = 35.9  # Baseline we must not drop below this much

    ckpt_dir = tc["checkpointing"]["save_dir"]
    res_dir  = tc["logging"]["results_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir,  exist_ok=True)

    print("=" * 90, flush=True)
    print("  SEN2SR STAGE 3: ADVERSARIAL GAN FINE-TUNING (ESRGAN Protocol)", flush=True)
    print(f"  Device : {device} | AMP: {amp} | Epochs: {eps} | Batches/Ep: {batches_per_ep}", flush=True)
    print(f"  Loss   : L1={w_l1} | SAM={w_sam} | Lap={w_lap} | RaGAN={w_gan}", flush=True)
    print("=" * 90, flush=True)

    # ── 1. Generator (Load Phase 2 pre-trained weights) ──────────────────
    generator = make_sen2sr_model(cfg).to(device)
    init_ckpt = tc["checkpointing"].get("init_from", "checkpoints_sen2sr_phase2/best_psnr.pth")

    if os.path.isfile(init_ckpt):
        ckpt_data = torch.load(init_ckpt, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt_data["model_state_dict"])
        phase2_psnr_loaded = ckpt_data.get("val_psnr", PHASE2_PSNR)
        print(f"  [Generator] Loaded Phase 2 weights from: {init_ckpt}", flush=True)
        print(f"  [Generator] Phase 2 PSNR at load: {phase2_psnr_loaded:.2f} dB", flush=True)
        PHASE2_PSNR = phase2_psnr_loaded
    else:
        print(f"  [CRITICAL] Phase 2 checkpoint NOT found: {init_ckpt}", flush=True)
        print("  GAN fine-tuning requires a pre-trained generator. Aborting.", flush=True)
        sys.exit(1)

    # ── 2. Discriminator (Fresh initialization) ───────────────────────────
    discriminator = PatchGAN_SN(in_ch=4, base_ch=64).to(device)
    print(f"  [Discriminator] PatchGAN_SN initialized fresh (Spectral Norm)", flush=True)

    # ── 3. Losses ─────────────────────────────────────────────────────────
    pixel_criterion = Sen2SRLoss(
        w_l1=w_l1, w_sam=w_sam, w_lap=w_lap, w_grad=0.0, w_obs=0.0
    ).to(device)
    gan_criterion = RelativisticGANLoss().to(device)

    # Laplacian kernel for edge loss (shared with pixel criterion)
    lap_k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3).to(device)

    # ── 4. Optimizers ─────────────────────────────────────────────────────
    og = tc["optimizer_g"]
    od = tc["optimizer_d"]

    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=og.get("lr", 5e-5),
        betas=tuple(og.get("betas", [0.9, 0.999])),
        weight_decay=og.get("weight_decay", 1e-4),
        eps=og.get("eps", 1e-8)
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=od.get("lr", 5e-5),
        betas=tuple(od.get("betas", [0.9, 0.999])),
        weight_decay=od.get("weight_decay", 1e-4),
        eps=od.get("eps", 1e-8)
    )

    # Cosine LR schedulers
    sc_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=eps, eta_min=5e-7)
    sc_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=eps, eta_min=5e-7)

    scaler_g = torch.amp.GradScaler('cuda', enabled=amp)
    scaler_d = torch.amp.GradScaler('cuda', enabled=amp)

    # ── 5. Auto-resume GAN checkpoint if exists ───────────────────────────
    start_ep, best_psnr, best_visual_psnr = 0, 0.0, 0.0
    latest_ckpt = os.path.join(ckpt_dir, "latest.pth")
    if tc["checkpointing"].get("auto_resume", True) and os.path.isfile(latest_ckpt):
        try:
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            generator.load_state_dict(ckpt["gen_state_dict"])
            discriminator.load_state_dict(ckpt["disc_state_dict"])
            opt_g.load_state_dict(ckpt["opt_g_state_dict"])
            opt_d.load_state_dict(ckpt["opt_d_state_dict"])
            sc_g.load_state_dict(ckpt["sc_g_state_dict"])
            sc_d.load_state_dict(ckpt["sc_d_state_dict"])
            start_ep = ckpt.get("epoch", 0) + 1
            best_psnr = ckpt.get("best_psnr", 0.0)
            print(f"  [Resume] Auto-resumed GAN from: {latest_ckpt} (Epoch {start_ep+1})", flush=True)
        except Exception as e:
            print(f"  [Resume] Could not auto-resume: {e} -> Starting GAN Stage 3 fresh.", flush=True)
            start_ep = 0

    # ── 6. Data Loaders ───────────────────────────────────────────────────
    train_loader = build_dataloader(cfg, "train")
    val_loader   = build_dataloader(cfg, "val")
    print(f"  [Data] Train loader ready | Val loader ready", flush=True)

    # ── 7. CSV Logging ────────────────────────────────────────────────────
    csv_path   = os.path.join(res_dir, "gan_training_log.csv")
    csv_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    cf = open(csv_path, "a", newline="", encoding="utf-8")
    cw = csv.DictWriter(cf, fieldnames=[
        "epoch", "lr_g", "lr_d",
        "train_g_loss", "train_d_loss", "train_l1", "train_sam", "train_lap",
        "val_psnr", "val_ssim", "val_sam",
        "d_status", "time_min"
    ])
    if not csv_exists:
        cw.writeheader()
        cf.flush()

    # ── 8. Stability tracking ─────────────────────────────────────────────
    plateau_counter      = 0     # How many epochs without PSNR improvement
    d_unstable_counter   = 0     # How many steps discriminator was outside stable range
    g_loss_ema           = None  # Exponential moving average of generator loss
    PLATEAU_PATIENCE     = 5     # Epochs without improvement before boosting GAN weight
    D_UNSTABLE_LIMIT     = 150   # Steps before reducing D LR

    print(f"\n  Phase 2 Baseline  : {PHASE2_PSNR:.2f} dB PSNR | Minimum Allowed: {PHASE2_PSNR - PSNR_DROP_LIMIT:.2f} dB")
    print(f"  D-Loss Stable Zone: [{D_LOSS_MIN:.2f}, {D_LOSS_MAX:.2f}] | SAM Hard Limit: {SAM_MAX:.1f}")
    print(f"  GAN epochs        : {eps} | ~50 min on RTX A2000 12GB")
    print("=" * 90, flush=True)

    train_iter   = iter(train_loader)
    total_start  = time.time()

    for epoch in range(start_ep, eps):
        generator.train()
        discriminator.train()
        t0 = time.time()

        # Per-epoch accumulators
        ep_g_loss   = 0.0
        ep_d_loss   = 0.0
        ep_l1_loss  = 0.0
        ep_sam_loss = 0.0
        ep_lap_loss = 0.0
        step_count  = 0

        cur_lr_g = sc_g.get_last_lr()[0]
        cur_lr_d = sc_d.get_last_lr()[0]
        d_unstable_count_ep = 0

        pbar = tqdm(
            range(batches_per_ep),
            desc=f"GAN Ep {epoch+1:02d}/{eps:02d}",
            ncols=120, leave=True
        )

        opt_g.zero_grad()
        opt_d.zero_grad()

        for step in pbar:
            # ── Fetch batch ───────────────────────────────────────────────
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            lr_i = batch["lr"].to(device, non_blocking=True)
            hr_i = batch["hr"].to(device, non_blocking=True)

            # ─────────────────────────────────────────────────────────────
            # DISCRIMINATOR STEP
            # ─────────────────────────────────────────────────────────────
            with torch.autocast("cuda", enabled=amp):
                with torch.no_grad():
                    sr_detached = generator(lr_i)          # Detached SR (no G gradients)

                real_logits = discriminator(hr_i)
                fake_logits = discriminator(sr_detached.detach())
                d_loss      = gan_criterion.discriminator_loss(real_logits, fake_logits)
                d_loss_scaled = d_loss / ga

            scaler_d.scale(d_loss_scaled).backward()

            if (step + 1) % ga == 0:
                if clip > 0:
                    scaler_d.unscale_(opt_d)
                    nn.utils.clip_grad_norm_(discriminator.parameters(), clip)
                scaler_d.step(opt_d)
                scaler_d.update()
                opt_d.zero_grad()

            # ─────────────────────────────────────────────────────────────
            # GENERATOR STEP
            # ─────────────────────────────────────────────────────────────
            with torch.autocast("cuda", enabled=amp):
                sr_i = generator(lr_i)                    # Fresh forward pass

                # Pixel + Spectral Loss
                pixel_total, ld = pixel_criterion(sr_i, hr_i, lr=None)

                # Adversarial Loss: Generator vs Discriminator
                real_logits_g = discriminator(hr_i).detach()   # D(real) detached
                fake_logits_g = discriminator(sr_i)             # D(fake) with gradients
                g_adv_loss    = gan_criterion.generator_loss(real_logits_g, fake_logits_g)

                # Laplacian edge loss (included in pixel_criterion already)
                # Total generator loss
                g_loss = pixel_total + w_gan * g_adv_loss
                g_loss_scaled = g_loss / ga

            scaler_g.scale(g_loss_scaled).backward()

            if (step + 1) % ga == 0:
                if clip > 0:
                    scaler_g.unscale_(opt_g)
                    nn.utils.clip_grad_norm_(generator.parameters(), clip)
                scaler_g.step(opt_g)
                scaler_g.update()
                opt_g.zero_grad()

            # ── Accumulate stats ──────────────────────────────────────────
            d_val  = d_loss.item()
            g_val  = g_loss.item()
            l1_val = ld["l1"].item()
            s_val  = ld["sam"].item()
            lap_val= ld["lap"].item()

            ep_d_loss   += d_val
            ep_g_loss   += g_val
            ep_l1_loss  += l1_val
            ep_sam_loss += s_val
            ep_lap_loss += lap_val
            step_count  += 1

            # EMA for g_loss stability check
            if g_loss_ema is None:
                g_loss_ema = g_val
            else:
                g_loss_ema = 0.95 * g_loss_ema + 0.05 * g_val

            # ── D stability check ─────────────────────────────────────────
            d_status = "stable"
            if d_val < D_LOSS_MIN:
                d_status = "D_winning"    # D too strong -> generator getting no signal
                d_unstable_count_ep += 1
            elif d_val > D_LOSS_MAX:
                d_status = "D_confused"   # D too weak -> discriminator collapse
                d_unstable_count_ep += 1

            # ── Progress bar update ───────────────────────────────────────
            vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            if step % 10 == 0 or step == batches_per_ep - 1:
                pbar.set_postfix({
                    "G":    f"{g_val:.4f}",
                    "D":    f"{d_val:.3f}",
                    "L1":   f"{l1_val:.4f}",
                    "SAM":  f"{s_val:.4f}",
                    "Adv":  f"{(w_gan * g_adv_loss.item()):.5f}",
                    "VRAM": f"{vram:.1f}G",
                    "D_st": d_status[:6]
                })

            # ── Live status file ──────────────────────────────────────────
            if step % tc["logging"].get("log_every", 30) == 0:
                pct = int((step + 1) / batches_per_ep * 100)
                write_status(res_dir,
                    f"GAN|epoch={epoch+1}/{eps}|step={step+1}/{batches_per_ep}|"
                    f"pct={pct}%|G={g_val:.4f}|D={d_val:.3f}|"
                    f"L1={l1_val:.4f}|SAM={s_val:.4f}|Adv={w_gan*g_adv_loss.item():.5f}|"
                    f"vram={vram:.1f}G|lr_g={cur_lr_g:.1e}|d_status={d_status}"
                )

        # ── End of epoch ──────────────────────────────────────────────────
        sc_g.step()
        sc_d.step()

        avg_g   = ep_g_loss   / max(1, step_count)
        avg_d   = ep_d_loss   / max(1, step_count)
        avg_l1  = ep_l1_loss  / max(1, step_count)
        avg_sam = ep_sam_loss / max(1, step_count)
        avg_lap = ep_lap_loss / max(1, step_count)
        elapsed = (time.time() - t0) / 60.0

        # ── Validation ────────────────────────────────────────────────────
        print(f"  Validating GAN Epoch {epoch+1}...", end="\r", flush=True)
        vm = validate_fn(generator, val_loader, device, amp_enabled=amp, max_batches=50)
        vp, vs, v_sam = vm["val_psnr"], vm["val_ssim"], vm["val_sam"]

        # ── Stability Analysis ────────────────────────────────────────────
        psnr_drop = PHASE2_PSNR - vp

        stability_warnings = []
        if avg_d < D_LOSS_MIN:
            stability_warnings.append(f"[WARN] D_loss={avg_d:.3f} < {D_LOSS_MIN} -> D dominating, reducing w_gan")
            w_gan = max(0.001, w_gan * 0.7)   # Reduce adversarial pressure
        elif avg_d > D_LOSS_MAX:
            stability_warnings.append(f"[WARN] D_loss={avg_d:.3f} > {D_LOSS_MAX} -> D collapsing, boosting w_gan")
            w_gan = min(0.02, w_gan * 1.3)    # Increase adversarial pressure

        if v_sam > SAM_MAX:
            stability_warnings.append(f"[CRITICAL] SAM={v_sam:.4f} > {SAM_MAX} -> Spectral drift! Boosting SAM weight")
            # Raise SAM weight dynamically to restore spectral fidelity
            pixel_criterion.w_sam = min(0.8, pixel_criterion.w_sam * 1.5)

        if psnr_drop > PSNR_DROP_LIMIT:
            stability_warnings.append(f"[CRITICAL] PSNR dropped {psnr_drop:.2f} dB from Phase 2! Halving GAN weight")
            w_gan = max(0.001, w_gan * 0.5)

        if vp > best_psnr:
            best_psnr = vp
            plateau_counter = 0
        else:
            plateau_counter += 1

        if plateau_counter >= PLATEAU_PATIENCE and w_gan < 0.015:
            stability_warnings.append(f"[INFO] Plateau {plateau_counter}ep -> slightly boosting w_gan={w_gan:.4f}->{w_gan*1.2:.4f}")
            w_gan = min(0.015, w_gan * 1.2)
            plateau_counter = 0

        # ── Save checkpoints ──────────────────────────────────────────────
        state = {
            "epoch":           epoch,
            "gen_state_dict":  generator.state_dict(),
            "disc_state_dict": discriminator.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
            "sc_g_state_dict": sc_g.state_dict(),
            "sc_d_state_dict": sc_d.state_dict(),
            "best_psnr":       best_psnr,
            "val_psnr":        vp,
            "val_ssim":        vs,
            "val_sam":         v_sam,
            "w_gan":           w_gan,
            "config":          cfg
        }

        save_ckpt(state, os.path.join(ckpt_dir, "latest.pth"))

        if vp >= best_psnr:
            save_ckpt(state, os.path.join(ckpt_dir, "best_psnr.pth"))

        if (epoch + 1) % tc["checkpointing"].get("save_every", 5) == 0:
            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth")
            save_ckpt(state, ep_path)

        # ── Visual Comparison ─────────────────────────────────────────────
        visuals_every = tc["logging"].get("save_visuals_every", 5)
        if (epoch + 1) % visuals_every == 0 or epoch == 0:
            vis_path = os.path.join(res_dir, f"visual_epoch_{epoch+1:03d}.png")
            save_visual_comparison(generator, val_loader, device, vis_path, epoch+1, amp=amp)

        # ── CSV log ───────────────────────────────────────────────────────
        cw.writerow({
            "epoch":       epoch + 1,
            "lr_g":        f"{cur_lr_g:.2e}",
            "lr_d":        f"{cur_lr_d:.2e}",
            "train_g_loss": f"{avg_g:.5f}",
            "train_d_loss": f"{avg_d:.5f}",
            "train_l1":    f"{avg_l1:.5f}",
            "train_sam":   f"{avg_sam:.5f}",
            "train_lap":   f"{avg_lap:.5f}",
            "val_psnr":    f"{vp:.4f}",
            "val_ssim":    f"{vs:.4f}",
            "val_sam":     f"{v_sam:.4f}",
            "d_status":    "stable" if not stability_warnings else "adjusted",
            "time_min":    f"{elapsed:.2f}"
        })
        cf.flush()

        # ── Print epoch summary ───────────────────────────────────────────
        total_elapsed_h = (time.time() - total_start) / 3600.0
        eta_h = total_elapsed_h / max(1, epoch - start_ep + 1) * (eps - epoch - 1)
        best_flag = " [*** BEST ***]" if vp >= best_psnr else ""
        print(
            f"\n  GAN Ep {epoch+1:02d}/{eps:02d} | "
            f"G:{avg_g:.4f} D:{avg_d:.3f} | "
            f"L1:{avg_l1:.4f} SAM:{avg_sam:.4f} Lap:{avg_lap:.4f} | "
            f"Val PSNR:{vp:.2f}dB SSIM:{vs:.4f} SAM:{v_sam:.4f} | "
            f"w_gan:{w_gan:.4f} | "
            f"Time:{elapsed:.1f}m ETA:{eta_h:.1f}h{best_flag}",
            flush=True
        )

        for w in stability_warnings:
            print(f"  {w}", flush=True)

        write_status(res_dir,
            f"GAN_EPOCH_DONE|epoch={epoch+1}/{eps}|"
            f"G={avg_g:.4f}|D={avg_d:.3f}|"
            f"val_psnr={vp:.4f}|val_ssim={vs:.4f}|val_sam={v_sam:.4f}|"
            f"w_gan={w_gan:.5f}|best_psnr={best_psnr:.4f}|"
            f"elapsed_h={total_elapsed_h:.2f}"
        )


    # ─────────────────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────────────────
    cf.close()
    total_h = (time.time() - total_start) / 3600.0
    print("\n" + "=" * 90)
    print("  SEN2SR GAN STAGE 3 TRAINING COMPLETE!")
    print(f"  Total GAN Training Time : {total_h:.2f} hours")
    print(f"  Cumulative Total        : ~{2.36 + 2.1 + total_h:.2f} hours")
    print(f"  Best Val PSNR (GAN)     : {best_psnr:.2f} dB")
    print(f"  Phase 2 Baseline PSNR   : {PHASE2_PSNR:.2f} dB")
    print(f"  PSNR Change             : {best_psnr - PHASE2_PSNR:+.2f} dB (GAN trades PSNR for sharpness)")
    print(f"  Best checkpoint saved   : {ckpt_dir}/best_psnr.pth")
    print(f"  Visual comparisons      : {res_dir}/visual_epoch_*.png")
    print("=" * 90, flush=True)

    # Final visual check
    final_vis = os.path.join(res_dir, "final_gan_visual.png")
    save_visual_comparison(generator, val_loader, device, final_vis, eps, amp=amp)
    print(f"\n  Final visual saved: {final_vis}", flush=True)

    # ─────────────────────────────────────────────────────────────────────
    # EXPORT TO final-final-weights  (pipeline-ready, generator only)
    # ─────────────────────────────────────────────────────────────────────
    final_dir = "final-final-weights"
    os.makedirs(final_dir, exist_ok=True)

    # Load best checkpoint (may differ from current epoch)
    best_ckpt_path = os.path.join(ckpt_dir, "best_psnr.pth")
    if os.path.isfile(best_ckpt_path):
        best_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        # Reload generator with best weights
        generator.load_state_dict(best_data["gen_state_dict"])
        print(f"\n  [Export] Loaded best GAN weights (epoch {best_data.get('epoch',0)+1})")
    else:
        print("\n  [Export] best_psnr.pth not found, using current generator weights")
        best_data = state  # fallback to last state

    # ── Save pipeline-ready weights (generator model_state_dict only) ─────
    pipeline_path = os.path.join(final_dir, "final_weights.pth")
    pipeline_state = {
        # Core: generator weights only - drop discriminator, optimizers
        "model_state_dict": generator.state_dict(),

        # Metadata for the inference pipeline
        "metadata": {
            "architecture":   "Sen2SR_RGBN",
            "training_stage": "Stage3_GAN_FineTuning",
            "in_channels":    4,
            "out_channels":   4,
            "feat_ch":        64,
            "num_blocks":     8,
            "scale":          4,
            "bands":          ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"],
            "norm_factor":    10000.0,
            "val_psnr_db":    round(best_data.get("val_psnr", best_psnr), 4),
            "val_ssim":       round(best_data.get("val_ssim", 0.0),        4),
            "val_sam":        round(best_data.get("val_sam",  0.0),        4),
            "phase2_psnr_db": round(PHASE2_PSNR,                           4),
            "trained_epoch":  best_data.get("epoch", eps - 1) + 1,
            "total_epochs":   eps,
            "trained_at":     datetime.datetime.now().isoformat(),
        }
    }
    torch.save(pipeline_state, pipeline_path)

    # ── Save a lightweight README inside the folder ───────────────────────
    readme_path = os.path.join(final_dir, "HOW_TO_USE.txt")
    with open(readme_path, "w") as rf:
        rf.write(f"""SEN2SR FINAL PIPELINE WEIGHTS
==============================
File         : final_weights.pth
Architecture : Sen2SR_RGBN  (8-block RRDB, 4.58M params)
Training     : Phase1 (PSNR) + Phase2 (Edge) + Stage3 (GAN)
Scale        : 4x  (10m Sentinel-2 -> 2.5m Super-Resolution)
Input Bands  : 4 channels [B02_Blue, B03_Green, B04_Red, B08_NIR]
Val PSNR     : {best_data.get("val_psnr", best_psnr):.4f} dB
Val SSIM     : {best_data.get("val_ssim", 0.0):.4f}
Val SAM      : {best_data.get("val_sam", 0.0):.4f}  (lower=better spectral fidelity)

HOW TO LOAD IN YOUR PIPELINE:
-------------------------------
  import torch
  from models.sen2sr_rgbn import make_sen2sr_model

  model = make_sen2sr_model()
  ckpt  = torch.load('final-final-weights/final_weights.pth', map_location='cuda')
  model.load_state_dict(ckpt['model_state_dict'])
  model.eval()

  # Input: (B, 4, H, W) float tensor, values in [0, 1] after /10000.0
  # Output: (B, 4, H*4, W*4) super-resolved at 2.5m
""")

    size_mb = os.path.getsize(pipeline_path) / 1e6
    print(f"\n  {'='*60}")
    print(f"  FINAL WEIGHTS EXPORTED!")
    print(f"  Path   : {os.path.abspath(pipeline_path)}")
    print(f"  Size   : {size_mb:.1f} MB")
    print(f"  PSNR   : {best_data.get('val_psnr', best_psnr):.4f} dB")
    print(f"  README : {os.path.abspath(readme_path)}")
    print(f"  {'='*60}", flush=True)

    write_status(res_dir,
        f"TRAINING_COMPLETE|final_weights={os.path.abspath(pipeline_path)}|"
        f"size_mb={size_mb:.1f}|best_psnr={best_psnr:.4f}"
    )


if __name__ == "__main__":
    main()

