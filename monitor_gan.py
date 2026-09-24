"""
Sen2SR GAN Training Monitor
============================
Run this in a separate CMD window to watch training in real-time.
Auto-refreshes every 3 seconds.

Usage:
    cd D:\stage1_rgbn
    python monitor_gan.py
"""
import os, time, sys, datetime

STATUS_FILE  = r"D:\stage1_rgbn\results_sen2sr_gan\live_status.txt"
LOG_CSV      = r"D:\stage1_rgbn\results_sen2sr_gan\gan_training_log.csv"
TOTAL_EPOCHS = 20
SEC_PER_EP   = 3 * 60   # ~3 min/epoch at 3 it/s x 600 steps

STABLE_COLOR  = "\033[92m"   # green
WARN_COLOR    = "\033[93m"   # yellow
ERR_COLOR     = "\033[91m"   # red
CYAN          = "\033[96m"
BOLD          = "\033[1m"
RESET         = "\033[0m"

os.system("cls")
print(f"{BOLD}{CYAN}")
print("=" * 72)
print("   SEN2SR  |  STAGE 3 GAN FINE-TUNING  |  LIVE MONITOR")
print("=" * 72)
print(f"{RESET}")

def clear():
    os.system("cls")

def read_status():
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def parse_status(line):
    d = {}
    for part in line.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d

def read_csv_history():
    rows = []
    try:
        with open(LOG_CSV, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header = lines[0].strip().split(",")
        for line in lines[1:]:
            vals = line.strip().split(",")
            if len(vals) == len(header):
                rows.append(dict(zip(header, vals)))
    except Exception:
        pass
    return rows

def bar(pct, width=40):
    filled = int(width * pct / 100)
    b = "#" * filled + "-" * (width - filled)
    return "[" + b + "] " + str(round(pct)) + "%"

start_ts = time.time()

while True:
    try:
        clear()
        now = datetime.datetime.now().strftime("%H:%M:%S")

        print("=" * 72)
        print(f"   SEN2SR GAN MONITOR  |  {now}  |  Press Ctrl+C to exit")
        print("=" * 72)
        print()

        raw = read_status()

        if raw is None:
            print("  Waiting for training to start...")
            print(f"  Status file: {STATUS_FILE}")
        else:
            s = parse_status(raw)

            # ── Detect training state ─────────────────────────────────────
            if raw.startswith("TRAINING_COMPLETE"):
                print()
                print("  ===  TRAINING COMPLETE!  ===")
                print(f"  Final weights: D:\\stage1_rgbn\\final-final-weights\\final_weights.pth")
                print(f"  Best PSNR    : {s.get('best_psnr','?')} dB")
                print()
                break

            elif raw.startswith("GAN_EPOCH_DONE"):
                ep   = s.get("epoch", "?")
                vp   = s.get("val_psnr", "?")
                vs   = s.get("val_ssim", "?")
                vsam = s.get("val_sam", "?")
                elh  = float(s.get("elapsed_h", 0))

                try:
                    ep_num = int(ep.split("/")[0])
                    ep_pct = ep_num / TOTAL_EPOCHS * 100
                    remaining_ep = TOTAL_EPOCHS - ep_num
                    eta_min = remaining_ep * SEC_PER_EP / 60
                    eta_str = f"~{eta_min:.0f} min" if eta_min < 60 else f"~{eta_min/60:.1f} hr"
                except Exception:
                    ep_pct, eta_str = 0, "?"

                print(f"  STATUS   : Epoch {ep} DONE - Validating / Saving checkpoint")
                print(f"  PROGRESS : {bar(ep_pct)}")
                print()
                print(f"  Val PSNR   : {vp} dB")
                print(f"  Val SSIM   : {vs}")
                print(f"  Val SAM    : {vsam}  (lower = better spectral fidelity)")
                print()
                print(f"  ETA to finish : {eta_str}")
                print(f"  Elapsed       : {elh:.2f} hr")

            elif raw.startswith("GAN"):
                ep   = s.get("epoch", "?")
                pct  = s.get("pct", "0%").replace("%", "")
                step = s.get("step", "?")
                G    = s.get("G", "?")
                D    = s.get("D", "?")
                L1   = s.get("L1", "?")
                SAM  = s.get("SAM", "?")
                Adv  = s.get("Adv", "?")
                vram = s.get("vram", "?")
                lr_g = s.get("lr_g", "?")
                dst  = s.get("d_status", "stable")

                try:
                    ep_num      = int(ep.split("/")[0])
                    step_pct    = float(pct)
                    overall_pct = (ep_num - 1 + step_pct / 100) / TOTAL_EPOCHS * 100

                    elapsed_s  = time.time() - start_ts
                    steps_done = (ep_num - 1) * 600 + (step_pct / 100 * 600)
                    total_steps = TOTAL_EPOCHS * 600
                    if steps_done > 30:
                        sps           = steps_done / elapsed_s
                        remaining_s   = (total_steps - steps_done) / sps
                        eta_min       = remaining_s / 60
                        eta_str       = f"{eta_min:.0f} min" if eta_min < 60 else f"{eta_min/60:.1f} hr"
                    else:
                        eta_min = (TOTAL_EPOCHS - ep_num + 1) * SEC_PER_EP / 60
                        eta_str = f"~{eta_min:.0f} min  (estimate)"
                except Exception:
                    overall_pct, eta_str = 0, "calculating..."

                print(f"  STATUS   : TRAINING  -  Epoch {ep}  |  Step {step}")
                print(f"  OVERALL  : {bar(overall_pct)}")
                print()
                print(f"  G  Loss    : {G}")
                print(f"  D  Loss    : {D}   [{dst}]")
                print(f"  L1 Loss    : {L1}")
                print(f"  SAM Loss   : {SAM}   (spectral integrity)")
                print(f"  Adv Loss   : {Adv}")
                print(f"  VRAM Used  : {vram}")
                print(f"  LR (Gen)   : {lr_g}")
                print()
                print(f"  ETA TO FINISH : {eta_str}")
                print()

                try:
                    d_val = float(D)
                    if d_val < 0.15:
                        print("  [WARN] D winning too hard! Auto-reducing w_gan...")
                    elif d_val > 2.5:
                        print("  [WARN] D collapsing! Auto-boosting w_gan...")
                    else:
                        print(f"  [OK]   Discriminator stable ({D}) - training healthy!")
                except Exception:
                    pass

        # ── Epoch History Table ───────────────────────────────────────────
        rows = read_csv_history()
        if rows:
            print()
            print("  " + "-" * 68)
            print("  EPOCH HISTORY:")
            print(f"  {'Ep':>4}  {'Val PSNR':>10}  {'SSIM':>7}  {'SAM':>7}  {'G Loss':>8}  {'D Loss':>8}  {'Min':>5}")
            print(f"  {'-'*4}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*5}")
            for r in rows[-10:]:
                ep_r = r.get("epoch", "")
                vp_r = r.get("val_psnr", "")
                vs_r = r.get("val_ssim", "")
                sa_r = r.get("val_sam", "")
                gl_r = r.get("train_g_loss", "")
                dl_r = r.get("train_d_loss", "")
                tm_r = r.get("time_min", "")
                try:
                    all_psnrs = [float(x.get("val_psnr", "0")) for x in rows]
                    is_best = float(vp_r) == max(all_psnrs)
                    star = " <-- BEST" if is_best else ""
                except Exception:
                    star = ""
                print(f"  {ep_r:>4}  {vp_r:>10}  {vs_r:>7}  {sa_r:>7}  {gl_r:>8}  {dl_r:>8}  {tm_r:>5}{star}")

        print()
        print("  " + "-" * 68)
        print("  OUTPUT PATHS:")
        print("  CSV Log  : D:\\stage1_rgbn\\results_sen2sr_gan\\gan_training_log.csv")
        print("  Visuals  : D:\\stage1_rgbn\\results_sen2sr_gan\\visual_epoch_*.png")
        print("  Weights  : D:\\stage1_rgbn\\checkpoints_sen2sr_gan\\")
        print("  FINAL    : D:\\stage1_rgbn\\final-final-weights\\final_weights.pth")
        print()
        print("  Refreshing every 3 sec...  Ctrl+C to exit")
        print()

        time.sleep(3)

    except KeyboardInterrupt:
        print("\n  Monitor stopped.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n  Monitor error: {e}")
        time.sleep(3)
