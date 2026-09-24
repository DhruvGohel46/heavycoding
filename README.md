# Stage 1 RGBN SR

Custom 18.5M-parameter SwinIR-style model for 4-band satellite super-resolution.
Input: 10m RGBN Sentinel-2 | Output: 2.5m RGBN (x4)

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download dataset (~100GB)
huggingface-cli download aliFerdinand/SEN2NAIPv2 --local-dir data/raw

# 2. Inspect
python scripts/00_inspect_dataset.py

# 3. Split
python scripts/01_create_splits.py

# 4. Build LMDB cache (run once, ~45 min)
python scripts/02_build_lmdb_cache.py

# 5. Overfit test (MUST PASS)
python scripts/03_overfit_test.py

# 6. Smoke test
python scripts/04_smoke_test.py

# 7. Ablations (D and E)
python scripts/05_run_ablation.py

# 8. Full training (auto-resumes on restart)
python training/train.py --config configs/stage1_swinir.yaml

# 9. Evaluate
python scripts/06_evaluate_final.py
```

## Auto-resume after shutdown

Just run the same command again. latest.pth is saved every 5 epochs.
