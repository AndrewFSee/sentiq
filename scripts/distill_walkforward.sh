#!/usr/bin/env bash
# Walk-forward students for the distillation recall check (step 3): student k is trained only on
# documents tradable before cut k and scores [cut k, cut k+1). Sequential (one GPU), each run
# resumes from its checkpoint after a crash (up to 5 attempts). Same settings as run full_v1.
#   bash scripts/distill_walkforward.sh        # from the project root; log: data/distill/runs/wf.log
set -u
cd "$(dirname "$0")/.."
# base Python + scripts/nf.py: Smart App Control blocks the venv's own python.exe (see nf.py)
PY="$APPDATA/uv/python/cpython-3.11.12-windows-x86_64-none/python.exe"
LOG=data/distill/runs/wf.log
mkdir -p data/distill/runs
cuts=(2021-01-01 2022-07-01 2024-01-01 2025-07-01 2026-07-01)
for k in 0 1 2 3; do
    name="wf$((k + 1))"
    if [ -f "data/distill/runs/$name/metrics.json" ]; then  # finished earlier: skip
        echo "=== $name already done, skipped $(date +%F\ %T)" >> "$LOG"
        continue
    fi
    for attempt in 1 2 3 4 5; do
        echo "=== $name attempt $attempt $(date +%F\ %T)" >> "$LOG"
        HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 "$PY" -u scripts/nf.py scripts/distill_train.py \
            --name "$name" --sources edgar,news_api --max-len 2048 --batch-tokens 3072 \
            --batch-size 64 --grad-accum 10 --epochs 1 --resume \
            --train-before "${cuts[$k]}" --predict-from "${cuts[$k]}" \
            --predict-to "${cuts[$((k + 1))]}" >> "$LOG" 2>&1
        rc=$?
        echo "exit=$rc ($name)" >> "$LOG"
        [ $rc -eq 0 ] && { rm -f "data/distill/runs/$name/ckpt.pt"; break; }
        sleep 30
    done
done
echo "=== all done $(date +%F\ %T)" >> "$LOG"
