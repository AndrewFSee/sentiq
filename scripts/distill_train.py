"""Train a small local "student" model on the LLM's labels (distillation dataset v1).

    uv sync --extra finbert        # torch (CUDA build) + transformers
    uv run newsfeed distill-data   # data/distill/v1/dataset.parquet
    uv run python scripts/distill_train.py --name news_smoke --sources news_api --max-len 256

Model: an encoder (default ModernBERT-base, 8k context) + three heads on the mean-pooled
output: sentiment (tanh, MSE), relevance (3 classes) and event type (8 classes), both
cross-entropy. Input = the exact user turn the LLM saw (target company, document type, title,
body), truncated to --max-len tokens. Trains on --train-splits, reports agreement with the LLM
on every other split, and with the hand labels on the gold split. Never sees the holdout (the
dataset has none). Outputs in data/distill/runs/<name>/: metrics.json, predictions.parquet,
config.json, and the weights with --save-model. No API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from newsfeed.distill.dataset import EVENT_TYPES, RELEVANCE  # noqa: E402

DIRECTION = {"negative": -1, "neutral": 0, "positive": 1}


class Student(nn.Module):
    def __init__(self, name: str, grad_ckpt: bool) -> None:
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        if grad_ckpt:
            self.enc.gradient_checkpointing_enable()
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.sent = nn.Linear(h, 1)
        self.rel = nn.Linear(h, len(RELEVANCE))
        self.evt = nn.Linear(h, len(EVENT_TYPES))

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hid = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(hid.dtype)
        pooled = self.drop((hid * m).sum(1) / m.sum(1).clamp(min=1.0))
        return torch.tanh(self.sent(pooled)).squeeze(-1), self.rel(pooled), self.evt(pooled)


def batches(
    lengths: np.ndarray,
    size: int,
    shuffle: bool,
    rng: random.Random,
    max_tokens: int | None = None,
) -> list[list[int]]:
    """Length-grouped batches: sort within chunks of 2,000 rows so padding stays small. A batch
    holds at most ``size`` rows and, with ``max_tokens``, at most that many padded tokens
    (rows x longest row), so short news and 2k-token 8-Ks both fill the GPU."""
    idx = list(range(len(lengths)))
    if shuffle:
        rng.shuffle(idx)
    out = []
    for s in range(0, len(idx), 2000):
        cur: list[int] = []
        for i in sorted(idx[s : s + 2000], key=lambda i: lengths[i]):
            width = lengths[i]  # ascending order: the newest row is the widest
            if cur and (len(cur) >= size or (max_tokens and (len(cur) + 1) * width > max_tokens)):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


class Tokens:
    """Ragged token ids as one flat int32 array + offsets: ~4 bytes per token instead of the
    ~36 of Python int lists (148k documents as lists took several GB and got paged out on a
    16 GB laptop, starving the GPU)."""

    def __init__(self, flat: np.ndarray, offs: np.ndarray) -> None:
        self.flat, self.offs = flat, offs
        self.lengths = np.diff(offs)

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.flat[self.offs[i] : self.offs[i + 1]]

    def subset(self, idx: np.ndarray) -> Tokens:
        parts = [self[int(i)] for i in idx]
        offs = np.zeros(len(parts) + 1, np.int64)
        offs[1:] = np.cumsum([len(p) for p in parts])
        flat = np.concatenate(parts) if parts else np.zeros(0, np.int32)
        return Tokens(flat, offs)


def tokenize(tok, texts: list[str], max_len: int, cache: Path | None) -> Tokens:
    """Tokenize in chunks (texts pre-cut to 8 characters per token, far above the ~4 a
    tokenizer averages, so nothing within max_len is lost) and cache to ``cache``."""
    if cache is not None and cache.exists():
        z = np.load(cache)
        if len(z["offs"]) == len(texts) + 1:
            return Tokens(z["flat"], z["offs"])
    chunks, lens = [], []
    for s in range(0, len(texts), 2000):
        part = [t[: max_len * 8] for t in texts[s : s + 2000]]
        for ids in tok(part, truncation=True, max_length=max_len)["input_ids"]:
            chunks.append(np.asarray(ids, np.int32))
            lens.append(len(ids))
    offs = np.zeros(len(lens) + 1, np.int64)
    offs[1:] = np.cumsum(lens)
    out = Tokens(np.concatenate(chunks) if chunks else np.zeros(0, np.int32), offs)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, flat=out.flat, offs=out.offs)
    return out


def collate(ids: Tokens, rows: list[int], pad: int, dev: str) -> tuple:
    width = int(max(ids.lengths[i] for i in rows))
    x = torch.full((len(rows), width), pad, dtype=torch.long)
    m = torch.zeros((len(rows), width), dtype=torch.long)
    for j, i in enumerate(rows):
        n = int(ids.lengths[i])
        x[j, :n] = torch.from_numpy(ids[i].astype(np.int64))
        m[j, :n] = 1
    return x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)


@torch.no_grad()
def predict(
    model: Student, ids: Tokens, pad: int, dev: str, bs: int, max_tokens: int | None
) -> dict:
    model.eval()
    lengths = ids.lengths
    sent = np.zeros(len(ids), np.float32)
    rel = np.zeros((len(ids), len(RELEVANCE)), np.float32)
    evt = np.zeros((len(ids), len(EVENT_TYPES)), np.float32)
    for rows in batches(lengths, bs, False, random.Random(0), max_tokens):
        x, m = collate(ids, rows, pad, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            s, r, e = model(x, m)
        sent[rows] = s.float().cpu().numpy()
        rel[rows] = torch.softmax(r.float(), -1).cpu().numpy()
        evt[rows] = torch.softmax(e.float(), -1).cpu().numpy()
    model.train()
    return {"sent": sent, "rel": rel, "evt": evt}


def agreement(df: pl.DataFrame) -> dict:
    """Student vs LLM on one slice (columns from the dataset + pred_*)."""
    out = {"n": df.height}
    if df.height < 3:
        return out
    s, p = df["sentiment"].to_numpy(), df["pred_sent"].to_numpy()
    out["sent_spearman"] = float(spearmanr(s, p).statistic)
    out["sent_pearson"] = float(np.corrcoef(s, p)[0, 1])
    out["sent_mae"] = float(np.mean(np.abs(s - p)))
    for col, pred in (("relevance_id", "pred_rel"), ("event_type_id", "pred_evt")):
        y, yh = df[col].to_numpy(), df[pred].to_numpy()
        key = col.removesuffix("_id")
        out[f"{key}_acc"] = float(np.mean(y == yh))
        out[f"{key}_macro_f1"] = float(f1_score(y, yh, average="macro"))
    return out


def gold_vs_hand(df: pl.DataFrame, label_dir: Path) -> dict:
    """Student and LLM vs hand labels on the gold pairs (news_sample_v1, edgar_8k_sample_v1)."""
    out = {}
    for name in ("news_sample_v1", "edgar_8k_sample_v1"):
        lab, key = label_dir / f"{name}_labels.csv", label_dir / f"{name}_key.csv"
        if not lab.exists() or not key.exists():
            continue
        h = pl.read_csv(lab).join(pl.read_csv(key, columns=["id", "item_id", "ticker"]), on="id")
        g = df.join(h, on=["item_id", "ticker"], how="inner")
        if g.height < 10:
            continue
        hd = g["direction"].replace_strict(DIRECTION).to_numpy()
        r = {"n": g.height}
        for who, col in (("student", "pred_sent"), ("llm", "sentiment")):
            r[f"{who}_sent_spearman_vs_hand"] = float(spearmanr(g[col].to_numpy(), hd).statistic)
        for who, col in (("student", "pred_event_type"), ("llm", "event_type")):
            r[f"{who}_event_acc_vs_hand"] = float((g[col] == g["event_type_right"]).mean())
        for who, col in (("student", "pred_relevance"), ("llm", "relevance")):
            r[f"{who}_relevance_acc_vs_hand"] = float((g[col] == g["relevance_right"]).mean())
        out[name] = r
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default=str(ROOT / "data/distill/v1/dataset.parquet"))
    ap.add_argument("--sources", default="edgar,news_api")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--batch-tokens", type=int, default=0, help="padded-token cap per batch")
    ap.add_argument("--sent-weight", type=float, default=4.0, help="MSE weight vs the two CEs")
    ap.add_argument("--train-splits", default="train")
    ap.add_argument("--limit-train", type=int, default=0, help="subsample training rows (smoke)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        # Windows drivers spill to system RAM instead of raising OOM, and a near-full GPU ran
        # 6x slower (measured: 4x2048 tokens at 7.1 GiB = 1k tok/s vs 2x2048 at 6k tok/s on an
        # 8 GB card); cap PyTorch so an oversized batch fails loudly instead
        torch.cuda.set_per_process_memory_fraction(0.92)
    out = ROOT / "data/distill/runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    sources = sorted(args.sources.split(","))
    # lazy scan: only the selected sources' rows are materialised (8-K text alone is ~1 GB)
    df = (
        pl.scan_parquet(args.dataset)
        .filter(pl.col("source").is_in(sources))
        .collect()
        .with_row_index("_row")
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    # one tokenization of every selected row, cached per dataset file + tokenizer + length
    ds = Path(args.dataset).stat()
    key = f"{ds.st_size}-{int(ds.st_mtime)}-{','.join(sources)}-{args.model}-{args.max_len}"
    cache = (
        Path(args.dataset).parent
        / "tok_cache"
        / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"
    )
    all_ids = tokenize(tok, df["text"].to_list(), args.max_len, cache)
    df = df.drop("text")
    print(f"tokenized {len(all_ids)} rows in {time.time() - t0:.0f}s (cache {cache.name})")
    train_mask = pl.col("split").is_in(args.train_splits.split(","))
    tr = df.filter(train_mask)
    if args.limit_train and tr.height > args.limit_train:
        tr = tr.sample(args.limit_train, seed=args.seed)
    ev = df.filter(~train_mask)
    tr_ids = all_ids.subset(tr["_row"].to_numpy())
    ev_ids = all_ids.subset(ev["_row"].to_numpy())
    del all_ids
    print(f"device {dev}; train {tr.height} rows; eval {ev.height} rows; max_len {args.max_len}")
    pad = tok.pad_token_id

    model = Student(args.model, args.grad_ckpt).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed)
    lengths = tr_ids.lengths
    max_tokens = args.batch_tokens or None
    steps_per_epoch = len(batches(lengths, args.batch_size, False, random.Random(0), max_tokens))
    total_opt_steps = math.ceil(steps_per_epoch * args.epochs / args.grad_accum)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_opt_steps), total_opt_steps)
    y_s = torch.tensor(tr["sentiment"].to_numpy(), dtype=torch.float32)
    y_r = torch.tensor(tr["relevance_id"].to_numpy(), dtype=torch.long)
    y_e = torch.tensor(tr["event_type_id"].to_numpy(), dtype=torch.long)
    ce = nn.CrossEntropyLoss()

    model.train()
    step, done, rows_done, t0, run_loss = 0, 0, 0, time.time(), 0.0
    budget = int(steps_per_epoch * args.epochs)
    while done < budget:
        for rows in batches(lengths, args.batch_size, True, rng, max_tokens):
            if done >= budget:
                break
            x, m = collate(tr_ids, rows, pad, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                s, r, e = model(x, m)
                loss = (
                    args.sent_weight * nn.functional.mse_loss(s.float(), y_s[rows].to(dev))
                    + ce(r.float(), y_r[rows].to(dev))
                    + ce(e.float(), y_e[rows].to(dev))
                )
            (loss / args.grad_accum).backward()
            run_loss += loss.item()
            done += 1
            rows_done += len(rows)
            if done % args.grad_accum == 0 or done == budget:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if done % 200 == 0 or done == budget:
                el = time.time() - t0
                print(
                    f"batch {done}/{budget}  loss {run_loss / 200:.3f}  "
                    f"{rows_done / el:.1f} rows/s  "
                    f"eta {(budget - done) * el / done / 60:.0f} min",
                    flush=True,
                )
                run_loss = 0.0
    train_minutes = (time.time() - t0) / 60

    p = predict(model, ev_ids, pad, dev, args.batch_size * 2, max_tokens and 2 * max_tokens)
    ev = ev.with_columns(
        pl.Series("pred_sent", p["sent"]),
        pl.Series("pred_rel", p["rel"].argmax(1)),
        pl.Series("pred_evt", p["evt"].argmax(1)),
        pl.Series("pred_rel_conf", p["rel"].max(1)),
        pl.Series("pred_evt_conf", p["evt"].max(1)),
    ).with_columns(
        pl.col("pred_rel").map_elements(lambda i: RELEVANCE[i], pl.String).alias("pred_relevance"),
        pl.col("pred_evt")
        .map_elements(lambda i: EVENT_TYPES[i], pl.String)
        .alias("pred_event_type"),
    )
    metrics = {"train_rows": tr.height, "train_minutes": round(train_minutes, 1), "slices": {}}
    for (src, split), g in ev.group_by("source", "split"):
        metrics["slices"][f"{src}/{split}"] = agreement(g)
    metrics["gold_vs_hand"] = gold_vs_hand(
        ev.filter(pl.col("split") == "gold"), ROOT / "data/labeling"
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ev.drop("_row").write_parquet(out / "predictions.parquet")
    if args.save_model:
        torch.save(model.state_dict(), out / "student.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
