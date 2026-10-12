"""Evaluation: does news carry incremental, tradable signal?

For every cell of the grid entry x horizon x adjustment (config ``eval``):
  * univariate rank IC (pooled Spearman) of selected features vs the forward return, overall
    and by within-universe size bucket;
  * out-of-sample models (ridge, LightGBM) under purged CV on three nested feature sets:
      M0 controls | M1 controls + non-LLM news (volume, outlets, novelty, LM, FinBERT)
      | M2 = M1 + LLM features;  lift = IC(M2) - IC(M1), paired.
Inference: moving-block bootstrap over decision sessions (block = horizon + 1, as labels of
nearby days overlap) for confidence intervals and two-sided p-values; Benjamini-Hochberg at
``fdr_q`` across *all* tests of the run. The experiments table counts every prompt variant,
feature set and eval configuration ever tried, reported alongside.

Statistical power is the first thing to read in the report: with a dozen tickers and a few
hundred news days, wide intervals are the expected (and honest) outcome.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from newsfeed.config import AppConfig
from newsfeed.eval.cv import pre_holdout_mask, purged_splits
from newsfeed.eval.targets import CONTROL_COLS

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
EVAL_VERSION = 1


# --------------------------------------------------------------------- stats helpers
def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, ranks) / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def block_bootstrap(
    sessions: np.ndarray,
    stat: Any,
    *,
    block: int,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular block bootstrap over *calendar* session indices; ``stat(idx) -> float``.

    ``sessions`` are trading-session indices (consecutive integers on the market calendar), so
    a block of ``block`` sessions spans exactly the overlap of h-day labels -- not ``block``
    sampled days, which on a sparse news sample would span months. Blocks wrap around
    (circular), so every session is equally likely to be drawn; the plain moving-block version
    under-samples the ends, which with few blocks makes intervals too narrow and off-centre.
    Each resample draws ceil(span / block) blocks; rows of the drawn sessions are pooled.
    """
    lo = int(sessions.min())
    span = int(sessions.max()) - lo + 1
    offset = sessions.astype(np.int64) - lo
    rows_at = [np.flatnonzero(offset == k) for k in range(span)]
    n_blocks = max(1, int(np.ceil(span / block)))
    out = np.empty(reps)
    for r in range(reps):
        starts = rng.integers(0, span, n_blocks)
        picked = ((starts[:, None] + np.arange(block)[None, :]) % span).ravel()
        idx = np.concatenate([rows_at[k] for k in picked])
        out[r] = stat(idx) if len(idx) else np.nan
    return out


def boot_summary(point: float, boots: np.ndarray) -> dict[str, float]:
    b = boots[~np.isnan(boots)]
    if np.isnan(point) or len(b) < 20:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "p": float("nan")}
    p = 2 * min(np.mean(b <= 0), np.mean(b >= 0))
    return {
        "ci_lo": float(np.quantile(b, 0.025)),
        "ci_hi": float(np.quantile(b, 0.975)),
        "p": float(min(1.0, p)),
    }


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of discoveries (NaN p-values never are)."""
    ok = ~np.isnan(p)
    out = np.zeros(len(p), dtype=bool)
    pv = p[ok]
    if not len(pv):
        return out
    order = np.argsort(pv)
    thresh = q * np.arange(1, len(pv) + 1) / len(pv)
    passed = pv[order] <= thresh
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    disc = np.zeros(len(pv), dtype=bool)
    disc[order[:k]] = True
    out[np.flatnonzero(ok)] = disc
    return out


# --------------------------------------------------------------------- dataset
@dataclass
class EvalInputs:
    features: pl.DataFrame  # ticker, as_of, <window>_<feature>...
    controls: pl.DataFrame  # ticker, date, ctl_*
    targets: pl.DataFrame  # ticker, date, entry, horizon, label_end, ret_raw/market/sector
    caps: pl.DataFrame  # ticker, date, mcap, size_bucket
    sessions: list[date]  # all trading sessions (benchmark calendar)


def feature_blocks(cols: list[str]) -> tuple[list[str], list[str]]:
    """(non-LLM news features, LLM features) among feature columns."""
    llm = [c for c in cols if "_llm_" in c]
    base = [c for c in cols if c not in llm and c not in ("ticker", "as_of", "date")]
    return base, llm


def assemble(inp: EvalInputs, cfg: AppConfig) -> pl.DataFrame:
    """Base rows (one per ticker-day in the sample) with features, controls and size."""
    f = inp.features.with_columns(
        pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("date")
    )
    if cfg.eval.sample == "news":
        col = f"{cfg.eval.sample_window}_n_stories"
        f = f.filter(pl.col(col).fill_null(0) > 0)
    if cfg.eval.sample_event_types:
        cols = [
            f"{cfg.eval.sample_window}_llm_evt_{t.replace('&', '')}"
            for t in cfg.eval.sample_event_types
        ]
        missing = [c for c in cols if c not in f.columns]
        if missing:
            raise ValueError(f"unknown event types for sampling: {missing}")
        f = f.filter(pl.sum_horizontal([pl.col(c).fill_null(0) for c in cols]) > 0)
    out = f.join(inp.controls, on=["ticker", "date"], how="left").join(
        inp.caps, on=["ticker", "date"], how="left"
    )
    if cfg.eval.sample_size_buckets:
        out = out.filter(pl.col("size_bucket").is_in(cfg.eval.sample_size_buckets))
    return out


# --------------------------------------------------------------------- models
def _fit_predict(kind: str, cfg: AppConfig, Xtr, ytr, Xte) -> np.ndarray:
    lo, hi = np.nanquantile(ytr, [0.01, 0.99])
    ytr = np.clip(ytr, lo, hi)  # winsorise the training target only
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        Xtr0, Xte0 = np.nan_to_num(Xtr), np.nan_to_num(Xte)
        sc = StandardScaler().fit(Xtr0)
        m = Ridge(alpha=cfg.eval.models.ridge_alpha).fit(sc.transform(Xtr0), ytr)
        return m.predict(sc.transform(Xte0))
    from lightgbm import LGBMRegressor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LGBMRegressor(**cfg.eval.models.lgbm, verbose=-1).fit(Xtr, ytr)
        return m.predict(Xte)


def oos_predictions(
    kind: str, cfg: AppConfig, X: np.ndarray, y: np.ndarray, folds: list
) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for fold in folds:
        if len(fold.train) < 20:
            continue
        pred[fold.test] = _fit_predict(kind, cfg, X[fold.train], y[fold.train], X[fold.test])
    return pred


# --------------------------------------------------------------------- run
def run_eval(
    inp: EvalInputs,
    cfg: AppConfig,
    *,
    univariate: list[str],
    use_holdout: bool = False,
) -> dict[str, Any]:
    ec = cfg.eval
    rng = np.random.default_rng(ec.seed)
    base = assemble(inp, cfg)
    news_cols, llm_cols = feature_blocks(inp.features.columns)
    sets = {
        "M0": list(CONTROL_COLS),
        "M1": list(CONTROL_COLS) + news_cols,
        "M2": list(CONTROL_COLS) + news_cols + llm_cols,
    }
    session_idx = {d: i for i, d in enumerate(inp.sessions)}
    uni_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    sample_info: dict[str, Any] = {}

    for entry in ec.entries:
        for h in ec.horizons:
            t = inp.targets.filter((pl.col("entry") == entry) & (pl.col("horizon") == h))
            df = base.join(t, on=["ticker", "date"], how="inner").sort("date", "ticker")
            if df.height == 0:
                continue
            dec = df["date"].to_numpy().astype("datetime64[D]")
            end = df["label_end"].to_numpy().astype("datetime64[D]")
            pre = pre_holdout_mask(dec, end, ec.holdout_start)
            if use_holdout:
                train_rows, test_rows = (
                    np.flatnonzero(pre),
                    np.flatnonzero(dec >= np.datetime64(ec.holdout_start)),
                )
            else:
                df = df.filter(pl.Series(pre))
                train_rows = test_rows = None
            s = np.array([session_idx.get(d, -1) for d in df["date"].to_list()])
            e = np.array([session_idx.get(d, -1) for d in df["label_end"].to_list()])
            block = h + 1 if entry == "close" else h + 2
            sample_info[f"{entry}/{h}"] = {
                "rows": df.height,
                "sessions": len(np.unique(s)),
                "tickers": int(df["ticker"].n_unique()),
                "first": str(df["date"].min()),
                "last": str(df["date"].max()),
            }
            eval_rows = test_rows if use_holdout else np.arange(df.height)
            for adj in ec.adjustments:
                y = df[f"ret_{adj}"].to_numpy().astype(float)
                cell = {"entry": entry, "horizon": h, "adjustment": adj}
                # univariate rank IC on the evaluation rows
                for feat in univariate:
                    if feat not in df.columns:
                        continue
                    x = df[feat].to_numpy().astype(float)
                    rows = eval_rows
                    ic = spearman(x[rows], y[rows])
                    boots = block_bootstrap(
                        s[rows],
                        lambda idx, x=x[rows], yy=y[rows]: spearman(x[idx], yy[idx]),
                        block=block,
                        reps=ec.bootstrap_reps,
                        rng=rng,
                    )
                    n = int(np.sum(~(np.isnan(x[rows]) | np.isnan(y[rows]))))
                    uni_rows.append(
                        {**cell, "feature": feat, "n": n, "ic": ic, **boot_summary(ic, boots)}
                    )
                    for bucket in ("small", "mid", "large", "unknown"):
                        bm = df["size_bucket"].to_numpy()[rows] == bucket
                        if bm.sum() >= 5:
                            bucket_rows.append(
                                {
                                    **cell,
                                    "feature": feat,
                                    "bucket": bucket,
                                    "n": int(bm.sum()),
                                    "ic": spearman(x[rows][bm], y[rows][bm]),
                                }
                            )
                # out-of-sample models with purged CV (or train-on-pre / test-on-holdout)
                if use_holdout:
                    from newsfeed.eval.cv import Fold

                    folds = [Fold(train_rows[e[train_rows] < s[test_rows].min()], test_rows)]
                else:
                    folds = purged_splits(s, e, ec.cv)
                if not folds:
                    continue
                covered = np.unique(np.concatenate([f.test for f in folds]))
                for kind in ("ridge", "lgbm"):
                    preds = {
                        m: oos_predictions(
                            kind, cfg, df.select(cols).to_numpy().astype(float), y, folds
                        )
                        for m, cols in sets.items()
                    }
                    ics = {m: spearman(p[covered], y[covered]) for m, p in preds.items()}
                    lift = ics["M2"] - ics["M1"]
                    yc, p1, p2 = y[covered], preds["M1"][covered], preds["M2"][covered]
                    boots = block_bootstrap(
                        s[covered],
                        lambda idx, yc=yc, p1=p1, p2=p2: (
                            spearman(p2[idx], yc[idx]) - spearman(p1[idx], yc[idx])
                        ),
                        block=block,
                        reps=ec.bootstrap_reps,
                        rng=rng,
                    )
                    model_rows.append(
                        {
                            **cell,
                            "model": kind,
                            "n_oos": len(covered),
                            "folds": len(folds),
                            "ic_M0": ics["M0"],
                            "ic_M1": ics["M1"],
                            "ic_M2": ics["M2"],
                            "lift": lift,
                            **boot_summary(lift, boots),
                        }
                    )

    uni = pl.DataFrame(uni_rows) if uni_rows else pl.DataFrame()
    models = pl.DataFrame(model_rows) if model_rows else pl.DataFrame()
    # Benjamini-Hochberg jointly over every test in this run
    p_all = np.concatenate(
        [
            uni["p"].to_numpy() if uni.height else np.array([]),
            models["p"].to_numpy() if models.height else np.array([]),
        ]
    ).astype(float)
    disc = benjamini_hochberg(p_all, ec.fdr_q)
    if uni.height:
        uni = uni.with_columns(pl.Series("bh_discovery", disc[: uni.height]))
    if models.height:
        models = models.with_columns(pl.Series("bh_discovery", disc[uni.height :]))
    return {
        "univariate": uni,
        "models": models,
        "buckets": pl.DataFrame(bucket_rows) if bucket_rows else pl.DataFrame(),
        "sample": sample_info,
        "n_tests": int(np.sum(~np.isnan(p_all))),
        "feature_blocks": {"news": news_cols, "llm": llm_cols, "controls": list(CONTROL_COLS)},
        "holdout": use_holdout,
    }


def eval_config_id(cfg: AppConfig, feature_set_id: str, univariate: list[str]) -> str:
    blob = json.dumps(
        [EVAL_VERSION, cfg.eval.model_dump(mode="json"), feature_set_id, univariate],
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


DEFAULT_UNIVARIATE = [
    "1d_llm_sent_decay",
    "1d_llm_sent_cwmean",
    "5d_llm_sent_decay",
    "1d_lm_tone_decay",
    "1d_lm_tone_mean",
    "5d_lm_tone_decay",
    "1d_volume_z",
    "1d_emb_novelty_mean",
    "1d_llm_sent_x_novelty",
]


def add_derived(feats: pl.DataFrame) -> pl.DataFrame:
    """Derived features, computed row-wise from same-row inputs (no extra information):
    ``1d_llm_sent_x_novelty`` = 1d LLM sentiment (cw mean) x 1d embedding novelty
    (pre-registered 2026-10-12: novel news should move prices more). LLM block (``_llm_``)."""
    a, b = "1d_llm_sent_cwmean", "1d_emb_novelty_mean"
    if a in feats.columns and b in feats.columns:
        feats = feats.with_columns((pl.col(a) * pl.col(b)).alias("1d_llm_sent_x_novelty"))
    return feats


# --------------------------------------------------------------------- rendering
def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:+.3f}"
    return str(v)


def render_markdown(result: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = [
        f"# Eval report {meta['run_id']}",
        "",
        f"- feature set `{meta['feature_set_id']}`, eval config `{meta['eval_id']}`, "
        f"LLM `{meta['llm_model']}` prompt `{meta['prompt_hash']}`",
        f"- {'**HOLDOUT evaluation**' if result['holdout'] else 'pre-holdout data only'} "
        f"(holdout starts {meta['holdout_start']})",
        f"- tests in this run: {result['n_tests']}; Benjamini-Hochberg q = {meta['fdr_q']}",
        f"- experiments tried so far: {meta['experiments']}",
        "",
        "## Sample",
        "",
        "| entry/horizon | rows | sessions | tickers | first | last |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in result["sample"].items():
        lines.append(
            f"| {k} | {v['rows']} | {v['sessions']} | {v['tickers']} | {v['first']} | {v['last']} |"
        )
    lines += [
        "",
        "> Power check: confidence intervals below are block-bootstrapped over decision days."
        " With this few independent days, intervals spanning zero are the expected outcome;"
        " read effect sizes and consistency, not single cells.",
        "",
    ]
    m = result["models"]
    if m.height:
        lines += [
            "## Incremental lift of LLM features (OOS rank IC: M2 - M1)",
            "",
            "| entry | h | adj | model | n | IC M0 | IC M1 | IC M2 | lift | 95% CI | p | BH |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in m.sort("entry", "horizon", "adjustment", "model").to_dicts():
            lines.append(
                f"| {r['entry']} | {r['horizon']} | {r['adjustment']} | {r['model']} | "
                f"{r['n_oos']} | {_fmt(r['ic_M0'])} | {_fmt(r['ic_M1'])} | {_fmt(r['ic_M2'])} | "
                f"{_fmt(r['lift'])} | [{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | "
                f"{_fmt(r['p'])} | {'yes' if r['bh_discovery'] else ''} |"
            )
        lines.append("")
    u = result["univariate"]
    if u.height:
        lines += ["## Univariate rank IC (market-adjusted targets)", ""]
        mk = u.filter(pl.col("adjustment") == "market")
        cells = mk.select("entry", "horizon").unique().sort("entry", "horizon").rows()
        lines.append("| feature | " + " | ".join(f"{e}/{h}d" for e, h in cells) + " |")
        lines.append("|---|" + "---|" * len(cells))
        for feat in mk["feature"].unique(maintain_order=True).to_list():
            vals = []
            for e, h in cells:
                r = mk.filter(
                    (pl.col("feature") == feat) & (pl.col("entry") == e) & (pl.col("horizon") == h)
                ).to_dicts()
                if r:
                    star = "*" if r[0]["bh_discovery"] else ""
                    vals.append(f"{_fmt(r[0]['ic'])}{star}")
                else:
                    vals.append("-")
            lines.append(f"| {feat} | " + " | ".join(vals) + " |")
        lines += ["", "`*` = Benjamini-Hochberg discovery across all tests of this run.", ""]
    disc_u = u.filter(pl.col("bh_discovery")) if u.height else u
    disc_m = m.filter(pl.col("bh_discovery")) if m.height else m
    lines += ["## Discoveries after multiple-testing correction", ""]
    if (disc_u.height if u.height else 0) + (disc_m.height if m.height else 0) == 0:
        lines.append("None at the configured FDR level.")
    else:
        for r in disc_m.to_dicts() if m.height else []:
            lines.append(
                f"- lift {r['model']} {r['entry']}/{r['horizon']}d/{r['adjustment']}: "
                f"{_fmt(r['lift'])} (p {_fmt(r['p'])})"
            )
        for r in disc_u.to_dicts() if u.height else []:
            lines.append(
                f"- IC {r['feature']} {r['entry']}/{r['horizon']}d/{r['adjustment']}: "
                f"{_fmt(r['ic'])} (p {_fmt(r['p'])})"
            )
    return "\n".join(lines) + "\n"


def decision_dates(features: pl.DataFrame) -> list[date]:
    return sorted(
        set(features["as_of"].dt.convert_time_zone("America/New_York").dt.date().to_list())
    )
