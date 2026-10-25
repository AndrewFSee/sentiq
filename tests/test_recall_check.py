"""Paired event-level IC (distillation recall check): IC of two values on identical events."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from newsfeed.eval.clusters import paired_event_ic


def _frames(n_days: int = 120, per_day: int = 8, seed: int = 3):
    rng = np.random.default_rng(seed)
    days = [date(2024, 1, 2) + timedelta(days=k) for k in range(n_days)]
    rows, tg = [], []
    for d in days:
        for j in range(per_day):
            a = rng.normal()
            ret = 0.5 * a + rng.normal()
            rows.append({"ticker": f"T{j}", "date": d, "a": a, "copy": a, "noise": rng.normal()})
            tg.append({"ticker": f"T{j}", "date": d, "entry": "close", "horizon": 2,
                       "label_end": d + timedelta(days=3), "ret_market": ret})  # fmt: skip
    return pl.DataFrame(rows), pl.DataFrame(tg), days


def test_identical_values_give_zero_difference_and_noise_gives_no_ic(cfg):
    ev, tg, days = _frames()
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"bootstrap_reps": 300})})
    cells = [("close", 2, "market")]
    same = paired_event_ic(ev, tg, days, cfg, cells, "a", "copy")[0]
    assert same["n"] == ev.height
    assert same["ic_a"] > 0.3 and abs(same["diff"]) < 1e-12
    assert same["ci_lo_diff"] == same["ci_hi_diff"] == 0.0
    noise = paired_event_ic(ev, tg, days, cfg, cells, "a", "noise")[0]
    assert abs(noise["ic_b"]) < 0.1 and noise["diff"] < -0.2 and noise["p_diff"] < 0.05
    assert noise["ci_lo_diff"] < noise["diff"] < noise["ci_hi_diff"]


def test_holdout_rows_are_dropped(cfg):
    ev, tg, days = _frames()
    late = cfg.eval.model_copy(update={"holdout_start": date(2024, 3, 1), "bootstrap_reps": 50})
    r = paired_event_ic(ev, tg, days, cfg.model_copy(update={"eval": late}),
                        [("close", 2, "market")], "a", "copy")[0]  # fmt: skip
    assert r["n"] == sum(d + timedelta(days=3) < date(2024, 3, 1) for d in ev["date"])
