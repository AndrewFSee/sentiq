"""Test 14 follow-up (EXPLORATORY, pre-specified 2026-11-22 in CLAUDE.md): is the daily 8-K tone
-> next-day SPY result a composition effect (earnings-heavy days), stable across halves, and
there under masked scores? Same construction as `newsfeed media-tone`.

    uv run python scripts/media_tone_explore.py --config config/variants/media_tone.yaml
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import typer

from newsfeed import cli
from newsfeed.config import load_config
from newsfeed.eval.media import CONTROLS, market_daily, market_table, session_tone
from newsfeed.eval.prices import price_panel
from newsfeed.eval.report import partial_rank_corr
from newsfeed.timeutil import utcnow


def main(config: Path = typer.Option(...), reps: int = 10_000) -> None:  # noqa: B008
    cfg = load_config(config)
    hold = cfg.eval.holdout_start

    def body(ctx: cli.Context, run_id: str, stats: dict) -> None:
        ev, grid = cli._llm_events(ctx, cfg, "edgar", [], masked=False)
        ev = ev.with_columns((pl.col("event_type") == "earnings").cast(pl.Float64).alias("earn"))
        docs = cli._session_docs(ev, grid, "sentiment")
        share = cli._session_docs(ev, grid, "earn").with_columns(
            (pl.col("earn") > 0).cast(pl.Float64).alias("earn")
        )
        non = cli._session_docs(ev.filter(pl.col("event_type") != "earnings"), grid, "sentiment")
        e_un = ev.filter(pl.col("event_type") == "earnings")
        em, gm = cli._llm_events(ctx, cfg, "edgar", ["earnings"], masked=True)
        # identical documents: earnings founders with both a masked and an unmasked score
        both = e_un.select("item_id").unique().join(em.select("item_id").unique(), on="item_id")
        e_un_b = cli._session_docs(e_un.join(both, on="item_id"), grid, "sentiment")
        e_m_b = cli._session_docs(em.join(both, on="item_id"), gm, "sentiment")

        bars = price_panel(
            cli._price_caches(cfg)["tradier"],
            [cfg.eval.market_etf],
            docs["date"].min() - timedelta(days=60),
            utcnow().date(),
        ).filter(pl.col("symbol") == cfg.eval.market_etf)
        mkt = market_table(market_daily(bars), hold).filter(pl.col("date") < hold)
        sessions = mkt["date"].to_list()
        sidx = np.arange(len(sessions))
        t_all = session_tone(docs, sessions, value="sentiment")
        t_non = session_tone(non, sessions, value="sentiment")
        sh = session_tone(share, sessions, value="earn")  # tone = earnings share, n = docs
        t_eu = session_tone(e_un_b, sessions, value="sentiment")
        t_em = session_tone(e_m_b, sessions, value="sentiment")
        rng = np.random.default_rng(cfg.eval.seed)

        def a(frame: pl.DataFrame, c: str) -> np.ndarray:
            return frame[c].cast(pl.Float64).fill_null(np.nan).to_numpy()

        base = np.column_stack([a(mkt, c) for c in CONTROLS])
        y = a(mkt, "fwd1")
        comp = np.column_stack([base, a(sh, "tone"), np.log(np.maximum(a(sh, "n"), 1))])
        yrs = np.array([d.year for d in sessions])
        early, late = yrs <= 2022, yrs >= 2023
        # masked vs unmasked on identical sessions
        same = ~(np.isnan(a(t_eu, "tone")) | np.isnan(a(t_em, "tone")))
        plan = [
            ("8-K all (= test 14 (e))", a(t_all, "tone_dm"), base, None),
            ("(1) + earnings share + log docs", a(t_all, "tone_dm"), comp, None),
            ("(2) non-earnings 8-Ks", a(t_non, "tone_dm"), base, None),
            ("(2) non-earnings + composition", a(t_non, "tone_dm"), comp, None),
            ("(3) 8-K all 2019-2022", a(t_all, "tone_dm"), base, early),
            ("(3) 8-K all 2023-2026", a(t_all, "tone_dm"), base, late),
            ("(4) earnings raw, unmasked", a(t_eu, "tone"), base, same),
            ("(4) earnings raw, masked", a(t_em, "tone"), base, same),
            ("earnings share alone", a(sh, "tone"), base, None),
        ]
        rows = []
        for label, x, ctl, m in plan:
            m = np.ones(len(x), bool) if m is None else m
            r = partial_rank_corr(x[m], y[m], ctl[m], sidx[m], block=2, reps=reps, rng=rng)
            rows.append({"test": label, **r})
            typer.echo(
                f"{label:36s} n {r['n']:5d} raw {r['ic']:+.3f} partial {r['partial_ic']:+.3f} "
                f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] p {r['p']:.4f}"
            )
        res = pl.DataFrame(rows)
        out = cfg.paths.data_dir / "reports" / f"media-tone-explore-{run_id}.csv"
        res.write_csv(out)
        stats.update(report=str(out))
        cli._record_experiment(ctx, "media_tone_explore", f"media-tone-explore:{run_id}", run_id,
                               {"reps": reps})  # fmt: skip

    cli._run("media-tone-explore", cfg, {"reps": reps}, body)


if __name__ == "__main__":
    typer.run(main)
