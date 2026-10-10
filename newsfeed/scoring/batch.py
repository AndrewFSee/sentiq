"""Message Batches API for backfill scoring (50% cheaper; results within 24h).

Submission records every batch and its request -> target mapping in the ops database
*immediately* after the API accepts it, so a crash never loses track of paid work, and target
selection skips anything in an uncollected batch (no double submission). Collection is
idempotent: a batch is marked collected only after its rows are written. Results arrive in any
order and are matched by ``custom_id``. Errored/expired/canceled requests are logged to the
usage ledger but produce no score row, so the next run re-selects them.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from newsfeed.scoring.client import parse_message
from newsfeed.scoring.runner import Prepared, ScoringJob
from newsfeed.scoring.targets import Target
from newsfeed.timeutil import utcnow

log = logging.getLogger(__name__)


def submit(job: ScoringJob, targets: Sequence[Target]) -> list[str]:
    if job.client is None:
        raise RuntimeError("no LLM client configured")
    job.register_experiment()
    prepared = job.prepare(targets)
    size = job.cfg.scoring.batch_max_requests
    batch_ids = []
    for i in range(0, len(prepared), size):
        chunk = prepared[i : i + size]
        requests = [{"custom_id": p.target.custom_id, "params": p.params} for p in chunk]
        if len({r["custom_id"] for r in requests}) != len(requests):
            raise ValueError("duplicate custom_id within a batch")
        batch_id = job.client.batch_create(requests)
        log.warning("submitted batch %s (%d requests)", batch_id, len(chunk))  # paid: log loudly
        _record_submission(job, batch_id, chunk)
        batch_ids.append(batch_id)
    return batch_ids


def _record_submission(job: ScoringJob, batch_id: str, chunk: Sequence[Prepared]) -> None:
    with job.store.ops() as con:
        con.execute(
            "INSERT INTO llm_batches VALUES (?, ?, ?, ?, ?, ?, NULL, 'in_progress', FALSE)",
            [batch_id, job.run_id, job.model, job.prompt_hash, len(chunk), utcnow()],
        )
        con.executemany(
            "INSERT INTO llm_batch_requests VALUES (?, ?, ?, ?)",
            [
                [
                    batch_id,
                    p.target.custom_id,
                    p.target.cache_key,
                    _target_record(p),
                ]
                for p in chunk
            ],
        )


def _target_record(p: Prepared) -> str:
    return json.dumps(
        {
            "target": p.target.to_json(),
            "user_text_hash": p.user_text_hash,
            "mask_residuals": p.mask_residuals,
        }
    )


def pending_batches(job: ScoringJob) -> list[str]:
    df = job.store.ops_query(
        "SELECT batch_id FROM llm_batches WHERE NOT collected ORDER BY submitted_at"
    )
    return df["batch_id"].to_list()


def collect(
    job: ScoringJob,
    *,
    wait: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    max_polls: int | None = None,
) -> dict[str, Any]:
    """Poll uncollected batches; ingest the ones that have ended. With ``wait``, keep polling
    every ``batch_poll_seconds`` until none remain (or ``max_polls`` rounds).

    A crash between writing rows and marking the batch collected means the batch is collected
    again next time: duplicate score rows are harmless (the view keeps one per cache_key), but
    the usage ledger would double-count that batch's cost."""

    if job.client is None:
        raise RuntimeError("no LLM client configured")
    summary: dict[str, Any] = {"collected": [], "still_running": []}
    polls = 0
    while True:
        running = []
        for batch_id in pending_batches(job):
            status = job.client.batch_status(batch_id)
            if status != "ended":
                running.append(batch_id)
                continue
            mapping = {
                r["custom_id"]: json.loads(r["target"])
                for r in job.store.ops_query(
                    "SELECT custom_id, target FROM llm_batch_requests WHERE batch_id = ?",
                    [batch_id],
                ).to_dicts()
            }
            rows, usage = [], []
            for res in job.client.batch_results(batch_id):
                rec = mapping.get(res.custom_id)
                if rec is None:
                    log.error("batch %s returned unknown custom_id %s", batch_id, res.custom_id)
                    continue
                target = Target.from_json(rec["target"])
                p = Prepared(target, {}, "", rec["user_text_hash"], rec["mask_residuals"])
                if res.type == "succeeded":
                    row, u = job.record(
                        p, parse_message(res.message), mode="batch", batch_id=batch_id
                    )
                    rows.append(row)
                    usage.append(u)
                else:
                    job.stats.add(res.type, {}, 0.0)
                    usage.append(
                        job.usage_row(
                            target,
                            mode="batch",
                            status=res.type,
                            usage={},
                            cost=0.0,
                            batch_id=batch_id,
                            error=res.error,
                        )
                    )
            job.store.write_scores(rows, job.run_id)
            job.log_usage(usage)
            with job.store.ops() as con:
                con.execute(
                    "UPDATE llm_batches SET collected = TRUE, ended_at = ?, "
                    "processing_status = 'ended' WHERE batch_id = ?",
                    [utcnow(), batch_id],
                )
            summary["collected"].append(batch_id)
        summary["still_running"] = running
        polls += 1
        if not wait or not running or (max_polls is not None and polls >= max_polls):
            return summary
        sleep(job.cfg.scoring.batch_poll_seconds)
