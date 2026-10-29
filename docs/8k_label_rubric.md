# 8-K event-type rubric — clarifications (accepted 2026-10-29)

Base definitions are those of the scoring prompt `newsfeed/scoring/prompts/score_v1.md` (used by
all existing scores and frozen for H1/H2). These rules settle the edge cases found in the 8-K
hand-labelling check (`edgar_8k_sample_v1`, 37 of 100 labels marked unsure). They apply to any
future labelling and to any new prompt version; they do not change existing scores, H1 or H2.

| # | Rule |
|---|---|
| R1 | **Results present → `earnings`**, whatever the outlook does (raised, cut, reaffirmed). |
| R2 | **Pre-announcements are results.** Any figure for a period that has *already ended* (preliminary quarter, monthly sales, "Q2 will come in below guidance" said after Q2 closed) → `has_results = yes`, `earnings`. Only statements about periods not yet ended are outlook. |
| R3 | **Guidance = a new or changed forward-looking outlook** for a period not yet ended: initiated, raised, lowered, narrowed, withdrawn/suspended, or new multi-year targets (investor day). Includes quantitative operating guidance (capex, production, occupancy, volumes), not only revenue/EPS. |
| R4 | **Reaffirmed, unchanged guidance with nothing new → `other`**, `has_outlook = no`. |
| R5 | **Fixed priority instead of "dominant event"** when an 8-K bundles several actions: results → `earnings`; else any new/changed/withdrawn outlook → `guidance`; else the dominant remaining event. |
| R6 | **Label only the visible text** (as the model saw it, possibly truncated). An outlook in a cut-off section does not exist. |

Flags: `has_results` = actual results for a completed period, including preliminary or
pre-announced ones (R2). `has_outlook` = a new, changed or withdrawn outlook (R3); a plain
reaffirmation is `no` (R4).

Examples from the sample (by id): R2 e047, e057, e067, e082 · R3 e044, e056, e060, e097 ·
R4 e020, e026, e027, e033, e035, e045, e048, e065, e074, e075, e100 · R5 e032, e049, e077, e085 ·
R6 e022, e098.
