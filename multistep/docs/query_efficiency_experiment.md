# Experiment: expected candidate reduction per expected query

The baseline is saved locally at `252cab5b8c8c909ee3460b58887c61061b0a519b`
on `codex/vertex-support-cache`. A full Git bundle is stored on Mac and Bench i7.
The experimental branch is `codex/expected-reduction-per-query-depth1-depth2`.
This is a query-selection experiment: selected queries and the number of actual
questions may change. Fixed-depth planning remains the default.

For current candidate count C, each permitted root query is evaluated at
horizons h=1 and h=2. Its score is

    (C - E[terminal candidate count]) / E[actually executed queries]

The denominator is a ratio of expectations, rather than an expectation of
per-path ratios. The root question costs one. For horizon two, each positive
answer branch costs one further question precisely when the existing depth-one
planner returns a query. Singleton children and children with no available
query cost zero further questions. A hypothetical path does not stop merely
because one candidate has 99% volume confidence. The actual solve loop keeps
the existing 99% volume confidence rule and 100-question guard.

Every root query is considered, not just the old minimum-terminal-candidate
query for each horizon. At horizon one only ratio queries are allowed; at
horizon two the root uses Grid5 plus ratios and the last level uses ratio
queries. Child costs at depth one are identical for all available questions,
so the existing child minimum-expectation choice is appropriate here. This
equivalence does not justify the same method at horizon three; the experiment
rejects horizons above two.

The deterministic ordering is descending score, smaller expected query count,
smaller horizon, and query identity (goal a, goal b, ratio value). Runtime
measurements are diagnostics and never enter query selection. Child results
are observed after the existing evaluation, without repeating geometry.
Invalid positive-probability branches produce explicit experimental errors;
they do not improve the score by inventing empty candidate sets.

`benchmark_exact_end_to_end.py --query-strategy reduction_per_query --depth 2`
opts into the new strategy. `--query-strategy fixed` is the unchanged default.
The complete chosen policy for each nonempty horizon is serialized, including
root/leaf probabilities, terminal counts, expected questions, gain and rate.

The experiment uses 100 fixed fresh problems with five objectives and five
alternatives, seed 2026101301. All three strategies run on the same inputs with
eight maximum workers, exact volume calculations, native thread limits of one,
and no per-problem time limit. Their order rotates by problem; every condition
starts in a fresh process. All planning, including both adaptive horizons,
pool startup/shutdown and observation overhead count toward solve time.
Errors and unanswered cases are retained, with no automatic error retries.

Artifacts and continuation instructions are under:

    outputs/benchmarks/bench-i7/20261013-query-efficiency-depth12-problems100-grid5-ratios-workers8

The Python source is frozen as 91 files, fingerprint
`7d52e67f234d1ff86aba7ffb9f48d70420b8c743d2da5c0bbc78900b25f8638f`.
256 Multi-Step tests and six One-Step tests pass. Three real Bench smoke runs
on a prior case outside the cohort pass; the fixed-depth2 query trajectory,
expectations and candidate states are bit-identical to its earlier benchmark.
The independent review is stored with the artifacts. Its conclusion concerns
implementation consistency, not proof of the best policy over an unlimited
future or exact symbolic geometry.

The durable controller and Mac mirror continue independently of the AI run.
`series-state.json` preserves attempts and accepted result hashes; completed
cases are skipped on recovery. Question checkpoints document progress but do
not restore recursion inside an interrupted case. Verify live process
identities before resuming; do not run a second solver alongside the controller.
Final conclusions belong in `comp.md` after all 300 outcomes are checked.

## Final Bench i7 results (13 October 2026)

All 300 prescribed executions are finished, with one preserved attempt per job and no automatic retries. Both fixed Depth 2 and adaptive rate solved all 100 problems correctly; neither had an incorrect selection, an unresolved case or a computation failure. Fixed Depth 1 solved22 correctly, stopped with an exhausted query set in77, and left case 61 unresolved after100 questions. These outcomes keep the original denominator100 per condition.

| Condition, all 100 paired successes | Total seconds | Mean seconds | Median seconds | P90 seconds | Total questions | Mean questions | Median questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_depth2 | 719.625 | 7.196 | 5.471 | 15.577 | 1010 | 10.10 | 8.0 |
| adaptive_rate | 621.498 | 6.215 | 5.213 | 12.076 | 928 | 9.28 | 7.0 |

The adaptive strategy used 13.6358% less total solve time and 8.1188% fewer actual questions. It was faster on 53 cases and slower on 47; it asked fewer questions on 47, the same number on 19, and more on 34. These are improvements in the aggregate, not guarantees for each problem. On case 86, Depth 2 used30 questions /16.934s, versus44 /22.787s for adaptive. Its largest additional time was case 73:21 /10.430s versus32 /18.601s. Timing is one observation per problem/strategy, with rotated order; tiny differences are not established timing effects.

Fixed Depth 2 ended with a single candidate in 41 cases and by volume confidence in 59. Adaptive ended with a single candidate in 90 and by volume confidence in 10. Both selected the exact-rational target utility winner in every solved case. The100-query Depth 1 case retained 3 candidates with a maximum volume share of 74.78%; its partial answers do not count as a solved result.

Adaptive selected horizon1 for 503 actual questions and horizon2 for 425. It nevertheless evaluated both horizons before every decision. Horizon1 evaluation totaled43.474s and horizon2 evaluation 559.893s; both together 603.367s, about97.08% of the 621.498s solve timer. These are sequential parent evaluation times, not summed worker CPU times. Each horizon time includes evaluating all its root queries. Selecting a depth does not make evaluation of the other depth free.

All inputs, full configurations, results, actual answer/state journals, worker/process identities, errors, summaries and recovery instructions are saved on Mac and Bench. The independent source/Smoke reviews,262 passing unit tests and separate raw-result analysis support consistency of this implementation and benchmark. Geometry was not independently recomputed symbolically. The experimental source remains separate from the fixed default. There was no push, no experiment commit and no merge into the baseline.

`comp.md` and `comp.json` contain all 100 case pairs for each available baseline, all 300 dispositions, error counts, distribution statistics, relative comparisons and planning diagnostics. `series-summary.json` is independently reaggregated from raw results. `final-independent-review.json` and `final-backup-receipt.json` record the final review and verified backup.
