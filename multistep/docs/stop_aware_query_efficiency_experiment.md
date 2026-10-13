# Stop-aware expected reduction per expected query

The checkpoint is commit `1e24d32`, tagged
`checkpoint/query-efficiency-depth12-2026-10-13`. The new implementation stays
on `codex/query-efficiency-stop-aware99` without another commit or push.

The optional strategy `reduction_per_query_stop_aware` enables
`OptimizedMultistepConfig.query_efficiency_stop_aware`. Fixed-depth planning and
the existing `reduction_per_query` strategy retain their original behavior.

For an unsolved state with actual candidate count C, every permitted root
query/horizon pair at horizons one and two is ranked by

    (C - E[terminal loss]) / E[questions executed within the plan]

Terminal loss is one when the existing singleton or volume-confidence
termination criterion is satisfied, and otherwise the actual candidate count.
The confidence threshold is unchanged at 99%. Every positive-probability
hypothetical state, including the last leaf, is checked. A finished state is
absorbing and incurs no further query. At horizon two, each unfinished child
with a valid continuation costs one further query weighted by its probability.
Its ratio-only continuation minimizes the same expected terminal loss.
An unfinished child with no available query retains its candidate count as
loss; query exhaustion is not represented as successful completion.

The terminal loss is a planning objective, not an actual candidate count.
`QueryEvaluation.expected_value` and `expected_terminal_candidates` retain
their genuine expected candidate-count meaning. `expected_terminal_loss` is
stored separately together with branch stop reasons, volume shares, costs,
and the complete selected continuation policies. Geometry and actual
termination use the existing functions, tolerances and exact-volume method.

This is a finite-horizon heuristic. Incorporating termination does not prove
globally minimal questions, guarantee improvements on every answer path, or
change the mathematical requirement for the actual 99% stop.

## Matched experiment

The same fixed input bytes as the previous 100-problem experiment are used:
SHA256 `a78c365f182479cc4092a17ee7bfcc5416dfe2da6ae4acb3e9722a5ed00f4857`.
All 100 problems are included under `fixed_depth2`, `adaptive_rate`, and
`adaptive_stop_aware`. Their order rotates by problem, each condition starts
in a fresh process, and execution remains sequential across jobs.

Every job uses five goals, five alternatives, Grid5 plus ratios at horizon two,
ratio-only queries at horizon one, eight maximum workers, `forkserver`, exact
volume, six native thread limits of one, actual 99% termination and the
existing 100-question guard. No time limit, sampled approximation, shortlist,
changed grid, automatic error retry, or outcome-based case selection is used.
Both adaptive horizons and the added candidate-volume computations count
toward elapsed solve time. Overlapping worker durations are not added to
elapsed solve time. Failed and unresolved cases remain in the quality
denominator; paired runtime comparisons identify their successful cohort.

The Python source and full protocol are frozen before launch. The durable
controller preserves attempts, accepted hashes, raw result files, question
journals, worker progress and Linux process identities. Mac mirroring runs
independently of the AI session. Restart only after verifying that the
controller and its solver group are no longer alive; completed cases are
verified and skipped, while interrupted cases restart at a case boundary.

Artifacts:
`outputs/benchmarks/bench-i7/20261013-query-efficiency-stop-aware99-problems100-grid5-ratios-workers8`.
Original benchmark data and provenance remain unchanged. Full source, patch,
new files, inputs, raw data and final reports are preserved on Mac and Bench.

Before launch, 271 Multi-Step and six One-Step tests passed, including 15 new
analytic geometry, stopping, objective-choice, validation and spawned-worker
tests. Results are added only after inspecting the completed raw runs.

## Final results, 2026-10-13

All 300 scheduled runs completed in one preserved attempt each. All 100 cases
under each strategy selected the true utility maximizer; there were no errors,
incorrect selections, unresolved cases or timeouts. These are fresh matched
timings on Bench i7 with the fixed configuration above.

| Strategy | Total questions | Mean questions | Median / P90 / max questions | Total seconds | Mean seconds | Median / P90 / max seconds |
|---|---:|---:|---:|---:|---:|---:|
| Fixed depth 2 | 1010 | 10.10 | 8 / 21 / 34 | 713.061 | 7.131 | 5.392 / 15.388 / 21.469 |
| Previous adaptive rate | 928 | 9.28 | 7 / 17 / 44 | 611.543 | 6.115 | 5.143 / 11.975 / 22.465 |
| Stop-aware adaptive rate | 824 | 8.24 | 7 / 17.1 / 38 | 2390.720 | 23.907 | 19.048 / 53.563 / 89.281 |

P90 uses linear interpolation, so a question-count percentile can be fractional.
Compared with the previous adaptive strategy, the new strategy saves 104 actual
questions (11.207%) in aggregate: 47 cases improve, 36 tie and 17 worsen. Improved
cases save 136 questions and worsened cases add 32. Computation takes 3.909 times
as long; 99 cases are slower and one is faster. Against fixed depth 2 it saves
186 questions (18.416%); 55 cases improve, 20 tie and 25 worsen, with 3.353 times
the aggregate computation time.

The median question count stays at seven versus the previous adaptive strategy.
Its question-count P90 is also essentially unchanged. The maximum falls from
44 to 38, but long tails remain: case 28 takes 38 rather than 37 questions, and
32 questions after first reaching two candidates. Case 86 falls from 44 to 37
questions but still takes 32 questions after first reaching two candidates.
Case 57 improves from 32 to 21 questions. Six detailed case analyses preserve
the first divergent queries, actual answers and saved hypothetical policy proofs.
They illustrate how a potentially successful answer branch can attract a query
without being the branch taken by the actual target. They do not establish a
single general cause for every improvement or regression.

Singleton and confidence termination are kept separate. Fixed depth 2 stops
41 times with one candidate and 59 times via volume confidence; previous
adaptive stops 90/10; stop-aware stops 39/61. Every selected alternative is
correct in this cohort, including confidence stops with multiple feasible
candidates. Case 96 starts with four candidates and 99.1596% confidence, requiring
zero questions in all three modes; it remains in all 100-case denominators.

## Where elapsed solve time goes

The horizon clocks measure the entire horizon evaluation, including its policy
work and observers; they do not isolate LP or volume computation alone.

| Adaptive strategy | h1 evaluation seconds | h2 evaluation seconds | Remaining solve seconds | Selected h1 / h2 |
|---|---:|---:|---:|---:|
| Previous rate | 43.695 | 549.971 | 17.878 | 503 / 425 |
| Stop-aware rate | 51.963 | 2310.311 | 28.446 | 357 / 467 |

The new strategy spends 96.636% of solve time evaluating horizon two. Its h2
work checks the unchanged geometric 99% criterion throughout hypothetical
positive-probability branches; that work is repeated for candidate policies.
The saved best-per-horizon proofs contain 424 confidence-absorbed branches and
1310 singleton-absorbed branches. These are hypothetical proof entries, not
actual questions or unique states. Fewer actual questions do not offset the
more expensive planning. A more precise split between geometry, LPs and volume
checks would require dedicated additional profiling; these timings alone
cannot supply it.

This is a measured tradeoff in question count versus computation. If every
human answer added a constant latency, this cohort's extra 1779.177 seconds of
computation would be offset at about 17.107 seconds per saved question. That
calculation assumes equal answer latency and serial interaction, and is not a
prediction for a new workload. Retain the strategy as an explicit experimental
option rather than silently replacing the cheaper baseline.

## Verification and preserved evidence

The standard-library-only independent auditor passed all 300 raw results and
answer journals. It verified exact-rational utility winners and answer signs
over the stored binary64 inputs, candidate-state boundaries, termination,
weighted planning costs/loss, absorbing branches, saved horizon selection,
all source/input/protocol hashes and the full preserved attempt set. All 200
old-mode semantic trajectories match the previous experiment exactly, excluding
only timing fields and the new disabled configuration flag.

The controller and all recorded solver groups were confirmed stopped after
completion. The Mac mirror finished successfully. Full final reports and
raw artifacts are copied to both Mac and Bench and verified by a final SHA256
manifest. Original frozen archives and original benchmark evidence are preserved.
The final documentation is saved separately from the pre-run archive.

Relevant experiment artifacts are `comp.md`, `comp.json`,
`query-counts-all100.md` (100 rows), `query-trajectories.json`,
`case-diagnostics.md/json`, `independent-final-audit.json`,
`baseline-reference-equivalence.json`, `test-verification.json`,
`final-backup-manifest.json` and its Mac/Bench verification receipts.

Limits: one timing observation per case/strategy with rotated order; no separate
holdout cohort. Stored volume/share arithmetic and policy proofs are independently
checked, but polytopes and LPs are not independently reconstructed. Only saved
best-per-horizon policies can be audited; unpersisted losing policies are not
recomputed. No claim of globally minimal questions or a universal improvement
follows from these 100 problems. No further commit or push was made.
