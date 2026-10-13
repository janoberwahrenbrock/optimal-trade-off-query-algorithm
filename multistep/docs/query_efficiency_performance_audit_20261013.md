# Performance audit of the two adaptive query strategies

Date: 2026-10-13. This is a diagnosis of the unchanged implementation, not an
optimization result. No algorithm changes, commits or pushes were made.

The previous 100-case comparison already shows that ordinary adaptive
reduction-per-query is cheaper than fixed depth two: 611.543 versus 713.061
seconds, with 928 versus 1010 questions. Stop-aware adaptive rate takes
2390.720 seconds and 824 questions. All three have 100/100 correct selections.
The extra runtime is concentrated in the stop-aware planning model.

## Matched diagnostics

Four fresh sequential diagnostic runs use the frozen 100-case source and
inputs: problems 1 and 28, each under ordinary adaptive and stop-aware rate.
Five goals, five alternatives, Grid5 plus ratios at horizon two, ratios at
horizon one, eight workers, exact volumes, actual 99% stopping, the existing
100-question guard and no time limit are unchanged. Six native thread limits
remain one. Both horizons and all permitted query candidates are evaluated.

Every submitted worker task collects cProfile, existing operation counters
and timers, and cache snapshots; the parent has a separate profile. Original
observers and answer/state journals remain active. All four complete result
trajectories and configurations match their accepted reference exactly after
excluding timing and observational cache fields. Frozen Python hashes are
checked before and after.
The controller and its four solver mains have exited.

**Measurement limits:** instrumentation and profile exports add overhead.
These solve times cannot be used to claim a speedup. Worker-time sums overlap
in elapsed time. Inclusive function/phase times are nested and cannot be
added as independent costs. The two selected cases are diagnostics rather
than a representative new performance series.

| Diagnostic quantity | Ordinary, case 1 | Stop-aware, case 1 | Ordinary, case 28 | Stop-aware, case 28 |
|---|---:|---:|---:|---:|
| Actual questions | 6 | 9 | 37 | 38 |
| Hypothetical query evaluations | 14390 | 18666 | 107588 | 104738 |
| Candidate-analysis batches | 1209 | 29492 | 5640 | 138012 |
| Vertex-enumeration calls | 22071 | 99610 | 84249 | 335425 |
| LP retry-wrapper calls | 29733 | 137969 | 121983 | 510436 |
| Profiled worker self-time sum, seconds | 36.26 | 201.59 | 188.31 | 881.45 |

Canonical exact counters and times are retained in `profile-summary.json`
and the raw pstats. The decisive
case-28 observation is independent of timing: stop-aware evaluates slightly
fewer hypothetical queries but enumerates nearly four times as many polytopes.
Its leaf answers require candidate volumes and confidence, where the ordinary
planner can use the existing fast terminal candidate-count calculation.

## Where stop-aware worker computation goes

Independent caller-edge aggregation separates candidate and probability geometry:

| Inclusive worker function cost | Case 1, seconds / calls | Case 28, seconds / calls |
|---|---:|---:|
| Complete candidate analysis | 137.77 / 29492 | 563.64 / 138012 |
| Candidate-region vertex enumeration | 85.53 / 77274 | 332.44 / 265523 |
| Probability-region vertex enumeration | 24.98 / 22336 | 85.69 / 69902 |
| Candidate volumes from existing vertices | 15.25 / 77274 | 66.73 / 265523 |
| All ratio arithmetic and result materialization | 17.04 / 77274 | 64.18 / 265523 |
| Separate child feasibility checks | 18.59 / 29492 | 108.64 / 138012 |
| Weight-space plus optimality-region construction | 11.93 | 86.01 |

These rows overlap, especially candidate analysis with its geometry and LPs.
They must not be summed. The two construction call trees are disjoint with
each other but belong to the larger overall evaluation.

Candidate analysis dominates. The confidence-threshold comparison itself is
cheap: operation timers record only 0.211 and 1.004 cumulative worker seconds
for `stop_aware_volume_termination`; the geometric work needed to supply the
shares is inside candidate analysis. Validation and score arithmetic are not
the principal bottleneck.

## Proposed experiments, ranked by scope and evidence

1. **Reuse immutable constraint prefixes and utility-difference rows.**
   Build a child from an independently owned copy of its canonical parent
   constraints plus the unchanged normalized answer row; prepare unchanged
   utility differences once. This removes repeated reconstruction of the whole
   history and repeated Python `add_*` work. Preserve coefficients, signed zeros,
   row/equality order, parent immutability and solver-matrix invalidation. For a
   noncanonical caller-supplied parent, retain the original history rebuild.
   The measured construction region is about 5.9% and 9.8% of profiled worker
   self-time; these are target costs, not promised solver speedups.

2. **Reuse a rigorously validated feasibility witness where available.**
   Candidate geometry or previously validated child geometry may supply proof
   that the child is feasible, avoiding a separate zero-objective LP. The
   separate checks account for about 9.2% and 12.3% of profiled worker self-time.
   Positive answer probability alone is not a sufficient replacement for the
   current numerical check. Respect row scaling, tolerances, arbitrary parent
   systems and existing failure handling; uncertain or degenerate cases retain
   the original LP. This needs an explicit correctness guard before adoption.

3. **Retain candidate parent geometry and cut it for the leaf queries.**
   Within each horizon-two continuation, many leaf queries cut the same already
   analyzed candidate optimality regions by just one new halfspace. Keeping
   parent vertices and genuine edges/active facets could produce child vertices
   from retained points and crossed-edge intersections, replacing repeated
   interior-point LPs and `HalfspaceIntersection` calls. Parent vertices currently
   exist and are then discarded; edge/facet data is not retained. This targets
   the largest cost and is more ambitious than the first two experiments.
   Volumes and candidate relevance still require exact geometric evaluation.
   Triangulation diagonals are not automatically genuine polytope edges.
   Mathematically identical regions can produce different floating-point vertices,
   volumes, 99% decisions or query rankings. Therefore this is a separate
   experimental geometry path requiring complete validated cuts, conservative
   fallback and strict paired result checks; no speedup or equivalence is yet
   demonstrated. This direct parent-to-child reuse differs from a broad cache
   that hopes unrelated constraint signatures repeat.

4. **Smaller allocation and shared-horizon improvements.**
   Leaf analysis can compute the same ratio bounds and candidate classification
   without constructing unused `RatioInterval`, `LinearOptimizationResult` and
   goal-pair maps. Materialize full intervals only when a continuation needs them.
   Ratio-specific Pydantic allocation is about 3.8% / 3.4% of worker self-time,
   so it is secondary. Keep the existing relevant-versus-feasible fallback,
   all numerical ratio reductions, tolerances and volume normalization over all
   checked candidates before filtering. Actual candidate count must not be
   replaced by the number of positive candidate volumes.
   Shared h1/h2 ratio roots can prepare identical probabilities and immediate
   child analyses once, then derive both original scores and source labels.
   h1 worker tasks total only about 0.5% / 0.7% here; in the original full series
   h1 elapsed clocks account for 2.2% of stop-aware solve time. This cannot alone
   recover the large cost increase.

## Ideas to deprioritize

Larger caches are not justified: the new diagnostic reaches only 2844 / 8853
volume or vertex entries and roughly 0.85 / 4.42 MiB of vertices per worker,
below the existing 32768-entry and 64-MiB limits. Affine decomposition needs
only eight actual misses per case across eight workers, so it already reuses
well. The earlier broad shared-geometry cache had zero hits and made the old
planner slower; it should not be reinstated without measured new reuse.
Probability cuts and history-built cuts also use different row scaling for
query values above one, which prevents some exact-signature matches.

Chunk-size changes and proof-journal removal are not first priorities. Native
profiles show substantial worker geometry cost, and the prior chunking trials
did not robustly help. Journals and selected-policy proofs are important evidence;
their wall-time overhead would need a matched diagnostic before any claim.

## Acceptance protocol for subsequent optimization work

No optimization is implemented by this audit. For each future candidate, keep
the current source/configuration as the frozen reference, change one mechanism
at a time, use repeated ordinary timings on the same cases and rotated order,
and keep diagnostic timings separate. Verify complete query trajectories,
horizons, raw expected C, effective loss, expected questions, rates, volumes,
candidate sets, selected alternative and stopping reason. Include threshold
boundaries, equality answers, thin/degenerate regions and noncanonical parent
systems. Retain only a measured improvement with the required unchanged
semantics, and document discarded experiments as well.

Artifacts: `protocol.json`, `run_audit.py`, `diag_replay.py`, `state.json`,
all four `jobs/*/result.json`, `questions.jsonl`, `progress.jsonl`, per-task
native profiles and pstats, `profile-summary.json`, and final hash receipts.
