# Laufzeitoptimierungen des Multi-Step-Algorithmus

## Ziel und Messfall

Die Optimierungen reduzieren die Kosten der exakten Auswertung, ohne die
Standardsemantik des Algorithmus zu ändern. Als reproduzierbarer Messfall dient
`onestep/data/a5_a10_case.json` mit Tiefe 2, 400 Samples, Grid-Größe 21 und
Seed 1.

```bash
python multistep/scripts/benchmark_a5_a10_optimized.py \
  --mode optimized-all \
  --samples 400 \
  --burn-in 200 \
  --thinning 5 \
  --seed 1 \
  --grid-size 21 \
  --max-query-value 100 \
  --workers 1
```

Der Benchmark gibt neben Laufzeit und Ergebnis auch interne Zähler sowie
Zeitanteile aus. Mit `--workers 1` umfasst das Profil auch die rekursiven
Kindzustände; bei mehreren Prozessen enthält es nur die Arbeit des
Hauptprozesses.

## Korrigierter Sampler

Der Hit-and-Run-Sampler startet nicht mehr am beliebigen Machbarkeitspunkt des
LP-Solvers. Dieser Punkt war auf dem Simplex typischerweise eine Ecke und konnte
die Kette, insbesondere bei sieben Zielen, für viele Schritte festhalten.

Jetzt wird zuerst ein relativer Chebyshev-Mittelpunkt im durch die Gleichungen
definierten affinen Raum bestimmt. Optional laufen mehrere unabhängige Ketten;
`sample_points_with_diagnostics(...)` liefert dazu Anzahl eindeutiger Samples,
effektive Stichprobengröße und Split-R-hat. Die optimierte Konfiguration steuert
dies über `sampling_chain_count`, die Terminierungsanalyse verwendet
standardmäßig vier Ketten.

## Exakte Optimierungen

Diese Änderungen sind standardmäßig aktiv und verändern weder Query-Menge noch
Lookahead-Tiefe:

1. **Quotientenintervalle je Zustand nur einmal berechnen.** Die
   Kandidatenanalyse liefert ihre Intervalle an die Query-Erzeugung weiter.
   Vorher wurden dieselben LPs im selben Zustand ein zweites Mal gelöst.
2. **Geometrische Ecken-Engine.** Für jeden Kandidaten wird dessen
   Optimalitätspolytope einmal in den affinen Gleichungsraum projiziert. Aus
   seinen Ecken folgen danach sämtliche Quotientenintervalle. Degenerierte,
   unbeschränkte oder numerisch unsichere Fälle fallen automatisch auf die
   bisherige exakte LP-Engine zurück. Für A/B-Messungen kann
   `ratio_interval_engine="lp"` gesetzt werden.
3. **Sample-Punkte als exakte Machbarkeitszeugen verwenden.** Liegen Samples
   strikt auf beiden Seiten einer Query, sind keine zusätzlichen LPs zur
   Bestimmung der unterstützten Antworten nötig. Bei einem einseitigen Zeugen
   wird nur die gegenüberliegende Schranke optimiert.
4. **Zustandsraum nicht je Query neu aufbauen.** Die aktuelle
   `LinearConstraintSystem`-Instanz wird an die Query-Auswertung weitergegeben.
5. **Solver-Matrizen cachen.** Die Listen der Nebenbedingungen werden einmal in
   NumPy-Arrays umgewandelt und bis zur nächsten Mutation wiederverwendet.
6. **Sampling vektorisieren.** Die zulässigen Schrittintervalle des
   Hit-and-Run-Samplers werden mit Matrixoperationen berechnet; die Matrizen
   werden nicht mehr in jedem Schritt neu aufgebaut.
7. **Prozesspool und Zustandsresultate wiederverwenden.** Die öffentliche
   `OptimizedValueFunctionSession` hält Worker am Leben und besitzt einen
   begrenzten LRU-Cache für identische Aufrufe ohne explizit übergebene Samples.
   Zusätzlich cached `analyze_state(...)` Machbarkeit, Kandidaten und
   Quotientenintervalle unabhängig von der Lookahead-Tiefe. Semantisch gleiche
   gespiegelte Queries teilen sich einen Cache-Key.
8. **Kindzustände vorfiltern.** Bereits berechnete Elternintervalle entfernen
   Kandidaten, die eine konkrete Query-Antwort unmöglich erfüllen können. Der
   Kindaufruf übernimmt außerdem den schon aufgebauten und auf Machbarkeit
   geprüften Gewichtsraum.

## Optionale approximative Begrenzungen

Die folgenden Schalter sind standardmäßig `None`. Sie können große Fälle stark
beschleunigen, dürfen aber die gewählte Query verändern und müssen deshalb
gegen Lösungsqualität und Anzahl der Folgefragen evaluiert werden:

- `max_query_candidates_per_state=N` bewertet nur die `N` Queries mit den
  ausgewogensten Sample-Partitionen.
- `adaptive_depth_candidate_threshold=K` verwendet Tiefe 1, solange mehr als
  `K` Kandidaten verbleiben, und schaltet danach auf die angeforderte Tiefe
  zurück.

In der Terminierungsanalyse heißen die zugehörigen CLI-Optionen
`--max-query-candidates` und `--adaptive-depth-candidate-threshold`.

## Ratio-, Quantil-, Entropie- und Regret-Politiken

Posterior-Quantil-Queries sind opt-in. Beispielsweise ergänzt
`posterior_quantile_levels=(0.25, 0.5, 0.75)` für jedes kanonische Zielpaar die
drei entsprechenden Sample-Quantile zu den Ratio-Queries. Die exakte
Value-Function kann danach alle Queries bewerten.

Für größere Query-Mengen gibt es zwei experimentelle Shortlist-Ziele:

- `posterior_query_objective="entropy"` minimiert die erwartete Entropie des
  posterioren Gewinner-Kandidaten.
- `posterior_query_objective="regret"` minimiert den erwarteten posterioren
  Entscheidungs-Regret.

`posterior_query_shortlist_size=N` begrenzt dabei nur die zusätzlichen
Grid-/Quantil-Queries. Alle geometrisch erzeugten Ratio-Queries bleiben in der
Auswertung. Die Shortlist ist approximativ und sollte anhand Laufzeit,
Folgefragen und finalem Regret evaluiert werden.

## Dimension-7-Benchmark

Der neue Benchmark erzeugt deterministische 7-Ziele-/10-Alternativen-Fälle und
schreibt optional alle Messwerte einschließlich Samplerdiagnostik und
Profiling-Zählern nach JSON:

```bash
python multistep/scripts/benchmark_dimension_seven.py --preset quick

python multistep/scripts/benchmark_dimension_seven.py \
  --preset standard \
  --output-json multistep/data/benchmarks/dimension7.json
```

`quick` misst standardmäßig Tiefe 1 auf einem Problem. `standard` umfasst drei
Probleme, Tiefe 1 und 2 sowie Ratio, Ratio+Quantil, Entropie, Regret und
Grid+Ratio. Einzelne Dimensionen lassen sich mit `--depths`, `--policies`,
`--samples`, `--chains`, `--workers` und `--ratio-engine` isolieren.

## Profilierung im Code

Für gezielte Messungen kann die Kernfunktion ohne Monkey-Patching instrumentiert
werden:

```python
from multistep.optimized import collect_optimization_profile

with collect_optimization_profile() as profile:
    result = compute_value_function_optimized(...)

print(profile.counters)
print(profile.seconds_by_operation)
```

Wichtige Zähler sind `state_calls`, `ratio_interval_batches`,
`query_evaluations`, `query_support_checks`, `sampling_calls` und
`branch_checks`.

## Gemessener Effekt

Einzelmessungen auf dem Entwicklungsrechner mit dem oben beschriebenen
a5/a10-Fall:

| Variante | Vorher | Nachher | Ergebnis |
|---|---:|---:|---|
| optimiert, seriell | 7,287 s | 3,436 s | Wert 1,453; gleiche beste Query |
| optimiert, 4 Worker | 2,774 s | 1,906 s | Wert 1,453; gleiche beste Query |

Auf einer zufälligen 7-Ziele-/10-Alternativen-Instanz benötigte ein kompletter
Quotientenintervall-Batch 0,029 s mit Geometrie gegenüber 1,083 s mit LP
(37-fach). In einem kurzen Dimension-7-Tiefe-2-Lauf mit 240 Samples und vier
Workern lagen Ratio, Entropie-Shortlist und Regret-Shortlist bei 2,26 s, 2,79 s
und 3,09 s; die vollständige Ratio+Quantil-Menge benötigte 5,47 s. Diese
Einzelwerte dienen nur als Smoke-Benchmark, nicht als statistische
Qualitätsaussage.

Die Werte sind keine belastbare Hardware-Benchmarkserie, zeigen aber die
Größenordnung. Für Vergleiche sollten mehrere Wiederholungen mit
`--repeats N` genutzt werden. `--reuse-worker-pool` misst dabei den
Session-Pfad.

## Korrektheitsprüfung

Neben der vollständigen Testsuite werden geometrische Quotientenintervalle in
Zufallstests gegen direkt gelöste LPs verglichen. Der Sampler besitzt
Regressionstests für den relativen Simplex-Innenpunkt, die Exploration in
Dimension 7 und seine Diagnostik. Die approximativen Schalter besitzen eigene
Tests, bleiben aber bewusst opt-in.
