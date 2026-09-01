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

## Exakte Optimierungen

Diese Änderungen sind standardmäßig aktiv und verändern weder Query-Menge noch
Lookahead-Tiefe:

1. **Quotientenintervalle je Zustand nur einmal berechnen.** Die
   Kandidatenanalyse liefert ihre Intervalle an die Query-Erzeugung weiter.
   Vorher wurden dieselben LPs im selben Zustand ein zweites Mal gelöst.
2. **Gespiegelte Zielpaare per Reziprokwert ableiten.** Für `(b, a)` wird das
   bereits gelöste Intervall für `(a, b)` invertiert. Wenn der Nenner auf null
   festgelegt ist oder eine sichere Inversion nicht möglich ist, bleibt der
   direkte LP-Fallback erhalten.
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
   begrenzten LRU-Cache für identische Aufrufe ohne explizit übergebene Samples. Die
   Terminierungsanalyse nutzt diese Session automatisch für alle Fragen eines
   Problems.

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

Die Werte sind keine belastbare Hardware-Benchmarkserie, zeigen aber die
Größenordnung. Für Vergleiche sollten mehrere Wiederholungen mit
`--repeats N` genutzt werden. `--reuse-worker-pool` misst dabei den
Session-Pfad.

## Korrektheitsprüfung

Neben der vollständigen Testsuite werden die abgeleiteten gespiegelten
Quotientenintervalle in Zufallstests gegen direkt gelöste LPs verglichen. Die
approximativen Schalter besitzen eigene Tests, bleiben aber bewusst opt-in.
