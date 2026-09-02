# Dimension 5: Grid 10 + Ratio, Tiefe 2, Volumenkonfidenz 0,99

## Konfiguration

- 5 Ziele
- 3, 5 und 7 erzeugte Alternativen
- jeweils 10 deterministisch reproduzierbare Probleme
- Lookahead-Tiefe 2
- Grid 10 + Ratio ab Tiefe 2
- Ratio auf Tiefe 1
- exakte Volumenwahrscheinlichkeiten
- Volumen-Konfidenzterminierung bei 0,99
- 4 persistente Root-Worker
- Seed 20260902

## Zusammenfassung

| Alternativen | Gelöst | Korrekt | Queries gesamt | Queries Mittel | Median | Min–Max | Zeit gesamt | Zeit Mittel | Exakt | Konfidenz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 10/10 | 10/10 | 79 | 7,9 | 7,5 | 1–23 | 323,31 s | 32,33 s | 4 | 6 |
| 5 | 10/10 | 10/10 | 79 | 7,9 | 7,0 | 4–16 | 415,71 s | 41,57 s | 4 | 6 |
| 7 | 10/10 | 10/10 | 92 | 9,2 | 7,0 | 2–20 | 589,27 s | 58,93 s | 4 | 6 |
| **Gesamt** | **30/30** | **30/30** | **250** | **8,33** | **7,0** | **1–23** | **1328,28 s** | **44,28 s** | **12** | **18** |

Die mittlere Zahl initial geometrisch relevanter Kandidaten betrug 2,7 bei 3,
4,4 bei 5 und 5,7 bei 7 erzeugten Alternativen. Dominierte Alternativen haben
von Beginn an kein positives Optimalitätsvolumen.

## Query-Verteilungen

- 3 Alternativen: 1×1, 2×3, 7×1, 8×1, 11×2, 12×1, 23×1 Probleme.
- 5 Alternativen: 4×3, 5×1, 7×2, 9×1, 11×1, 12×1, 16×1 Probleme.
- 7 Alternativen: 2×1, 4×2, 5×1, 7×2, 8×1, 17×1, 18×1, 20×1 Probleme.

Die Notation `q×n` bedeutet: `n` Probleme benötigten jeweils `q` Queries.

## Vergleich mit Dimension 3

| Dimension | Queries Mittel | Zeit Mittel | Gesamtzeit für 30 Probleme |
|---:|---:|---:|---:|
| 3 | 3,83 | 4,45 s | 133,60 s |
| 5 | 8,33 | 44,28 s | 1328,28 s |

Dimension 5 benötigte damit im Mittel 2,17-mal so viele Queries und ungefähr
9,94-mal so viel Laufzeit pro Problem. Pro gespeicherter Query stieg die
mittlere Laufzeit von ungefähr 1,16 s auf 5,31 s.

## Datenqualität

Alle 250 Query-Datensätze enthalten den Erwartungswert, die verbleibenden
Kandidaten, deren Rohvolumina und deren normierte Volumenanteile. Der größte
numerische Fehler der Summe der Volumenanteile gegenüber 1 betrug
2,22e-16. Alle 18 Konfidenzentscheidungen wählten den tatsächlichen Gewinner.

## Rohdaten

- `multistep/data/d5_grid10_ratio_depth2_confidence099_a3_p10_seed20260902.json`
- `multistep/data/d5_grid10_ratio_depth2_confidence099_a5_p10_seed20260902.json`
- `multistep/data/d5_grid10_ratio_depth2_confidence099_a7_p10_seed20260902.json`

Jede Datei speichert pro Problem unter `queries` für jede gestellte Query unter
anderem `expected_value`, `remaining_candidates`, `candidate_volumes` und
`candidate_volume_shares`. Der Endzustand enthält zusätzlich
`termination_reason`, `selected_candidate`, `selected_candidate_volume_share`,
`residual_volume_share`, `final_candidates` und `selection_is_correct`.
