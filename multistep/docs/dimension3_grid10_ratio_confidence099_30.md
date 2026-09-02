# Dimension 3: Grid 10 + Ratio, Tiefe 2, Volumenkonfidenz 0,99

## Konfiguration

- 3 Ziele
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
| 3 | 10/10 | 10/10 | 32 | 3,2 | 2,5 | 1–8 | 29,52 s | 2,95 s | 8 | 2 |
| 5 | 10/10 | 10/10 | 44 | 4,4 | 3,5 | 0–9 | 52,57 s | 5,26 s | 5 | 5 |
| 7 | 10/10 | 10/10 | 39 | 3,9 | 3,5 | 2–7 | 51,51 s | 5,15 s | 8 | 2 |
| **Gesamt** | **30/30** | **30/30** | **115** | **3,83** | **3,0** | **0–9** | **133,60 s** | **4,45 s** | **21** | **9** |

Die mittlere Zahl initial geometrisch relevanter Kandidaten betrug 2,3 bei 3,
3,3 bei 5 und 3,7 bei 7 erzeugten Alternativen. Dominierte Alternativen haben
von Beginn an kein positives Optimalitätsvolumen.

## Query-Verteilungen

- 3 Alternativen: 1×2, 2×3, 3×2, 4×1, 6×1, 8×1 Probleme.
- 5 Alternativen: 0×1, 1×1, 3×3, 4×1, 6×1, 7×1, 8×1, 9×1 Probleme.
- 7 Alternativen: 2×2, 3×3, 4×1, 5×3, 7×1 Probleme.

## Datenqualität

Alle 115 Query-Datensätze enthalten den Erwartungswert, die verbleibenden
Kandidaten, deren Rohvolumina und deren normierte Volumenanteile. Der größte
numerische Fehler der Summe der Volumenanteile gegenüber 1 betrug
2,22e-16. Alle neun Konfidenzentscheidungen wählten den tatsächlichen Gewinner.

## Rohdaten

- `multistep/data/d3_grid10_ratio_depth2_confidence099_a3_p10_seed20260902.json`
- `multistep/data/d3_grid10_ratio_depth2_confidence099_a5_p10_seed20260902.json`
- `multistep/data/d3_grid10_ratio_depth2_confidence099_a7_p10_seed20260902.json`

Jede Datei speichert pro Problem unter `queries` für jede gestellte Query unter
anderem `expected_value`, `remaining_candidates`, `candidate_volumes` und
`candidate_volume_shares`. Der Endzustand enthält zusätzlich
`termination_reason`, `selected_candidate`, `selected_candidate_volume_share`,
`residual_volume_share`, `final_candidates` und `selection_is_correct`.
