# Dimension 5: Grid 10 + Ratio, Tiefe 2, 150 Probleme

## Konfiguration

- 5 Ziele
- 3, 5 und 7 erzeugte Handlungsalternativen
- jeweils 50 deterministisch reproduzierbare Probleme, insgesamt 150
- Lookahead-Tiefe 2
- Grid 10 + Ratio für die Root-Query
- Ratio für die Query-Kandidaten auf Tiefe 1
- exakte Volumenwahrscheinlichkeiten
- Volumen-Konfidenzterminierung bei 0,99
- maximal 100 Queries pro Problem
- 4 persistente Root-Worker
- Seed 20260902

Die vorhandenen Probleme 1–25 wurden nicht neu berechnet. Die Analyse wurde
reproduzierbar um Probleme 26–50 erweitert und danach zu vollständigen
P50-Dateien zusammengeführt.

## Zusammenfassung

| Alternativen | Beendet | Zielgewinner getroffen | Queries gesamt | Queries Mittel | Median | Std.-Abw. | Min–Max | Zeit gesamt | Zeit Mittel | Zeitmedian | Exakt | Konfidenz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 50/50 | 50/50 | 397 | 7,94 | 4 | 7,91 | 0–33 | 1.580,26 s | 31,61 s | 14,84 s | 20 | 30 |
| 5 | 50/50 | 50/50 | 435 | 8,70 | 7 | 7,09 | 0–35 | 2.555,08 s | 51,10 s | 35,23 s | 14 | 36 |
| 7 | 50/50 | 49/50 | 523 | 10,46 | 8 | 8,00 | 1–35 | 3.264,79 s | 65,30 s | 54,97 s | 17 | 33 |
| **Gesamt** | **150/150** | **149/150** | **1.355** | **9,03** | **7** | **7,75** | **0–35** | **7.400,14 s** | **49,33 s** | **35,53 s** | **51** | **99** |

Die 75 neu berechneten Probleme 26–50 benötigten zusammen 3.702,08 Sekunden
beziehungsweise rund 1 Stunde, 1 Minute und 42 Sekunden reine Solverzeit. Die
vollständigen 150 Probleme enthalten insgesamt rund 2 Stunden, 3 Minuten und
20 Sekunden gespeicherte Solverzeit.

Die mittlere Zahl initial geometrisch relevanter Kandidaten betrug 2,66 bei 3,
4,10 bei 5 und 5,50 bei 7 erzeugten Alternativen. Mit wachsender Zahl
relevanter Kandidaten steigen sowohl die mittlere Query-Zahl als auch die
Laufzeit. Die Verteilungen bleiben stark rechtsschief: Der Gesamtmedian beträgt
35,53 Sekunden, während der Mittelwert 49,33 Sekunden beträgt. Der langsamste
Fall benötigte 203,35 Sekunden.

## Query-Verteilungen

- 3 Alternativen: 0×3, 1×3, 2×8, 3×7, 4×5, 6×1, 7×3, 8×4, 9×1,
  11×3, 12×2, 13×1, 14×2, 15×1, 22×1, 23×2, 26×1, 29×1 und 33×1
  Probleme.
- 5 Alternativen: 0×1, 1×2, 2×1, 3×6, 4×8, 5×4, 6×2, 7×3, 8×3,
  9×5, 10×1, 11×2, 12×1, 13×3, 16×3, 18×1, 23×1, 25×1, 27×1 und
  35×1 Probleme.
- 7 Alternativen: 1×1, 2×2, 3×2, 4×8, 5×5, 6×1, 7×3, 8×4, 9×4,
  10×2, 11×3, 12×2, 13×1, 16×2, 17×2, 18×1, 20×1, 21×1, 22×1,
  23×1, 31×1, 34×1 und 35×1 Probleme.

Die Notation `q×n` bedeutet: `n` Probleme benötigten jeweils `q` Queries. Kein
Problem erreichte das Limit von 100 Queries.

## Konfidenzentscheidung und Treffergenauigkeit

51 Probleme wurden erst beendet, als exakt ein Kandidat übrig war. Bei 99
Problemen griff vorher die 0,99-Volumenkonfidenz. 98 dieser 99 Entscheidungen
stimmten mit dem Gewinner für das gezogene Zielgewicht überein.

Der einzige abweichende Lauf ist Problem 43 mit sieben Alternativen. Das
Verfahren stoppte für Kandidat 4 bei einem Volumenanteil von 99,102366 %. Das
gezogene Zielgewicht lag im verbleibenden Restbereich von 0,897634 % und machte
Kandidat 2 zum Zielgewinner. Dies ist kein Geometrie- oder Numerikfehler,
sondern das bewusst akzeptierte Restrisiko des 0,99-Konfidenzstopps.

## Datenqualität

- Alle 150 Probleme wurden beendet; 149 Entscheidungen trafen den jeweiligen
  Zielgewinner.
- Die Summe der `question_count`-Werte und die Zahl der gespeicherten
  Query-Datensätze stimmen überein: jeweils 1.355.
- Jeder Query-Datensatz enthält `expected_value`, `remaining_candidates`,
  `candidate_volumes` und `candidate_volume_shares`.
- Die gespeicherte Kandidatenzahl stimmt bei jeder Query mit der Länge der
  Kandidatenliste überein.
- Der größte numerische Fehler der Summe der Volumenanteile gegenüber 1 beträgt
  2,22e-16.
- Es traten keine Solver-, Geometrie- oder Laufzeitabbrüche auf.

## Rohdaten

- `multistep/data/d5_grid10_ratio_depth2_confidence099_a3_p50_seed20260902.json`
- `multistep/data/d5_grid10_ratio_depth2_confidence099_a5_p50_seed20260902.json`
- `multistep/data/d5_grid10_ratio_depth2_confidence099_a7_p50_seed20260902.json`

Jede Datei speichert die vollständigen Alternativen und das gezogene
Zielgewicht. Für jede Query werden Antwort, Laufzeit, Erwartungswert,
verbleibende Kandidaten sowie deren absolute und normierte Volumina festgehalten.
