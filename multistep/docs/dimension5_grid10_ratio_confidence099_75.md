# Dimension 5: Grid 10 + Ratio, Tiefe 2, 75 Probleme

## Konfiguration

- 5 Ziele
- 3, 5 und 7 erzeugte Handlungsalternativen
- jeweils 25 deterministisch reproduzierbare Probleme, insgesamt 75
- Lookahead-Tiefe 2
- Grid 10 + Ratio für die Root-Query
- Ratio für die Query-Kandidaten auf Tiefe 1
- exakte Volumenwahrscheinlichkeiten
- Volumen-Konfidenzterminierung bei 0,99
- maximal 100 Queries pro Problem
- 4 persistente Root-Worker
- Seed 20260902

Die bereits vorhandenen Probleme 1–10 wurden nicht neu berechnet. Die Analyse
wurde reproduzierbar um Probleme 11–25 erweitert und anschließend zu
vollständigen P25-Dateien zusammengeführt.

## Zusammenfassung

| Alternativen | Gelöst | Zielgewinner getroffen | Queries gesamt | Queries Mittel | Median | Std.-Abw. | Min–Max | Zeit gesamt | Zeit Mittel | Zeitmedian | Exakt | Konfidenz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 25/25 | 25/25 | 144 | 5,76 | 3 | 5,04 | 1–23 | 540,24 s | 21,61 s | 9,59 s | 9 | 16 |
| 5 | 25/25 | 25/25 | 259 | 10,36 | 9 | 7,71 | 3–35 | 1.603,23 s | 64,13 s | 43,79 s | 8 | 17 |
| 7 | 25/25 | 25/25 | 241 | 9,64 | 8 | 7,86 | 1–35 | 1.554,59 s | 62,18 s | 53,34 s | 5 | 20 |
| **Gesamt** | **75/75** | **75/75** | **644** | **8,59** | **7** | **7,28** | **1–35** | **3.698,06 s** | **49,31 s** | **33,76 s** | **22** | **53** |

Die 45 neu berechneten Probleme 11–25 benötigten zusammen 2.369,78 Sekunden
beziehungsweise etwa 39 Minuten 30 Sekunden reine Solverzeit. Einschließlich
der früheren 30 Probleme beträgt die gespeicherte Solverzeit rund 1 Stunde,
1 Minute und 38 Sekunden.

Die mittlere Zahl initial geometrisch relevanter Kandidaten betrug 2,64 bei 3,
4,40 bei 5 und 5,60 bei 7 erzeugten Alternativen. Die Laufzeitverteilung ist
deutlich rechtsschief: Der Median aller Probleme liegt bei 33,76 Sekunden, der
Mittelwert wegen weniger Langläufer bei 49,31 Sekunden. Der langsamste Fall
benötigte 180,16 Sekunden.

## Query-Verteilungen

- 3 Alternativen: 1×2, 2×6, 3×5, 4×2, 6×1, 7×1, 8×3, 11×2, 12×1,
  13×1 und 23×1 Probleme.
- 5 Alternativen: 3×1, 4×5, 5×2, 6×1, 7×2, 8×1, 9×5, 11×1, 12×1,
  13×1, 16×2, 23×1, 27×1 und 35×1 Probleme.
- 7 Alternativen: 1×1, 2×2, 3×1, 4×4, 5×2, 7×2, 8×4, 9×1, 10×1,
  12×1, 17×2, 18×1, 20×1, 23×1 und 35×1 Probleme.

Die Notation `q×n` bedeutet: `n` Probleme benötigten jeweils `q` Queries. Kein
Problem erreichte das Limit von 100 Queries.

## Einordnung

Mit 25 Problemen pro Gruppe ist die Streuung besser sichtbar als im früheren
Zehnerblock. Insbesondere enthielten die neuen Probleme mehrere Langläufer mit
23 bis 35 Queries. In dieser Stichprobe benötigten fünf Alternativen im Mittel
etwas mehr Queries und Laufzeit als sieben Alternativen. Daraus folgt keine
generelle Überlegenheit der größeren Gruppe; die tatsächlich geometrisch
relevanten Kandidaten und der konkrete Verlauf bestimmen den Schwierigkeitsgrad
stärker als die bloße Zahl erzeugter Alternativen.

## Datenqualität

- Alle 75 Probleme wurden beendet und der Zielgewinner wurde in allen 75 Fällen
  getroffen.
- Die Summe der `question_count`-Werte und die Zahl der gespeicherten
  Query-Datensätze stimmen überein: jeweils 644.
- Jeder Query-Datensatz enthält `expected_value`, `remaining_candidates`,
  `candidate_volumes` und `candidate_volume_shares`.
- Die gespeicherte Kandidatenzahl stimmt bei jeder Query mit der Länge der
  Kandidatenliste überein.
- Der größte numerische Fehler der Summe der Volumenanteile gegenüber 1 beträgt
  2,22e-16.
- Es traten keine Solver-, Geometrie- oder Laufzeitabbrüche auf.

## Rohdaten

- `multistep/data/d5_grid10_ratio_depth2_confidence099_a3_p25_seed20260902.json`
- `multistep/data/d5_grid10_ratio_depth2_confidence099_a5_p25_seed20260902.json`
- `multistep/data/d5_grid10_ratio_depth2_confidence099_a7_p25_seed20260902.json`

Jede Datei speichert für jedes Problem die vollständigen Alternativen und das
gezogene Zielgewicht. Für jede Query werden Antwort, Laufzeit, Erwartungswert,
verbleibende Kandidaten sowie deren absolute und normierte Volumina festgehalten.
