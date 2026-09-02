# Dimension 3: Grid 10 + Ratio, Tiefe 2, 300 Probleme

## Konfiguration

- 3 Ziele
- 3, 5 und 7 erzeugte Handlungsalternativen
- jeweils 100 deterministisch reproduzierbare Probleme, insgesamt 300
- Lookahead-Tiefe 2
- Grid 10 + Ratio für die Root-Query
- Ratio für die Query-Kandidaten auf Tiefe 1
- exakte Volumenwahrscheinlichkeiten
- Volumen-Konfidenzterminierung bei 0,99
- maximal 100 Queries pro Problem
- 4 persistente Root-Worker
- Seed 20260902

## Zusammenfassung

| Alternativen | Gelöst | Zielgewinner getroffen | Queries gesamt | Queries Mittel | Median | Std.-Abw. | Min–Max | Zeit gesamt | Zeit Mittel | Median | Exakt | Konfidenz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 100/100 | 100/100 | 260 | 2,60 | 2 | 1,99 | 0–10 | 205,12 s | 2,05 s | 1,99 s | 80 | 20 |
| 5 | 100/100 | 99/100 | 367 | 3,67 | 3 | 2,70 | 0–12 | 308,77 s | 3,09 s | 2,99 s | 68 | 32 |
| 7 | 100/100 | 100/100 | 393 | 3,93 | 3 | 2,53 | 0–13 | 371,83 s | 3,72 s | 3,58 s | 66 | 34 |
| **Gesamt** | **300/300** | **299/300** | **1.020** | **3,40** | **3** | **2,49** | **0–13** | **885,71 s** | **2,95 s** | **2,65 s** | **214** | **86** |

Die reine, in den Ergebnissen gemessene Solverzeit betrug insgesamt rund
14 Minuten 46 Sekunden. Mit mehr erzeugten Alternativen steigt die Zahl der
initial geometrisch relevanten Kandidaten im Mittel von 2,28 über 2,92 auf
3,63. Entsprechend wächst die mittlere Query-Zahl von 2,60 auf 3,93.

## Query-Verteilungen

- 3 Alternativen: 0×11, 1×16, 2×30, 3×21, 4×11, 5×3, 6×2, 7×2,
  8×1, 9×2 und 10×1 Probleme.
- 5 Alternativen: 0×10, 1×15, 2×12, 3×16, 4×15, 5×9, 6×7, 7×9,
  8×1, 9×2, 10×1, 11×2 und 12×1 Probleme.
- 7 Alternativen: 0×5, 1×4, 2×21, 3×26, 4×13, 5×8, 6×9, 7×5,
  8×3, 9×1, 10×3, 12×1 und 13×1 Probleme.
- Insgesamt: 0×26, 1×35, 2×63, 3×63, 4×39, 5×20, 6×18, 7×16,
  8×5, 9×5, 10×5, 11×2, 12×2 und 13×1 Probleme.

Damit benötigten 187 von 300 Problemen höchstens drei Queries und 226 von
300 höchstens vier Queries. Kein Problem erreichte das Limit von 100 Queries.

## Terminierung und Treffergenauigkeit

214 Probleme wurden erst beendet, als exakt ein Kandidat übrig war. Bei 86
Problemen griff vorher die 0,99-Volumenkonfidenz. In 85 dieser 86 Fälle war der
gewählte Kandidat auch für das gezogene Zielgewicht optimal.

Der einzige abweichende Lauf ist Problem 71 mit fünf Alternativen. Bereits im
Ausgangszustand entfielen 99,889350 % des zulässigen Gewichtsraums auf Kandidat
4 und 0,110650 % auf Kandidat 2. Daher stoppte das Verfahren ohne Query für
Kandidat 4. Das zufällig gezogene Zielgewicht lag jedoch in dem kleinen
Restbereich und machte Kandidat 2 zum Zielgewinner. Dies ist kein numerischer
oder geometrischer Fehler, sondern genau das zugelassene Restrisiko einer
0,99-Konfidenzterminierung. Wer für jeden Testlauf zwingend den Zielgewinner
treffen möchte, muss die exakte Ein-Kandidaten-Terminierung verlangen oder den
Schwellenwert erhöhen.

## Datenqualität

- Alle 300 Probleme wurden beendet und gespeichert.
- Die Summe der `question_count`-Werte und die Zahl der gespeicherten
  Query-Datensätze stimmen überein: jeweils 1.020.
- Jeder Query-Datensatz enthält `expected_value`, `remaining_candidates`,
  `candidate_volumes` und `candidate_volume_shares`.
- Die gespeicherte Kandidatenzahl stimmt bei jeder Query mit der Länge der
  Kandidatenliste überein.
- Der größte numerische Fehler der Summe der Volumenanteile gegenüber 1 beträgt
  2,22e-16.
- Es traten keine Solver-, Geometrie- oder Laufzeitabbrüche auf.

## Rohdaten

- `multistep/data/d3_grid10_ratio_depth2_confidence099_a3_p100_seed20260902.json`
- `multistep/data/d3_grid10_ratio_depth2_confidence099_a5_p100_seed20260902.json`
- `multistep/data/d3_grid10_ratio_depth2_confidence099_a7_p100_seed20260902.json`

Jede Datei enthält neben den Einstellungen und den vollständigen Alternativen
auch das gezogene Zielgewicht. Pro Problem werden der Anfangs- und Endzustand,
alle Queries mit Antwort und Laufzeit, der Erwartungswert, die verbleibenden
Kandidaten sowie deren absolute und normierte Volumina festgehalten. Die Datei
wurde nach jedem abgeschlossenen Problem als Checkpoint aktualisiert.
