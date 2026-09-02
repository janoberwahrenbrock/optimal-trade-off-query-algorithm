# Dimension 7: Grid 10 + Ratio, Tiefe 2, Volumenkonfidenz 0,99

## Konfiguration

- 7 Ziele
- 3, 5 und 7 erzeugte Alternativen
- jeweils 10 deterministisch reproduzierbare Probleme
- Lookahead-Tiefe 2
- Grid 10 + Ratio ab Tiefe 2
- Ratio auf Tiefe 1
- exakte Volumenwahrscheinlichkeiten
- Volumen-Konfidenzterminierung bei 0,99
- 4 persistente Root-Worker
- maximal 100 Queries pro Problem
- Seed 20260902

## Zusammenfassung

| Alternativen | Gelöst | Korrekt | Queries gesamt | Queries Mittel | Median | Min–Max | Zeit gesamt | Zeit Mittel | Exakt | Konfidenz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 10/10 | 10/10 | 216 | 21,6 | 20,5 | 2–68 | 5167,79 s | 516,78 s | 2 | 8 |
| 5 | 10/10 | 10/10 | 231 | 23,1 | 14,0 | 6–54 | 5923,35 s | 592,34 s | 0 | 10 |
| 7 | 10/10 | 10/10 | 143 | 14,3 | 13,5 | 4–23 | 4361,68 s | 436,17 s | 1 | 9 |
| **Gesamt** | **30/30** | **30/30** | **590** | **19,67** | **15,5** | **2–68** | **15452,82 s** | **515,09 s** | **3** | **27** |

Die erfolgreiche Rechenzeit aller 30 Probleme beträgt 4 h 17 min 33 s. Ein
Problem benötigte im Mittel 8 min 35 s und im Median 6 min 40 s. Die mittlere
Zeit pro beantworteter Query beträgt 26,19 s.

Die mittlere Zahl initial geometrisch relevanter Kandidaten betrug 3,0 bei 3,
4,7 bei 5 und 6,7 bei 7 erzeugten Alternativen. Dominierte Alternativen haben
von Beginn an kein positives Optimalitätsvolumen.

## Query-Verteilungen

- 3 Alternativen: 2×1, 6×1, 9×1, 14×1, 20×1, 21×1, 23×1, 24×1, 29×1, 68×1 Probleme.
- 5 Alternativen: 6×1, 9×2, 10×1, 12×1, 16×1, 18×1, 48×1, 49×1, 54×1 Probleme.
- 7 Alternativen: 4×1, 8×2, 11×1, 12×1, 15×1, 20×2, 22×1, 23×1 Probleme.

Die Notation `q×n` bedeutet: `n` Probleme benötigten jeweils `q` Queries.
Die Streuung ist sehr groß: Die Standardabweichung beträgt 15,77 Queries. Der
schwierigste Lauf benötigte 68 Queries und 2358,45 s, der leichteste nur 2
Queries und 17,90 s.

Die zehn Probleme mit 7 erzeugten Alternativen waren in dieser Stichprobe
leichter als die Blöcke mit 3 oder 5 Alternativen. Daraus folgt nicht, dass mehr
Alternativen die Lösung vereinfachen: Die Blöcke enthalten unterschiedliche
Zufallsprobleme und umfassen jeweils nur zehn Fälle. Der belastbare Befund ist
die hohe Varianz zwischen konkreten Problemgeometrien.

## Vergleich der Dimensionen

| Dimension | Gelöst und korrekt | Queries Mittel | Queries Median | Zeit Mittel | Zeit Median | Zeit pro Query | Gesamtzeit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 30/30 | 3,83 | 3,0 | 4,45 s | 4,19 s | 1,16 s | 133,60 s |
| 5 | 30/30 | 8,33 | 7,0 | 44,28 s | 36,03 s | 5,31 s | 1328,28 s |
| 7 | 30/30 | 19,67 | 15,5 | 515,09 s | 399,89 s | 26,19 s | 15452,82 s |

Gegenüber Dimension 5 benötigt Dimension 7 im Mittel 2,36-mal so viele
Queries und 11,63-mal so viel Zeit pro Problem. Eine einzelne Query ist im
Mittel 4,93-mal so teuer. Gegenüber Dimension 3 steigt die Problemzeit um den
Faktor 115,7.

Die Laufzeit wächst somit aus zwei Gründen:

1. Der Algorithmus benötigt in Dimension 7 deutlich mehr Queries.
2. Jede Query ist wegen der exakten Volumenberechnungen in der komplexeren
   7D-Geometrie erheblich teurer.

## Terminierung und Lösungsgüte

- 27 Probleme endeten bei mindestens 99 % Volumenanteil.
- 3 Probleme endeten mit genau einem geometrisch möglichen Kandidaten.
- Kein Problem erreichte das Limit von 100 Queries.
- Alle 30 ausgewählten Kandidaten waren die tatsächlichen Gewinner.
- Volumenanteile sind entlang eines einzelnen Antwortpfads nicht monoton. In
  mehreren schwierigen Fällen fiel ein Kandidat nach mehr als 98 % zeitweise
  wieder deutlich zurück, bevor die korrekte Terminierung erreicht wurde.

Die Ergebnisse zeigen damit, dass Dimension 7 funktional lösbar ist. Die
aktuelle Methode besitzt aber eine schwere Laufzeitverteilung: Einzelne
Grenzfälle benötigen 48 bis 68 Queries und 23 bis 39 Minuten. Für planbare
Produktivlaufzeiten reicht der Mittelwert daher nicht aus.

## Numerische Robustheit

Die Analyse deckte drei reproduzierbare numerische Grenzfälle auf:

1. Ein nicht gesetzter HiGHS-Status in einem tiefen LP-Zweig.
2. Qhull-Ecken mit skalenbedingt zu großen absoluten Residuen.
3. Nahezu degenerierte Halbräume, bei denen Qhull leicht außerhalb liegende
   Schnittpunkte lieferte.

Die Engine wurde während der Analyse um folgende sichere Fallbacks erweitert:

- HiGHS Dual-Simplex und Interior-Point nach dem normalen Retry,
- skalenrelative Validierung der Nebenbedingungen,
- Retry von formal erfolgreichen, aber ungültigen Qhull-Ergebnissen,
- affine Vorkonditionierung um Chebyshev-Zentrum und Innenradius,
- validiertes Zurücksetzen leicht gestörter Ecken auf unabhängige aktive
  Facetten.

Große Abweichungen werden weiterhin nicht akzeptiert. Nach den Änderungen
laufen 122 Tests erfolgreich. Die Teil-Checkpoints wurden deterministisch ab
dem betroffenen Problem fortgesetzt; die verworfene Retry-Rechenzeit ist nicht
in den oben angegebenen erfolgreichen Problemlaufzeiten enthalten.

## Datenqualität

Alle 590 Query-Datensätze enthalten den Erwartungswert, die verbleibenden
Kandidaten, deren Rohvolumina und deren normierte Volumenanteile. Für jedes
Problem stimmen `question_count`, die Zahl der Query-Protokolle und die Zahl
der Query-Laufzeiten überein. Der größte numerische Fehler der Summe der
Volumenanteile gegenüber 1 beträgt 2,22e-16.

## Schlussfolgerung

Dimension 7 funktioniert mit der aktuellen exakten Methode korrekt, ist aber
noch nicht schnell und vorhersehbar genug. Der wichtigste nächste
Optimierungshebel ist weiterhin die exakte Geometrie innerhalb des
Tiefe-2-Lookaheads. Zusätzlich sollte eine Query-Auswahl untersucht werden,
die bei nahezu gleichem Erwartungswert ungünstige lange Antwortpfade vermeidet.
Ein bloßes Senken der 99-%-Schwelle wäre wegen der beobachteten nicht monotonen
Volumenverläufe keine saubere allgemeine Lösung.

## Rohdaten

- `multistep/data/d7_grid10_ratio_depth2_confidence099_a3_p10_seed20260902.json`
- `multistep/data/d7_grid10_ratio_depth2_confidence099_a5_p10_seed20260902.json`
- `multistep/data/d7_grid10_ratio_depth2_confidence099_a7_p10_seed20260902.json`

Jede Datei speichert pro Problem unter `queries` für jede gestellte Query unter
anderem `expected_value`, `remaining_candidates`, `candidate_volumes` und
`candidate_volume_shares`. Der Endzustand enthält zusätzlich
`termination_reason`, `selected_candidate`, `selected_candidate_volume_share`,
`residual_volume_share`, `final_candidates` und `selection_is_correct`.
