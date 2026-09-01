# Replikation: Tiefe 2 gegen Tiefe 3 in 100 gepaarten 3D-Problemen

## Aufbau

Die zweite Stichprobe verwendet denselben Versuchsaufbau wie die erste:

- 3 Ziele und 10 Alternativen
- exakte Volumenwahrscheinlichkeiten
- geometrische Ratio-Intervall-Engine
- eine Schwerpunkt-Query je kanonischem Zielpaar
- lexikographische Query-Auswahl nach `(E_d, ..., E_1)`

Nur der Seed wurde von `20260902` auf `20260903` geändert. Alle 50 Probleme
des zweiten Blocks sind von den Problemen des ersten Blocks verschieden.
Innerhalb eines Blocks wurden Tiefe 2 und Tiefe 3 jeweils mit exakt denselben
Alternativen und Zielgewichten ausgeführt.

## Zweiter Block: 50 neue Probleme

| Kennzahl | Tiefe 2 | Tiefe 3 |
|---|---:|---:|
| Gelöst | 50/50 | 50/50 |
| Queries gesamt | 274 | 291 |
| Queries, Mittelwert | 5,48 | 5,82 |
| Queries, Median | 5,0 | 4,5 |
| Queries, Standardabweichung | 3,57 | 3,71 |
| Queries, Minimum / Maximum | 0 / 17 | 0 / 17 |
| Laufzeit gesamt | 25,357 s | 124,243 s |
| Laufzeit, Mittelwert | 0,507 s | 2,485 s |

Die gepaarte Differenz `Tiefe 3 - Tiefe 2` beträgt im Mittel `+0,34`
Queries. Tiefe 3 war in 14 Problemen besser, in 15 gleich und in 21
schlechter. Das 95-%-Konfidenzintervall des gepaarten Mittelwerts ist
`[-0,239; +0,919]`. Der zweite Block allein ist damit statistisch nicht
eindeutig (`Wilcoxon p = 0,191`, Vorzeichentest `p = 0,311`), zeigt aber
dieselbe Richtung wie der erste Block.

## Beide Blöcke zusammen: 100 Probleme

| Kennzahl | Tiefe 2 | Tiefe 3 |
|---|---:|---:|
| Gelöst | 100/100 | 100/100 |
| Queries gesamt | 498 | 545 |
| Queries, Mittelwert | 4,98 | 5,45 |
| Queries, Median | 4 | 4 |
| Queries, Standardabweichung | 3,16 | 3,33 |
| Laufzeit gesamt | 54,071 s | 268,065 s |
| Laufzeit, Mittelwert | 0,541 s | 2,681 s |

Die kombinierte gepaarte Differenz beträgt `+0,47` Queries mit einem
95-%-Konfidenzintervall von `[+0,086; +0,854]`. Tiefe 3 war in 24 Problemen
besser, in 33 gleich und in 43 schlechter. Der Wilcoxon-Vorzeichen-Rang-Test
ergibt `p = 0,00844`, der reine Vorzeichentest über die 67 ungleichen Paare
`p = 0,0271`.

Die Tiefe-3-Läufe benötigten zusammen rund das `4,96`-Fache der Laufzeit.

## Schlussfolgerung

Der zweite Block repliziert die Richtung des ersten Ergebnisses, auch wenn er
für sich allein nicht signifikant ist. In der kombinierten Stichprobe spricht
die Evidenz dagegen, dass das höhere Query-Mittel von Tiefe 3 lediglich eine
Besonderheit der ersten 50 Probleme war.

Das Ergebnis gilt für den untersuchten Problemgenerator und die aktuelle
Zielfunktion. Die Zielfunktion minimiert die erwartete Kandidatenzahl nach
einem festen Horizont und nicht die erwartete Anzahl der Queries bis zur
Terminierung. Eine monotone Verbesserung der realisierten Queryzahl mit
größerer Tiefe folgt daraus nicht.

## Rohdaten des zweiten Blocks

- `multistep/data/exact_depth2_g3_50_lexicographic_seed20260903.json`
- `multistep/data/exact_depth3_g3_50_lexicographic_seed20260903.json`
