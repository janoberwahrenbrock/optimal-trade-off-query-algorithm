# Tiefe 2 gegen Tiefe 3: Pilot mit zehn gepaarten 5D-Problemen

## Aufbau

- 5 Ziele und 10 Alternativen
- zehn verschiedene Seeds von `20260904` bis `20260913`
- identische Alternativen und Zielgewichte für beide Tiefen je Seed
- exakte Volumenwahrscheinlichkeiten
- geometrische Ratio-Intervall-Engine
- eine Schwerpunkt-Query je kanonischem Zielpaar
- lexikographische Query-Auswahl nach `(E_d, ..., E_1)`

Die unabhängigen Probleme wurden in Vierergruppen parallel ausgeführt. Das
ändert die deterministischen Queryfolgen nicht, macht die dabei gemessenen
Einzellaufzeiten aber ungeeignet für einen sauberen absoluten
Laufzeitvergleich.

## Queryzahlen

| Seed | Tiefe 2 | Tiefe 3 | Differenz `T3 - T2` |
|---:|---:|---:|---:|
| 20260904 | 11 | 15 | +4 |
| 20260905 | 37 | 29 | -8 |
| 20260906 | 7 | 7 | 0 |
| 20260907 | 25 | 23 | -2 |
| 20260908 | 18 | 13 | -5 |
| 20260909 | 21 | 22 | +1 |
| 20260910 | 3 | 4 | +1 |
| 20260911 | 7 | 7 | 0 |
| 20260912 | 6 | 6 | 0 |
| 20260913 | 17 | 12 | -5 |

| Kennzahl | Tiefe 2 | Tiefe 3 |
|---|---:|---:|
| Gelöst | 10/10 | 10/10 |
| Queries gesamt | 152 | 138 |
| Queries, Mittelwert | 15,2 | 13,8 |
| Queries, Median | 14,0 | 12,5 |
| Queries, Standardabweichung | 10,55 | 8,42 |

Tiefe 3 war viermal besser, dreimal gleich und dreimal schlechter. Die
gepaarte mittlere Differenz beträgt `-1,4` Queries. Das 95-%-
Konfidenzintervall ist mit `[-3,97; +1,17]` breit und enthält null. Der
Wilcoxon-Vorzeichen-Rang-Test ergibt `p = 0,3125`; der Vorzeichentest über die
sieben ungleichen Paare `p = 1,0`.

Der Pilot liefert somit einen Punktwert zugunsten von Tiefe 3, aber noch
keinen belastbaren Nachweis eines systematischen Queryvorteils.

## Laufzeit

Zwei separat ausgeführte, nicht durch andere Problemprozesse belastete
Vergleiche zeigen den deutlichen Laufzeitnachteil:

- Seed `20260904`, seriell: Tiefe 2 `8,08 s`, Tiefe 3 `242,90 s`
  (`30,1`-fach).
- Seed `20260905`, jeweils vier parallele Root-Worker: Tiefe 2 `12,71 s`,
  Tiefe 3 `169,83 s` (`13,4`-fach).

Im parallel ausgeführten Zehnerblock betrug das Verhältnis der summierten
Prozesswandzeiten `16,9`. Wegen CPU-Konkurrenz ist dieser Wert nur als
Größenordnung zu verstehen.

## Fazit

Bei 5D sieht das Queryverhalten bislang günstiger für Tiefe 3 aus als bei 3D,
aber wesentlich uneinheitlicher. Die aktuelle Stichprobe reicht nicht aus, um
einen systematischen Vorteil zu belegen. Der Laufzeitnachteil von Tiefe 3 ist
dagegen bereits eindeutig und liegt je nach Ausführungsmodus ungefähr im
Bereich Faktor 13 bis 30.

Ein vollständiger 50er-Block würde mit der aktuellen Engine und vier
gleichzeitigen Problemprozessen anhand des gemessenen Zehnerblocks ungefähr
eine weitere Stunde benötigen. Vor einer solchen Vergrößerung ist es sinnvoll,
die Tiefe-3-Rekursion beziehungsweise die Wiederverwendung von Teilzuständen
zu optimieren.

## Rohdaten

Die gepaarten Dateien liegen unter
`multistep/data/exact_depth{2,3}_g5_lexicographic_seed20260904.json` bis
`...seed20260913.json`.
