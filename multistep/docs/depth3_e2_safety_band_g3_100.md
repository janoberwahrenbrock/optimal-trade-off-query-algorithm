# Tiefe 3 mit \(E_2\)-Sicherheitsband

## Verfahren

Für jede mögliche nächste Query \(q\) werden sowohl der Zwei-Schritt-Wert
\(E_2(q)\) als auch der Drei-Schritt-Wert \(E_3(q)\) berechnet. Mit

\[
E_2^*=\min_q E_2(q)
\]

werden nur Queries zugelassen, für die

\[
E_2(q)\le E_2^*+\delta
\]

gilt. Innerhalb dieser Menge wird die Query mit dem kleinsten \(E_3\)-Wert
gewählt. Damit darf Tiefe 3 einen begrenzten kurzfristigen Nachteil in Kauf
nehmen, aber keinen beliebig großen Zwei-Schritt-Fortschritt für einen kleinen
Drei-Schritt-Vorteil opfern.

Getestet wurden zwei Ausführungsarten:

1. **Receding:** Nach jeder Antwort wird erneut der geschützte
   Tiefe-3-Schritt berechnet.
2. **Countdown:** Auf einen geschützten Tiefe-3-Schritt folgen ein normaler
   Tiefe-2- und ein normaler Tiefe-1-Schritt; danach beginnt der nächste
   Dreierblock.

Alle Antwortwahrscheinlichkeiten wurden weiterhin durch exakte
Polytope-Volumen bestimmt. Queryquelle, Kandidatenlogik und geometrische Engine
blieben unverändert.

## Delta-Pilot auf 20 Problemen

| Strategie | \(\delta=0{,}05\) | \(\delta=0{,}10\) | \(\delta=0{,}25\) |
|---|---:|---:|---:|
| geschützte Tiefe 3, Receding | 93 | 94 | 105 |
| geschützte Tiefe 3, Countdown | 91 | 91 | 91 |

Die normale Tiefe 2 benötigte auf denselben Problemen 93 Queries, die
ungeschützte Tiefe 3 dagegen 107. Receding wurde mit wachsendem Band wieder
schlechter. Countdown war in diesem Pilot gegenüber der Bandbreite wesentlich
robuster. Für die vollständige Auswertung wurde \(\delta=0{,}05\) verwendet.

## Ergebnis auf 100 gepaarten 3D-Problemen

| Strategie | Queries gesamt | Mittelwert | Median | Standardabw. | Laufzeit |
|---|---:|---:|---:|---:|---:|
| normale Tiefe 2 | 498 | 4,98 | 4 | 3,156 | 54,1 s |
| ungeschützte Tiefe 3 | 545 | 5,45 | 4 | 3,328 | 268,1 s |
| Sicherheitsband, Receding | 493 | 4,93 | 4 | 3,239 | 276,1 s |
| Sicherheitsband, Countdown | 493 | 4,93 | 4 | 3,141 | 129,7 s |

Gegenüber normaler Tiefe 2 sparten beide Guard-Varianten insgesamt fünf
Queries beziehungsweise 0,05 Queries pro Problem. Dieser kleine Unterschied
ist statistisch nicht belastbar:

| Paarvergleich, Differenz erste minus zweite Strategie | Mittel | 95-%-Bootstrap-KI | besser / gleich / schlechter | Wilcoxon \(p\) |
|---|---:|---:|---:|---:|
| Receding − Tiefe 2 | −0,05 | [−0,35; 0,25] | 27 / 57 / 16 | 0,712 |
| Countdown − Tiefe 2 | −0,05 | [−0,36; 0,27] | 27 / 54 / 19 | 0,951 |
| Countdown − Receding | 0,00 | [−0,19; 0,19] | 14 / 68 / 18 | 0,924 |

Die ersten 20 Probleme wurden zur Wahl von \(\delta\) verwendet. Auf den
verbleibenden 80 Problemen lagen Receding bei 400, Countdown bei 402 und Tiefe
2 bei 405 Queries. Auch dort war keiner der Unterschiede statistisch
signifikant.

Gegenüber der ungeschützten Tiefe 3 waren beide Guard-Varianten dagegen klar
besser: jeweils 52 Queries weniger, mit gepaartem Wilcoxon-\(p<0{,}001\).

## Beantwortung der Ausführungsfrage

Bei dem engen Band \(\delta=0{,}05\) musste der Plan für die Queryzahl nicht
konsequent als \(3\to2\to1\) ausgeführt werden. Receding und Countdown kamen
über alle 100 Probleme auf exakt dieselbe Gesamtzahl. Das Band allein beseitigte
damit den zuvor beobachteten deutlichen Replanning-Nachteil.

Countdown bleibt trotzdem die praktisch bessere Variante:

- Es war mit 129,7 s mehr als doppelt so schnell wie Receding mit 276,1 s.
- Im Delta-Pilot blieb es auch bei einem lockereren Sicherheitsband stabil,
  während Receding wieder in Richtung der schlechten ungeschützten Tiefe 3
  driftete.

Ein nachweisbarer Vorteil gegenüber Tiefe 2 wurde noch nicht gefunden. Der
aktuelle Befund lautet daher: Das Sicherheitsband macht Tiefe 3 sicherer und
beseitigt ihren bisherigen Nachteil, aber die zusätzlichen Berechnungen führen
auf diesen 100 Problemen noch nicht zu einer statistisch gesicherten Reduktion
der Queryzahl.

## Reproduzierbarkeit

- Benchmarkskript: `multistep/scripts/benchmark_depth3_e2_guard.py`
- Receding-Rohdaten:
  `multistep/data/depth3_e2_guard_receding_delta005_g3_100.json`
- Countdown-Rohdaten:
  `multistep/data/depth3_e2_guard_countdown_delta005_g3_100.json`
