# Pilot: Stoppzeit-Rollout gegen die Tiefe-2-Strategie

## Ziel

Die bisherige Wertfunktion minimiert die erwartete Kandidatenzahl nach einem
festen Horizont. Der Versuch ersetzt den Blattwert eines dreistufigen
Lookaheads durch die erwartete Anzahl weiterer Fragen, wenn ab dem Blatt die
unveränderte Tiefe-2-Strategie bis zur Terminierung ausgeführt wird.

Für den Rollout-Horizont \(h\) lautet die Rekursion

\[
W_0(T)=J_{\pi_2}(T),
\qquad
W_h(T)=\min_q\left(1+\sum_a p(a\mid T,q)W_{h-1}(T_{q,a})\right).
\]

Die Antwortwahrscheinlichkeiten werden weiterhin durch exakte
Polytope-Volumen berechnet. Die Tiefe-2-Query ist in jedem Zustand explizit
unter den erlaubten Queries enthalten.

## Praktische Grenze der exakten Fortsetzung

Der vollständige Entscheidungsbaum von \(\pi_2\) lässt sich nicht praktisch
bis zum Ende enumerieren. Bereits beim ersten 3D-Testproblem existierte ein
extrem schmaler Ast, der nach 50 weiteren Halbierungen noch nicht terminiert
war. Die Geometrieberechnung scheiterte dort schließlich an der numerischen
Auflösung des sehr dünnen Polytops.

Der experimentelle Code verwirft deshalb einzelne Pfade, sobald ihre vom
Ausgangszustand aus kumulierte Wahrscheinlichkeit unter einen konfigurierten
Grenzwert fällt. Die Summe der tatsächlich verworfenen Wahrscheinlichkeitsmasse
wird separat protokolliert. Diese Summe kann deutlich größer als der Grenzwert
eines einzelnen Pfades sein; die Berechnung ist daher ausdrücklich eine
Approximation und keine exakte Stoppzeitauswertung.

## Pilot auf dem ersten Problem des 3D-Benchmarks

Verglichen wurde dasselbe Problem aus
`exact_depth2_g3_50_lexicographic.json` und
`exact_depth3_g3_50_lexicographic.json`.

| Strategie | Fragen | Laufzeit |
|---|---:|---:|
| bisherige Tiefe 2 | 3 | 0,384 s |
| bisherige Tiefe 3 | 4 | 2,074 s |
| Stoppzeit-Rollout, Horizont 1, Pfadgrenze \(10^{-2}\) | 3 | 44,116 s |

Der Ein-Schritt-Rollout wich bei einer der drei tatsächlich gestellten Queries
von der Tiefe-2-Query ab, reduzierte die Gesamtzahl der Fragen aber nicht. Er
war rund 115-mal langsamer als Tiefe 2 und rund 21-mal langsamer als die
bisherige Tiefe 3. Über alle dabei bewerteten Fortsetzungen summierte sich die
größte verworfene Wahrscheinlichkeitsmasse auf 0,259. Der schnelle Pilot ist
damit zu grob, um kleine Wertunterschiede zuverlässig zu beurteilen.

Die eigentliche Drei-Schritt-Variante mit derselben bereits aggressiven
Pfadgrenze lieferte nach 90 Sekunden noch nicht einmal die erste Query und wurde
abgebrochen. Mit einer Pfadgrenze von \(10^{-3}\) war schon die
Ein-Schritt-Variante nach 60 Sekunden nicht abgeschlossen. Eine feinere
vollständig exakte Fortsetzung war nach mehr als zwei Minuten noch mit der
ersten Baseline-Auswertung beschäftigt und wurde ebenfalls abgebrochen.

## Ergebnis

Der Ansatz bildet das gewünschte Optimierungsziel korrekt ab, ist als direkt
enumerierter Rollout aber nicht konkurrenzfähig. Ein Benchmark über 100
Probleme wäre mit dieser Implementierung nicht sinnvoll. Bevor die Idee erneut
getestet wird, braucht \(J_{\pi_2}\) eine wesentlich billigere Approximation,
zum Beispiel ein einmal offline gelerntes Kostenmodell oder eine gezielte
Monte-Carlo-Schätzung mit gemeinsamen Zufallszahlen. Die bestehende Tiefe-2-
und Tiefe-3-Implementierung wurde durch den Versuch nicht verändert.
