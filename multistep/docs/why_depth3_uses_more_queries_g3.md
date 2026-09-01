# Warum Tiefe 3 in 3D mehr reale Queries benötigt

## Ausgangspunkt

Bei identischen Querymengen ist die intuitive Monotonieaussage für die
implementierte Wertfunktion korrekt:

\[
V_3(T) \leq V_2(T).
\]

Ein zusätzlicher simulierter Schritt kann die minimale erwartete Zahl der am
Horizont verbleibenden Kandidaten nicht erhöhen. Das ist jedoch eine Aussage
über den Horizontwert und nicht über die reale Anzahl der Queries bis zur
Terminierung.

## Direkte Prüfung der Wertfunktion

Für die 100 gepaarten 3D-Probleme wurden die initialen Werte erneut exakt
berechnet:

- In 96 Problemen galt streng `V_3 < V_2`.
- In 4 bereits trivialen beziehungsweise wertgleichen Problemen galt
  `V_3 = V_2`.
- In keinem Problem galt `V_3 > V_2`.
- Im Mittel betrug `V_3 - V_2 = -0,443` Kandidaten.

Tiefe 3 optimiert ihre definierte Horizont-Zielfunktion also tatsächlich
besser als Tiefe 2.

## Zwei verschiedene Zielgrößen

Die aktuelle Wertfunktion minimiert

\[
V_h(T)
=
\min_q \mathbb{E}\left[V_{h-1}(T_q)\right],
\]

also die erwartete Kandidatenzahl nach einem festen Horizont. Gemessen wird im
End-to-End-Benchmark dagegen die Stoppzeit

\[
\tau
=
\min\{t: |K(T_t)|=1\}.
\]

Aus einem kleineren Wert von `V_3` folgt nicht, dass die realisierte oder
erwartete Stoppzeit kleiner ist. Eine Query kann im Mittel nach drei Schritten
weniger Kandidaten übrig lassen und für ein bestimmtes Zielgewicht trotzdem
einen längeren Antwortpfad erzeugen.

## Der zusätzliche Receding-Horizon-Effekt

Die Bewertung einer ersten Tiefe-3-Query nimmt folgende Fortsetzung an:

\[
3 \longrightarrow 2 \longrightarrow 1.
\]

Nach einer realen Antwort führt die aktuelle End-to-End-Ausführung jedoch
nicht mit Tiefe 2 fort. Sie setzt den Horizont wieder auf 3:

\[
3 \longrightarrow 3 \longrightarrow 3 \longrightarrow \dots
\]

Damit wird der Entscheidungsbaum, auf dessen Grundlage `E_3` berechnet wurde,
nicht ausgeführt. Der Algorithmus bewertet eine abnehmende Resttiefe, handelt
aber mit einem bei jeder Query erneuerten Horizont.

Auf den 100 gespeicherten Tiefe-3-Pfaden gab es 449 Übergänge, für die eine
Folgequery vorlag. In 172 Fällen (`38,3 %`) unterschied sich die erneut mit
Tiefe 3 gewählte Query von der Tiefe-2-Fortsetzung, die der vorherige
Tiefe-3-Baum angenommen hatte. 72 von 94 nichttrivialen Problemen enthielten
mindestens eine solche Abweichung.

Tiefe 2 besitzt denselben Mechanismus: Von 402 Übergängen wich die erneute
Tiefe-2-Planung in 156 Fällen (`38,8 %`) von der zuvor angenommenen
Tiefe-1-Fortsetzung ab. 64 von 92 nichttrivialen Problemen waren betroffen.
Der Unterschied liegt daher nicht primär in der Häufigkeit der Neuplanung,
sondern in ihren Folgen. Tiefe 3 kann einen kurzfristig schlechteren Schritt
zugunsten eines Vorteils am dritten Horizontschritt wählen. Nach dem Reset
liegt dieser dritte Schritt erneut drei Schritte entfernt.

## Countdown-Gegenexperiment

Als Gegenprobe wurden auf denselben 100 Problemen zwei Blockausführungen
verwendet:

\[
2 \to 1 \to 2 \to 1 \to \dots
\]

und

\[
3 \to 2 \to 1 \to 3 \to 2 \to 1 \to \dots
\]

Sie berechnet die Query in jedem erreichten Zustand weiterhin exakt neu, hält
aber die Resttiefe des gerade bewerteten Blocks ein.

| Ausführung | Queries gesamt | Mittelwert | Median |
|---|---:|---:|---:|
| konstante Tiefe 2 | 498 | 4,98 | 4 |
| konstante Tiefe 3 | 545 | 5,45 | 4 |
| Countdown `2,1` | 493 | 4,93 | 4 |
| Countdown `3,2,1` | 506 | 5,06 | 4 |

Gegenüber der konstanten Tiefe 3 spart der Countdown im Mittel `0,39`
Queries. Das 95-%-Konfidenzintervall beträgt `[-0,651; -0,129]`; der
Wilcoxon-Test ergibt `p = 0,00489`. Der Countdown war in 33 Problemen besser,
in 51 gleich und in 16 schlechter.

Im fairen Vergleich der beiden Countdown-Ausführungen beträgt die mittlere
Differenz `Countdown 3,2,1 - Countdown 2,1` nur `+0,13` Queries. Das
95-%-Konfidenzintervall ist `[-0,175; +0,435]`; der Wilcoxon-Test ergibt
`p = 0,396`. Die beiden Countdown-Verfahren sind in dieser Stichprobe
bezüglich der Queryzahl nicht unterscheidbar.

Der Reset erhöht die Gesamtzahl bei Tiefe 2 gegenüber ihrem Countdown nur von
493 auf 498, also um 5 Queries. Bei Tiefe 3 steigt sie dagegen von 506 auf 545,
also um 39 Queries. Beide Tiefen ändern ihre Fortsetzung ähnlich häufig, aber
die Änderungen der Tiefe-3-Politik sind für die Stoppzeit wesentlich teurer.

Auch die Laufzeit sinkt: Der Countdown benötigte für die 100 Probleme rund
`111 s`, die konstante Tiefe 3 rund `268 s`. Tiefe 2 lag bei rund `54 s`.

## Konkretes erstes Problem

Das erste Problem startet mit vier Kandidaten. Die konstante Tiefe 3 benötigt
vier Queries. Die Countdown-Ausführung verwendet:

1. Tiefe 3: `q=(1,2; s=1.0)`, danach drei Kandidaten.
2. Tiefe 2: `q=(0,1; s=0.666667)`, danach zwei Kandidaten.
3. Tiefe 1: `q=(0,1; s=0.262626)`, danach ein Kandidat.

Sie terminiert somit nach drei Queries. Die konstante Tiefe 3 verwirft nach
der ersten Antwort diese angenommene Fortsetzung und beginnt stattdessen mit
einer neuen Tiefe-3-Query `q=(1,2; s=3.0)`; der resultierende Pfad benötigt vier
Queries.

## Schlussfolgerung

Die Tiefe-3-Wertfunktion ist nicht schlechter. Sie erreicht auf ihrem eigenen
Kriterium durchgehend einen gleich guten oder besseren Wert. Die zusätzlichen
realen Queries entstehen aus zwei Gründen:

1. Horizont-Kandidatenzahl und Stoppzeit sind unterschiedliche Zielgrößen.
2. Die Ausführung setzt nach jeder Antwort den Horizont zurück und folgt daher
   nicht der Fortsetzung, mit der `E_3` berechnet wurde.

Der faire Countdown-Vergleich liefert keinen statistischen Hinweis darauf,
dass Tiefe 3 mehr Queries als Tiefe 2 benötigt. Der signifikante Nachteil trat
erst durch die Kombination aus größerem Horizont und vollständigem Reset nach
jeder Antwort auf. Wenn die eigentliche Zielgröße die Anzahl der Nutzerfragen
ist, sollte zusätzlich eine geeignete Schätzung der verbleibenden Stoppzeit am
Lookahead-Horizont verwendet werden.
