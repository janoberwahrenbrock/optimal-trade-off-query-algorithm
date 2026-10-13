# Feste Tiefe 2: tatsächlicher 95%- gegenüber 99%-Abbruch

Alle 200 vorgeschriebenen Jobs sind beendet und unabhängig geprüft. Dieselben 100 Probleme mit fünf Zielen, fünf Alternativen, Grid5 plus Ratios, fester Tiefe 2, acht Workern auf Bench i7, exact_volume und einem nativen Thread pro Prozess. Vorhandene Grenze: 100 beantwortete Fragen; kein Zeitlimit.

Ausschließlich der tatsächliche Konfidenzabbruch unterscheidet sich: 95 % gegenüber 99 %. Beide Modi beenden außerdem bei genau einem Kandidaten. Die feste Tiefe-2-Querybewertung bleibt identisch; es gibt weder adaptive Auswahl noch Konfidenzabbruch in der hypothetischen Planung. Kandidatenvolumina an tatsächlichen Zuständen werden in beiden Modi berechnet.

## Qualität: alle 100 Probleme pro Modus

| Modus | Zurückgegeben | Korrekt /100 | Falsch /100 | Ungelöst bei 100 Fragen | Fehler/Timeout | Ohne akzeptiertes Resultat |
|---|---:|---:|---:|---:|---:|---:|
| 99 % | 100 | 100 | 0 | 0 | 0 | 0 |
| 95 % | 100 | 99 | 1 | 0 | 0 | 0 |

## Primäre Kosten aller zurückgegebenen Resultate

Falsche Auswahlen und zurückgegebene ungelöste Fälle werden hier ausdrücklich mitgezählt. Bei unterschiedlicher Rückgabezahl darf diese Tabelle nicht als vollständiger gepaarter Vergleich gelesen werden. Die Qualitätsnenner bleiben jeweils 100.

| Modus | n | Solverzeit gesamt s | Mittel s | Median s | P90 s | Minimum s | Maximum s | Queries gesamt | Mittel | Median | P90 | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99 % | 100 | 736,193 | 7,362 | 5,740 | 15,675 | 0,011 | 21,655 | 1.010 | 10,10 | 8,00 | 21,00 | 0,00 | 34,00 |
| 95 % | 100 | 635,741 | 6,357 | 5,033 | 13,884 | 0,011 | 20,293 | 831 | 8,31 | 7,00 | 17,10 | 0,00 | 34,00 |

## Gepaarter Kostenvergleich: 100/100 gemeinsam zurückgegebene Fälle

Diese Primärkohorte enthält auch falsche Auswahlen und zurückgegebene ungelöste Fälle. Qualitätsunterschiede stehen separat oben.

| Größe | 99 % gesamt | 95 % gesamt | Einsparung 95 % absolut | Einsparung relativ zu 99 % | Weniger bei 95 % | Gleich | Mehr bei 95 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tatsächliche Queries | 1.010 | 831 | 179 | 17,72 % | 55 | 45 | 0 |
| Solverzeit s | 736,193 | 635,741 | 100,452 | 13,64 % | 73 | 0 | 27 |

Ausgeschlossene Paarfälle ohne zwei akzeptierte Resultate: keine.
Die ergänzende gemeinsam korrekte Teilgruppe umfasst 99/100 Fälle; ihre getrennten Kostenwerte stehen in series-summary.json. Sie ersetzt nicht die Primärkohorte.

## Abbruchgründe und Gesamtrechenaufwand aller Versuche

| Modus | Abbruchgründe aller Rückgaben | Erhaltene Versuche | Beantwortete Queries aller Versuche | Dokumentierte Prozesszeit s | Versuche ohne dokumentierte Dauer |
|---|---|---:|---:|---:|---:|
| 99 % | exact_single_candidate: 41; volume_confidence: 59 | 100 | 1010 | 786,364 | 0 |
| 95 % | exact_single_candidate: 15; volume_confidence: 85 | 100 | 831 | 686,224 | 0 |

Die Prozesszeitsumme enthält ausschließlich vorhandene dokumentierte Dauern. Diese schließen zusätzlich Imports und Reportausgabe ein und können auch fehlgeschlagene oder unterbrochene Versuche umfassen, sofern deren Dauer gespeichert wurde. Fehlende Dauern sind keine Nullkosten; betroffene Versuchsidentitäten stehen pro Fall und Modus in series-summary.json. Die Summe ist bei fehlenden Dauern unvollständig und separat von der unveränderten solve_problem-Solverzeit zu interpretieren.

## Falsche Auswahlen und Nutzenverlust

Absoluter Nutzenverlust bedeutet: tatsächlicher linearer Nutzen der optimalen Alternative minus tatsächlicher Nutzen der ausgewählten Alternative, bei den gespeicherten Zielgewichten. Er ist kein Volumenanteil und keine Prozent-Fehlerquote; seine Größe hängt von der Nutzenskala ab. Die exakte rationale Differenz ist ebenfalls gespeichert. Alternative A1 entspricht dem Codeindex 0.

99 %: 0 falsche Auswahlen.

95 %: 1 falsche Auswahlen.
- Problem 81: gewählt A5, optimal A2; absoluter Nutzenverlust 0,01665408 (exakt `43236452491247316204147554146581/2596148429267413814265248164610048`); Volumenanteil gewählt 98,0819 %, wahrer Gewinner 1,9181 %; verbleibend ['A2', 'A5'].

## Alle 100 Fallpaare

Gewählt und Ziel verwenden A1 bis A5. ΔQueries = 99%-Queries minus 95%-Queries; positive Werte bedeuten weniger Fragen bei 95 %. Fehlgeschlagene Versuche zeigen nur beantwortete Teilqueries in Klammern, keine fertige Lösung. Prozesszeiten, Fehlerdetails, Nutzenverluste und Endzustände stehen vollständig in series-summary.json.

| Problem | Ziel | 99 % Status | 99 % Queries | 99 % s | 99 % gewählt | 95 % Status | 95 % Queries | 95 % s | 95 % gewählt | ΔQueries |
|---:|---|---|---:|---:|---|---|---:|---:|---|---:|
| 1 | A4 | korrekt | 14 | 9,147 | A4 | korrekt | 11 | 7,478 | A4 | 3 |
| 2 | A5 | korrekt | 27 | 14,156 | A5 | korrekt | 20 | 9,995 | A5 | 7 |
| 3 | A1 | korrekt | 12 | 11,840 | A1 | korrekt | 12 | 12,053 | A1 | 0 |
| 4 | A4 | korrekt | 17 | 16,837 | A4 | korrekt | 17 | 15,978 | A4 | 0 |
| 5 | A4 | korrekt | 9 | 4,758 | A4 | korrekt | 9 | 4,742 | A4 | 0 |
| 6 | A3 | korrekt | 5 | 2,916 | A3 | korrekt | 5 | 2,861 | A3 | 0 |
| 7 | A5 | korrekt | 1 | 1,035 | A5 | korrekt | 1 | 1,047 | A5 | 0 |
| 8 | A5 | korrekt | 10 | 5,157 | A5 | korrekt | 7 | 3,384 | A5 | 3 |
| 9 | A2 | korrekt | 3 | 2,973 | A2 | korrekt | 2 | 2,372 | A2 | 1 |
| 10 | A1 | korrekt | 22 | 10,859 | A1 | korrekt | 14 | 7,437 | A1 | 8 |
| 11 | A5 | korrekt | 4 | 4,632 | A5 | korrekt | 3 | 3,633 | A5 | 1 |
| 12 | A5 | korrekt | 2 | 1,985 | A5 | korrekt | 1 | 1,524 | A5 | 1 |
| 13 | A3 | korrekt | 8 | 6,588 | A3 | korrekt | 8 | 6,608 | A3 | 0 |
| 14 | A1 | korrekt | 21 | 21,655 | A1 | korrekt | 18 | 20,293 | A1 | 3 |
| 15 | A1 | korrekt | 12 | 10,566 | A1 | korrekt | 9 | 8,813 | A1 | 3 |
| 16 | A1 | korrekt | 3 | 2,364 | A1 | korrekt | 0 | 0,011 | A1 | 3 |
| 17 | A5 | korrekt | 13 | 8,816 | A5 | korrekt | 9 | 6,455 | A5 | 4 |
| 18 | A3 | korrekt | 20 | 15,797 | A3 | korrekt | 18 | 14,863 | A3 | 2 |
| 19 | A4 | korrekt | 3 | 3,604 | A4 | korrekt | 3 | 3,634 | A4 | 0 |
| 20 | A5 | korrekt | 16 | 9,631 | A5 | korrekt | 16 | 9,681 | A5 | 0 |
| 21 | A2 | korrekt | 5 | 3,002 | A2 | korrekt | 3 | 1,866 | A2 | 2 |
| 22 | A1 | korrekt | 7 | 4,509 | A1 | korrekt | 2 | 1,961 | A1 | 5 |
| 23 | A1 | korrekt | 16 | 8,798 | A1 | korrekt | 14 | 7,801 | A1 | 2 |
| 24 | A2 | korrekt | 9 | 4,381 | A2 | korrekt | 9 | 4,398 | A2 | 0 |
| 25 | A2 | korrekt | 2 | 2,170 | A2 | korrekt | 2 | 2,130 | A2 | 0 |
| 26 | A5 | korrekt | 1 | 0,892 | A5 | korrekt | 1 | 0,889 | A5 | 0 |
| 27 | A3 | korrekt | 7 | 7,962 | A3 | korrekt | 7 | 7,907 | A3 | 0 |
| 28 | A3 | korrekt | 34 | 19,892 | A3 | korrekt | 34 | 20,225 | A3 | 0 |
| 29 | A2 | korrekt | 7 | 5,199 | A2 | korrekt | 5 | 3,947 | A2 | 2 |
| 30 | A2 | korrekt | 16 | 9,491 | A2 | korrekt | 11 | 6,652 | A2 | 5 |
| 31 | A2 | korrekt | 4 | 1,829 | A2 | korrekt | 2 | 1,224 | A2 | 2 |
| 32 | A5 | korrekt | 17 | 11,515 | A5 | korrekt | 4 | 4,938 | A5 | 13 |
| 33 | A2 | korrekt | 8 | 4,675 | A2 | korrekt | 7 | 4,075 | A2 | 1 |
| 34 | A2 | korrekt | 25 | 15,661 | A2 | korrekt | 17 | 11,675 | A2 | 8 |
| 35 | A2 | korrekt | 5 | 4,892 | A2 | korrekt | 3 | 3,855 | A2 | 2 |
| 36 | A3 | korrekt | 13 | 8,592 | A3 | korrekt | 13 | 8,565 | A3 | 0 |
| 37 | A5 | korrekt | 16 | 16,127 | A5 | korrekt | 16 | 16,207 | A5 | 0 |
| 38 | A3 | korrekt | 12 | 9,125 | A3 | korrekt | 7 | 6,366 | A3 | 5 |
| 39 | A2 | korrekt | 3 | 2,653 | A2 | korrekt | 2 | 1,772 | A2 | 1 |
| 40 | A4 | korrekt | 7 | 4,830 | A4 | korrekt | 6 | 4,307 | A4 | 1 |
| 41 | A5 | korrekt | 3 | 2,736 | A5 | korrekt | 2 | 2,096 | A5 | 1 |
| 42 | A3 | korrekt | 4 | 3,664 | A3 | korrekt | 4 | 3,632 | A3 | 0 |
| 43 | A4 | korrekt | 5 | 3,818 | A4 | korrekt | 5 | 3,877 | A4 | 0 |
| 44 | A1 | korrekt | 6 | 4,194 | A1 | korrekt | 3 | 2,585 | A1 | 3 |
| 45 | A2 | korrekt | 15 | 16,642 | A2 | korrekt | 11 | 13,866 | A2 | 4 |
| 46 | A1 | korrekt | 13 | 9,419 | A1 | korrekt | 10 | 8,058 | A1 | 3 |
| 47 | A1 | korrekt | 3 | 2,503 | A1 | korrekt | 3 | 2,551 | A1 | 0 |
| 48 | A5 | korrekt | 8 | 6,347 | A5 | korrekt | 8 | 6,339 | A5 | 0 |
| 49 | A4 | korrekt | 7 | 5,504 | A4 | korrekt | 7 | 5,458 | A4 | 0 |
| 50 | A4 | korrekt | 4 | 2,524 | A4 | korrekt | 4 | 2,512 | A4 | 0 |
| 51 | A2 | korrekt | 11 | 9,993 | A2 | korrekt | 10 | 9,130 | A2 | 1 |
| 52 | A4 | korrekt | 7 | 5,546 | A4 | korrekt | 6 | 5,003 | A4 | 1 |
| 53 | A4 | korrekt | 10 | 13,733 | A4 | korrekt | 10 | 13,755 | A4 | 0 |
| 54 | A2 | korrekt | 11 | 6,036 | A2 | korrekt | 10 | 5,372 | A2 | 1 |
| 55 | A1 | korrekt | 1 | 1,008 | A1 | korrekt | 1 | 1,011 | A1 | 0 |
| 56 | A4 | korrekt | 15 | 9,623 | A4 | korrekt | 2 | 3,380 | A4 | 13 |
| 57 | A5 | korrekt | 22 | 12,909 | A5 | korrekt | 22 | 12,909 | A5 | 0 |
| 58 | A5 | korrekt | 7 | 6,225 | A5 | korrekt | 5 | 4,468 | A5 | 2 |
| 59 | A5 | korrekt | 7 | 6,133 | A5 | korrekt | 7 | 6,178 | A5 | 0 |
| 60 | A5 | korrekt | 7 | 5,007 | A5 | korrekt | 7 | 4,909 | A5 | 0 |
| 61 | A5 | korrekt | 16 | 13,966 | A5 | korrekt | 16 | 14,046 | A5 | 0 |
| 62 | A1 | korrekt | 12 | 6,955 | A1 | korrekt | 8 | 4,126 | A1 | 4 |
| 63 | A4 | korrekt | 3 | 2,711 | A4 | korrekt | 3 | 2,761 | A4 | 0 |
| 64 | A4 | korrekt | 3 | 1,742 | A4 | korrekt | 2 | 1,322 | A4 | 1 |
| 65 | A3 | korrekt | 2 | 1,281 | A3 | korrekt | 2 | 1,319 | A3 | 0 |
| 66 | A2 | korrekt | 19 | 12,671 | A2 | korrekt | 19 | 12,683 | A2 | 0 |
| 67 | A1 | korrekt | 27 | 16,598 | A1 | korrekt | 23 | 14,754 | A1 | 4 |
| 68 | A4 | korrekt | 7 | 5,520 | A4 | korrekt | 5 | 4,180 | A4 | 2 |
| 69 | A1 | korrekt | 9 | 11,821 | A1 | korrekt | 9 | 11,853 | A1 | 0 |
| 70 | A5 | korrekt | 8 | 6,206 | A5 | korrekt | 8 | 6,198 | A5 | 0 |
| 71 | A4 | korrekt | 11 | 7,293 | A4 | korrekt | 11 | 7,301 | A4 | 0 |
| 72 | A3 | korrekt | 7 | 4,072 | A3 | korrekt | 6 | 3,543 | A3 | 1 |
| 73 | A3 | korrekt | 21 | 10,864 | A3 | korrekt | 21 | 10,801 | A3 | 0 |
| 74 | A1 | korrekt | 9 | 7,047 | A1 | korrekt | 8 | 6,466 | A1 | 1 |
| 75 | A4 | korrekt | 14 | 9,489 | A4 | korrekt | 8 | 6,468 | A4 | 6 |
| 76 | A4 | korrekt | 9 | 5,814 | A4 | korrekt | 8 | 5,062 | A4 | 1 |
| 77 | A2 | korrekt | 21 | 19,618 | A2 | korrekt | 14 | 15,403 | A2 | 7 |
| 78 | A4 | korrekt | 11 | 9,369 | A4 | korrekt | 11 | 9,494 | A4 | 0 |
| 79 | A2 | korrekt | 8 | 5,547 | A2 | korrekt | 8 | 5,533 | A2 | 0 |
| 80 | A3 | korrekt | 6 | 4,136 | A3 | korrekt | 6 | 4,124 | A3 | 0 |
| 81 | A2 | korrekt | 17 | 10,130 | A2 | falsche Auswahl | 10 | 5,548 | A5 | 7 |
| 82 | A2 | korrekt | 2 | 1,696 | A2 | korrekt | 2 | 1,733 | A2 | 0 |
| 83 | A1 | korrekt | 5 | 4,168 | A1 | korrekt | 1 | 1,843 | A1 | 4 |
| 84 | A4 | korrekt | 4 | 2,629 | A4 | korrekt | 2 | 1,930 | A4 | 2 |
| 85 | A4 | korrekt | 14 | 12,738 | A4 | korrekt | 11 | 11,292 | A4 | 3 |
| 86 | A4 | korrekt | 30 | 17,008 | A4 | korrekt | 30 | 17,100 | A4 | 0 |
| 87 | A5 | korrekt | 4 | 2,290 | A5 | korrekt | 4 | 2,306 | A5 | 0 |
| 88 | A4 | korrekt | 32 | 18,281 | A4 | korrekt | 31 | 17,886 | A4 | 1 |
| 89 | A1 | korrekt | 4 | 2,787 | A1 | korrekt | 0 | 0,011 | A1 | 4 |
| 90 | A1 | korrekt | 14 | 11,336 | A1 | korrekt | 13 | 10,704 | A1 | 1 |
| 91 | A2 | korrekt | 2 | 1,275 | A2 | korrekt | 1 | 0,775 | A2 | 1 |
| 92 | A3 | korrekt | 7 | 5,616 | A3 | korrekt | 7 | 5,603 | A3 | 0 |
| 93 | A1 | korrekt | 15 | 11,869 | A1 | korrekt | 7 | 6,949 | A1 | 8 |
| 94 | A1 | korrekt | 11 | 5,666 | A1 | korrekt | 9 | 4,474 | A1 | 2 |
| 95 | A3 | korrekt | 3 | 2,066 | A3 | korrekt | 2 | 1,488 | A3 | 1 |
| 96 | A3 | korrekt | 0 | 0,011 | A3 | korrekt | 0 | 0,011 | A3 | 0 |
| 97 | A3 | korrekt | 3 | 3,579 | A3 | korrekt | 3 | 3,771 | A3 | 0 |
| 98 | A4 | korrekt | 1 | 1,136 | A4 | korrekt | 0 | 0,011 | A4 | 1 |
| 99 | A2 | korrekt | 6 | 5,170 | A2 | korrekt | 6 | 5,230 | A2 | 0 |
| 100 | A5 | korrekt | 11 | 8,946 | A5 | korrekt | 11 | 8,926 | A5 | 0 |

## Unabhängige Prüfung und Quellenidentität

Die 99%-Ergebnisse stimmen in 100/100 geprüften Fällen mit der eingefrorenen festen Tiefe-2-Referenz semantisch exakt überein. Gemeinsame Query-, Antwort-, Bewertungs- und Zustandspräfixe stimmen in 100/100 Fällen exakt überein. Die vollständige 95%-Trajektorie bis zum 95%-Abbruch ist in 100 Fällen vom 99%-Lauf abgedeckt.

Alle 92 Python-Quelldateien, das Quellarchiv und die eingefrorenen Eingaben sind unverändert geprüft. Algorithmus-Fingerprint: `1e534ba5d86b997b3a17399f3ef61534b2fb1961acbb95b112646def4aeff1ab`. Eingabe-SHA256: `a78c365f182479cc4092a17ee7bfcc5416dfe2da6ae4acb3e9722a5ed00f4857`. Der Controller bestätigt die unveränderten Quellhashes nach Abschluss. Zustands- und Audithashes sind in series-summary.json gespeichert.

Fehler und ungelöste Fälle: keine.

## Grenzen

- Eine frische Ausführung pro Problem und Modus mit wechselnder Reihenfolge; keine wiederholten Laufzeitmessungen.
- Dieselben Probleme wie in früheren Entwicklungsvergleichen; kein unabhängiger Holdout.
- Solverzeit enthält tatsächliche Zustandsvolumina, Planung, Poolstart/-ende und identische Beobachter; Imports und Ergebnisspeicherung sind ausgeschlossen.
- Falsche Auswahlen und zurückgegebene ungelöste Fälle sind in den primären Kosten enthalten. Bei Fehlern gibt es keinen abgeschlossenen Solverzeitwert; dokumentierte Prozesszeit und beantwortete Teilqueries werden separat ausgewiesen. Fehlende Prozessdauern, etwa bei unterbrochenen Versuchen, sind ausdrücklich als fehlend markiert und nicht als Nullkosten behandelt.
- Die Qualitätsnenner bleiben 100 pro Modus. Kostenmittel verwenden die ausdrücklich ausgewiesene Anzahl zurückgegebener Resultate; fehlende Ergebnisse werden nicht als Nullkosten behandelt.
- Der Nutzenverlust wird aus den gespeicherten binären Eingabewerten mit exakten rationalen Operationen bestimmt. LPs, Polytopevolumina und ungespeicherte Querykandidaten werden vom unabhängigen Audit nicht neu berechnet.
- Ein Konfidenzanteil ist ein Volumenanteil im verbleibenden Gewichtsraum und keine garantierte Trefferquote für jeden einzelnen Fall.

## Artefakte

Die vollständigen Rohresultate, fsynced Queryjournale, Eingaben, Quelle, Versuche und Prüfnachweise liegen unter `/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-depth2-confidence95-vs99-problems100-grid5-ratios-workers8` und dem gleichnamigen Ordner unter `/home/cruncher/otqa-benchmarks` auf Bench i7. `final-backup-manifest.json` und die beiden Prüfbelege dokumentieren die Vollständigkeit der Kopien. Es wurde kein Commit erstellt und nichts gepusht.
