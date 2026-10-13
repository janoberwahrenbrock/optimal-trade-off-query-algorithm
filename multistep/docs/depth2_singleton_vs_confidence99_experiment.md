# Tiefe 2: Einzelkandidat gegenüber 99%-Abbruch

Alle 200 vorgeschriebenen Jobs sind beendet und unabhängig geprüft. Dieselben 100 Probleme mit fünf Zielen/fünf Alternativen, Grid5 plus Ratios, feste Tiefe 2, acht Worker, exact_volume, native Threads 1. Vorhandene Grenze 100 Fragen, kein Zeitlimit. Keine Algorithmusänderung und keine Änderung der Queryplanung.

Nur die tatsächliche Abbruchregel unterscheidet sich: vorhandene 99%-Konfidenzentscheidung gegenüber ausgeschalteter Konfidenzschwelle (`None`) und genau einem verbleibenden Kandidaten. Die tatsächlichen Kandidatenvolumina werden in beiden Modi weiterhin berechnet.

## Qualität aller 100 Fälle pro Modus

| Modus | Korrekt gelöst | Falsch ausgewählt | Ungelöst bei 100 Fragen | Fehler |
|---|---:|---:|---:|---:|
| confidence99 | 100 | 0 | 0 | 0 |
| singleton | 100 | 0 | 0 | 0 |

## Laufzeit und Queries erfolgreicher Lösungen

Bei unterschiedlichen Erfolgszahlen sind diese Zeilen keine gepaarte Laufzeitrangliste. Der passende Paarvergleich folgt darunter.

| Modus | Erfolgreiche Fälle | Zeit gesamt s | Zeit Mittel s | Zeit Median s | Zeit P90 s | Queries gesamt | Queries Mittel | Queries Median | Queries Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| confidence99 | 100 | 736,571 | 7,366 | 5,733 | 15,694 | 1010 | 10,10 | 8,00 | 34 |
| singleton | 100 | 872,379 | 8,724 | 7,397 | 16,696 | 1256 | 12,56 | 10,00 | 35 |

## Direkter Paarvergleich: 100/100 gemeinsam korrekt gelöste Fälle

| Größe | 99% gesamt | Einzelkandidat gesamt | Zusatz Einzelkandidat | Einsparung durch 99% |
|---|---:|---:|---:|---:|
| Tatsächliche Queries | 1.010,000 | 1.256,000 | 246,000 | 19,59% |
| Solverzeit s | 736,571 | 872,379 | 135,809 | 15,57% |

Ausgeschlossene Paarfälle: keine.

## Alle 100 Probleme und Zusatzaufwand

Bei Fehlern stehen beantwortete Teilqueries in Klammern; das sind keine fertigen Lösungen. Prozesszeiten der Fehler und alle Versuche stehen in comparison.json.

| Problem | 99% Status | 99% Queries | 99% s | Einzelkandidat Status | Einzelkandidat Queries | Einzelkandidat s | Zusätzliche Queries bei gemeinsamem Erfolg |
|---:|---|---:|---:|---|---:|---:|---:|
| 1 | completed | 14 | 9,100 | completed | 14 | 9,102 | 0 |
| 2 | completed | 27 | 14,116 | completed | 27 | 13,974 | 0 |
| 3 | completed | 12 | 11,968 | completed | 17 | 14,670 | 5 |
| 4 | completed | 17 | 16,689 | completed | 17 | 16,255 | 0 |
| 5 | completed | 9 | 4,780 | completed | 17 | 9,124 | 8 |
| 6 | completed | 5 | 2,844 | completed | 6 | 3,500 | 1 |
| 7 | completed | 1 | 1,077 | completed | 2 | 1,476 | 1 |
| 8 | completed | 10 | 5,093 | completed | 10 | 5,158 | 0 |
| 9 | completed | 3 | 3,004 | completed | 3 | 2,994 | 0 |
| 10 | completed | 22 | 10,788 | completed | 22 | 10,821 | 0 |
| 11 | completed | 4 | 4,616 | completed | 12 | 9,886 | 8 |
| 12 | completed | 2 | 1,990 | completed | 2 | 1,984 | 0 |
| 13 | completed | 8 | 6,644 | completed | 8 | 6,627 | 0 |
| 14 | completed | 21 | 21,892 | completed | 23 | 22,845 | 2 |
| 15 | completed | 12 | 10,584 | completed | 12 | 10,566 | 0 |
| 16 | completed | 3 | 2,424 | completed | 3 | 2,431 | 0 |
| 17 | completed | 13 | 8,716 | completed | 29 | 17,051 | 16 |
| 18 | completed | 20 | 15,837 | completed | 22 | 17,105 | 2 |
| 19 | completed | 3 | 3,604 | completed | 4 | 4,084 | 1 |
| 20 | completed | 16 | 9,684 | completed | 16 | 9,723 | 0 |
| 21 | completed | 5 | 2,945 | completed | 7 | 4,144 | 2 |
| 22 | completed | 7 | 4,552 | completed | 7 | 4,550 | 0 |
| 23 | completed | 16 | 8,820 | completed | 17 | 9,624 | 1 |
| 24 | completed | 9 | 4,403 | completed | 10 | 5,135 | 1 |
| 25 | completed | 2 | 2,198 | completed | 3 | 2,582 | 1 |
| 26 | completed | 1 | 0,868 | completed | 3 | 2,030 | 2 |
| 27 | completed | 7 | 8,025 | completed | 8 | 8,630 | 1 |
| 28 | completed | 34 | 20,013 | completed | 35 | 20,688 | 1 |
| 29 | completed | 7 | 5,225 | completed | 26 | 13,425 | 19 |
| 30 | completed | 16 | 9,438 | completed | 16 | 9,439 | 0 |
| 31 | completed | 4 | 1,823 | completed | 6 | 3,042 | 2 |
| 32 | completed | 17 | 11,533 | completed | 22 | 13,918 | 5 |
| 33 | completed | 8 | 4,694 | completed | 8 | 4,689 | 0 |
| 34 | completed | 25 | 15,678 | completed | 25 | 15,568 | 0 |
| 35 | completed | 5 | 4,813 | completed | 7 | 5,947 | 2 |
| 36 | completed | 13 | 8,621 | completed | 18 | 11,362 | 5 |
| 37 | completed | 16 | 16,073 | completed | 16 | 16,100 | 0 |
| 38 | completed | 12 | 9,133 | completed | 12 | 9,147 | 0 |
| 39 | completed | 3 | 2,684 | completed | 5 | 3,697 | 2 |
| 40 | completed | 7 | 4,768 | completed | 7 | 4,774 | 0 |
| 41 | completed | 3 | 2,699 | completed | 3 | 2,657 | 0 |
| 42 | completed | 4 | 3,627 | completed | 7 | 5,115 | 3 |
| 43 | completed | 5 | 3,874 | completed | 5 | 3,879 | 0 |
| 44 | completed | 6 | 4,146 | completed | 6 | 4,177 | 0 |
| 45 | completed | 15 | 16,711 | completed | 21 | 19,681 | 6 |
| 46 | completed | 13 | 9,448 | completed | 13 | 9,388 | 0 |
| 47 | completed | 3 | 2,523 | completed | 3 | 2,542 | 0 |
| 48 | completed | 8 | 6,351 | completed | 8 | 6,326 | 0 |
| 49 | completed | 7 | 5,542 | completed | 8 | 6,184 | 1 |
| 50 | completed | 4 | 2,549 | completed | 4 | 2,529 | 0 |
| 51 | completed | 11 | 9,855 | completed | 11 | 9,901 | 0 |
| 52 | completed | 7 | 5,495 | completed | 8 | 6,237 | 1 |
| 53 | completed | 10 | 13,672 | completed | 15 | 16,657 | 5 |
| 54 | completed | 11 | 6,008 | completed | 14 | 7,765 | 3 |
| 55 | completed | 1 | 1,009 | completed | 8 | 4,392 | 7 |
| 56 | completed | 15 | 9,619 | completed | 20 | 12,480 | 5 |
| 57 | completed | 22 | 12,850 | completed | 22 | 12,897 | 0 |
| 58 | completed | 7 | 6,273 | completed | 8 | 6,886 | 1 |
| 59 | completed | 7 | 6,131 | completed | 10 | 7,683 | 3 |
| 60 | completed | 7 | 4,978 | completed | 7 | 4,936 | 0 |
| 61 | completed | 16 | 14,093 | completed | 16 | 14,035 | 0 |
| 62 | completed | 12 | 6,937 | completed | 16 | 9,822 | 4 |
| 63 | completed | 3 | 2,713 | completed | 4 | 3,357 | 1 |
| 64 | completed | 3 | 1,774 | completed | 5 | 2,897 | 2 |
| 65 | completed | 2 | 1,283 | completed | 2 | 1,297 | 0 |
| 66 | completed | 19 | 12,710 | completed | 23 | 15,339 | 4 |
| 67 | completed | 27 | 16,564 | completed | 27 | 16,601 | 0 |
| 68 | completed | 7 | 5,496 | completed | 7 | 5,509 | 0 |
| 69 | completed | 9 | 11,881 | completed | 14 | 14,381 | 5 |
| 70 | completed | 8 | 6,226 | completed | 9 | 7,001 | 1 |
| 71 | completed | 11 | 7,257 | completed | 11 | 7,285 | 0 |
| 72 | completed | 7 | 4,120 | completed | 7 | 4,074 | 0 |
| 73 | completed | 21 | 10,839 | completed | 23 | 12,110 | 2 |
| 74 | completed | 9 | 7,084 | completed | 9 | 7,050 | 0 |
| 75 | completed | 14 | 9,524 | completed | 14 | 9,578 | 0 |
| 76 | completed | 9 | 5,758 | completed | 21 | 11,797 | 12 |
| 77 | completed | 21 | 19,540 | completed | 22 | 20,465 | 1 |
| 78 | completed | 11 | 9,397 | completed | 12 | 10,057 | 1 |
| 79 | completed | 8 | 5,531 | completed | 8 | 5,546 | 0 |
| 80 | completed | 6 | 4,111 | completed | 12 | 7,509 | 6 |
| 81 | completed | 17 | 10,169 | completed | 17 | 10,186 | 0 |
| 82 | completed | 2 | 1,706 | completed | 7 | 4,267 | 5 |
| 83 | completed | 5 | 4,177 | completed | 7 | 5,483 | 2 |
| 84 | completed | 4 | 2,640 | completed | 5 | 3,212 | 1 |
| 85 | completed | 14 | 12,832 | completed | 28 | 19,628 | 14 |
| 86 | completed | 30 | 17,018 | completed | 34 | 19,588 | 4 |
| 87 | completed | 4 | 2,295 | completed | 5 | 2,912 | 1 |
| 88 | completed | 32 | 18,198 | completed | 33 | 18,867 | 1 |
| 89 | completed | 4 | 2,801 | completed | 4 | 2,790 | 0 |
| 90 | completed | 14 | 11,276 | completed | 14 | 11,337 | 0 |
| 91 | completed | 2 | 1,283 | completed | 4 | 2,276 | 2 |
| 92 | completed | 7 | 5,650 | completed | 7 | 5,613 | 0 |
| 93 | completed | 15 | 11,873 | completed | 20 | 14,760 | 5 |
| 94 | completed | 11 | 5,709 | completed | 32 | 17,880 | 21 |
| 95 | completed | 3 | 1,994 | completed | 4 | 2,563 | 1 |
| 96 | completed | 0 | 0,012 | completed | 8 | 5,667 | 8 |
| 97 | completed | 3 | 3,736 | completed | 5 | 4,731 | 2 |
| 98 | completed | 1 | 1,164 | completed | 5 | 3,312 | 4 |
| 99 | completed | 6 | 5,149 | completed | 18 | 11,805 | 12 |
| 100 | completed | 11 | 8,935 | completed | 12 | 9,614 | 1 |

## Prüfung und Interpretation

Die 99%-Ergebnisse stimmen mit der früheren festen Tiefe-2-Referenz in 100/100 Fällen semantisch exakt überein. Gemeinsame Query-/Antwort-/Bewertungs-/Zustandsprefixe stimmen in 100/100 Fällen exakt überein; vollständige frühere 99%-Trajektorie abgedeckt in 100 Fällen.

Diese Reihe misst den Einfluss des tatsächlichen Konfidenzabbruchs bei unveränderter fester Querystrategie. Die Singleton-Definition verwendet die bestehende numerische Kandidatenklassifikation. Ausbleibende falsche Auswahlen in diesem Kollektiv garantieren keine allgemeine Fehlerfreiheit einer 99%-Konfidenzregel.

Die früheren Tiefe-1-Ausfälle wurden mit 99%-Abbruch gemessen. Eine reine Singleton-Tiefe-1-Erfolgsquote wurde in dieser Reihe nicht untersucht.

Die Konfidenz-Volumenberechnung wurde für den isolierten Vergleich bewusst auch im Singleton-Modus belassen. Ein zusätzlicher Performanceversuch könnte später deren Weglassen im Singleton-Modus prüfen; er wäre eine andere Implementierungsvariante.

Rohdaten, fsynced Fragenjournale, Eingaben, eingefrorene Quelle, alle Versuche, unabhängige Prüfungen und finale Hashbelege liegen auf Mac und Bench. Laufzeiten sind komplette solve_problem-Timer und enthalten Poolstart/-ende und identische Beobachter. Fehlerprozesszeiten sind separat. Keine cProfile-Zeiten als Laufzeitmessung.

## Grenzen

- One fresh solve per case/mode with alternating order, not repeated timing trials.
- Only the actual confidence threshold differs; both retain actual-state volume calculations.
- Same inputs as earlier experiments; no independent holdout.
- Candidate singleton is the existing numerical classification, not symbolic reconstruction.
- Unsolved cases and errors remain in denominator100; paired solve-time comparison excludes them explicitly.
- Rational utility/answer audit does not independently solve LPs or reconstruct polytopes.
