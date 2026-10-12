# Performance-Zwischenstand vom 12. Oktober 2026

Dieser Zwischenstand gehört zu `codex/vertex-support-cache`. Er enthält die
beibehaltenen Nacht- und Tagesoptimierungen S06+E12 auf Basis des bereits
gesicherten Commits `074fbf9d3c09ae1318155a15f549833df661a9b3`.
Der ursprüngliche Vergleichsstand ist
`6c48de1b9868e87eacc0a2caa4905a9e268cdc2e` auf
`codex/grid10-ratio-depth2-vs-depth3`.

## Behaltene Änderungen

Die bereits in `074fbf9` gespeicherten Änderungen verwenden vorhandene
Polytopecken für Antwortgrenzen, cachen affine Gleichungszerlegungen und
verwenden LP-Matrizen beziehungsweise das HiGHS-Setup erneut. Der Solverzustand
wird weiterhin kalt zurückgesetzt; unsichere oder inkompatible Fälle verwenden
die bisherigen Berechnungen. Exakte Volumenwahrscheinlichkeiten waren bereits
im ursprünglichen Branch enthalten.

Seitdem kamen vier beibehaltene nächtliche Stufen hinzu:

1. Weniger Kopien und Schlüsselaufbau für Constraints und Antwortzweige,
   exakte Byte-Erkennung identischer Zeilen, kompakte interne LP-Ergebnisse
   und direkte Vorbereitung sparsamer CSC-Matrizen.
2. Eine bereits gesicherte Antwortgrenze weiterverwenden, auch wenn die andere
   Grenze noch per LP bestimmt werden muss.
3. Ratiointervalle mehrerer Zielpaare gemeinsam beziehungsweise mit
   vorbereiteten Daten berechnen; die geometrischen und LP-Fallbacks bleiben.
4. Kontinuierliche LP-Ergebnisse gezielt prüfen. Der optimierte Checker wird
   nur für die geprüfte kompatible SciPy-Implementierung verwendet; sonst
   bleibt deren tatsächlicher ursprünglicher Checker aktiv. Der übernommene
   SciPy-Code wird mit seiner BSD-Lizenz ausgeliefert.

Die zusätzliche Tagesänderung E12 behandelt unendliche float64-LP-Grenzen mit
einfacheren Masken. Die Berechnungsaufgabe, numerischen Toleranzen,
Kandidatenabdeckung und Abbruchregeln wurden nicht reduziert.

## Direkter Vergleich ursprünglicher Branch gegen diesen Stand

Alle Fälle wurden frisch und nacheinander auf Bench i7 gemessen: Intel Core
i7-10700, acht physische Kerne und 16 logische CPUs. **Beide Versionen verwenden
acht Worker**, forkserver und jeweils einen Thread pro numerischer Bibliothek.
Eingaben und vollständige Konfiguration sind identisch: fünf Alternativen,
Grid 10 plus Ratios, Planungstiefe zwei, Ratio-Queries auf Tiefe eins,
exakte geometrische Volumenwahrscheinlichkeiten, maximal 100 Fragen und
Abbruch bei einem Kandidaten oder mindestens 99 % Volumen-Konfidenz.

| Ziele | Fälle | Ursprünglich, Mittelwert | Dieser Stand, Mittelwert | Weniger Laufzeit |
|---:|---:|---:|---:|---:|
| 3 | 3 | 3,110 s | 1,178 s | 62,12 % |
| 5 | 3 | 26,476 s | 8,142 s | 69,25 % |
| 7 | 2 | 269,817 s | 102,801 s | 61,90 % |

Alle 16 vollständigen Läufe lösten die acht Probleme korrekt. Jede Version
stellte insgesamt 54 Queries. Gewählte Queries, Antworten, Kandidaten,
Auswahl, Abbruch, Erwartungswerte und aufgezeichnete Kandidatenvolumina/-anteile
stimmen exakt überein, einschließlich 576 Floatwerten in Hexdarstellung.
Nicht gewählte rekursive Bäume und sämtliche Zweigwahrscheinlichkeiten wurden
in diesem Vergleich nicht zusätzlich aufgezeichnet.

Der besonders schwere dritte Sieben-Ziele-Fall wurde während der Messung auf
ausdrücklichen Nutzerwunsch ausgelassen. Sein gestarteter ursprünglicher Lauf
wurde beendet und geht nicht in diese Mittelwerte ein. Die aktuelle
Gegenmessung wurde nicht gestartet. Die Zahlen erfassen die Codeänderungen
bei gleicher Parallelisierung; die frühere Erhöhung von vier auf acht Worker
ist kein Bestandteil dieses Gewinns.

Die [kompakten Vergleichsdaten](benchmarks/20261012_original_vs_current_workers8.json)
enthalten die exakten Eingaben, Einzelresultate, Konfiguration,
Softwareumgebung, Quellenidentitäten, Vergleichsgrenzen und Prüfsummen der
ausführlichen Belege. Je Version und Fall gab es einen Durchlauf; die kleine
feste Stichprobe ist kein statistischer Signifikanznachweis.

## Nacht, Tagesversuche und weiterer Dimensionstest

Der direkte Vergleich gegen den vorherigen Commit `074fbf9` misst auf zwei
festen Zehner-Problemsätzen mit jeweils fünf Zielen und fünf Alternativen:

| Seed | Median vorher | Median S06+E12 | Weniger Laufzeit | Queries je Version |
|---:|---:|---:|---:|---:|
| 20260902 | 132,937 s | 104,872 s | 21,11 % | 75 |
| 20261012 | 121,630 s | 97,920 s | 19,49 % | 73 |

Je Problemsatz gab es vier neue Kontroll- und drei neue Endstandläufe.
Warmups und Diagnose-/Validierungsläufe sind ausgeschlossen. Dieser gemessene
Gesamtgewinn wird nicht aus den inkrementellen Stufengewinnen berechnet.
Die zusätzliche E12-Änderung wurde separat gegen S06 mit 0,89 % beziehungsweise
0,42 % weniger Zeit gemessen.

Ein eigener instrumentierter Test mit je drei Problemen für drei, fünf und
sieben Ziele löste alle neun Fälle korrekt. Mittlere Zeiten waren 0,920 s,
7,382 s und 296,079 s. Er enthält den schweren dritten Sieben-Ziele-Fall mit
687,245 s und 49 Queries. Diese Zeit gehört zum Dimensionstest, nicht zum
oben gezeigten Originalvergleich mit zwei Sieben-Ziele-Fällen. Beide Quellen
des Originalvergleichs wurden mit derselben schlankeren Replay-Routine neu
gemessen; die unterschiedlichen Messreihen werden nicht vermischt.

- [Nacht- und Tagesdaten](benchmarks/20261012_overnight_daytime_summary.json)
- [Daten des vollständigen Dimensionstests](benchmarks/20261012_current_dimension_timings.json)
- [Ausführliches historisches Journal](overnight_optimization_20261012.md)
- [Frühe Änderungen und entfernte Caches](performance_optimizations.md)

## Verworfene Versuche und Validierung

Der zusätzliche gemeinsame Geometriecache und vollständige
Wahrscheinlichkeitscache wurden wegen längerer Laufzeiten aus dem aktiven Code
entfernt. Normwiederverwendung und Worker-Chunkgrößen zwei beziehungsweise vier
brachten in der Tagesmessung ebenfalls keinen robusten Vorteil und wurden
verworfen. Das Journal und die kompakten Daten unterscheiden gemessene,
behaltene, verworfene und lediglich vorbereitete Ideen.

Der gemessene Endstand besteht aus 87 Pythondateien mit dem Quellfingerprint
`d1a36969125c2ebcbb84d4bdcebf62da91e94e76d3581f029261e1b86583c60d`.
Alle 87 Dateien wurden vor der Sicherung nochmals mit dem eingefrorenen
gemessenen Endstand verglichen und stimmen bytegenau überein. Bestehende
Nachweise: 224 Multi-Step- und sechs One-Step-Tests auf Mac und Bench sowie
nach kanonischer Integration bestanden. Für den Commit werden unveränderte
Code-Tests nicht nochmals ausgeführt; Dokumentation, kompakte JSON-Daten,
Dateiumfang und `git diff --check` werden zusätzlich geprüft.

Die ausführlichen Rohdaten, Quellarchive, Replay-Skripte und Profiling-Dateien
bleiben separat unter `outputs/benchmarks/bench-i7/` auf dem Mac und
`/home/cruncher/otqa-benchmarks/` auf Bench i7 erhalten. Die kompakten JSONs
benennen die zugehörigen relativen Artefaktpfade und SHA256-Werte; die großen
Archive werden nicht in Git aufgenommen.

Das Journal enthält historische Zustände wie „noch uncommittet“ und frühere
Experimentstatus. Sie bleiben als Verlauf erhalten; diese Übersicht und die
abschließenden Journaleinträge beschreiben den gesicherten Endstand.
