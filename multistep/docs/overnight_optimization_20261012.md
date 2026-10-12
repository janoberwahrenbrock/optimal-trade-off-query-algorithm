# Achtstündige Optimierung auf Bench-i7 — 12. Oktober 2026

## Auftrag und unveränderliche Bedingungen

Beginn: **01:55:53 UTC / 03:55:53 Europe/Berlin**. Geplantes Ende:
**09:55:53 UTC / 11:55:53 Europe/Berlin**. Alle Änderungen und Nachweise bleiben
während dieses Zeitraums lokal; es erfolgt kein Commit und kein Push.

Ausgangspunkt ist Commit `074fbf9d3c09ae1318155a15f549833df661a9b3`
auf `codex/vertex-support-cache`. Dessen erfolgreiche Verbesserungen vom Abend
sind in [performance_optimizations.md](performance_optimizations.md) beschrieben.
Die dort verworfenen Geometrie- und vollständigen Wahrscheinlichkeits-Caches
werden nicht erneut als vermeintlich neue Optimierung eingeführt.

Die Testkonfiguration wird für alle Laufzeitvergleiche festgehalten:

| Eigenschaft | Unveränderter Wert |
|---|---|
| Ziele / Alternativen | 5 / 5 |
| Grid | 10, logarithmisch, Bereich 0.001 bis 100 |
| Queryquellen | Wurzel: Grid **und** Ratios; Tiefe eins: Ratios |
| Planungstiefe | 2 |
| Antwortwahrscheinlichkeiten | exakte geometrische Volumenberechnung |
| Worker | 8 |
| Prozessstart | forkserver |
| Native numerische Threads je Prozess | 1 |
| Abbruch | einzelner Kandidat oder bisherige Volumenkonfidenz 0.99 |
| Höchste Fragenzahl | 100 |
| Numerik | alle Toleranzen und bisherigen Fallbacks bleiben erhalten |
| Queryauswahl | bisheriges lexikografisches Ziel und vollständiger Queryumfang |

Der Messharness speichert außerdem die **gesamte** tatsächliche
`OptimizedMultistepConfig` und deren SHA256-Fingerprint. Ein abweichender
Fingerprint verhindert einen Vergleich. Einstellungen wie Samplingzahlen werden
nicht verändert; der für diese Läufe aktive Wahrscheinlichkeitsmodus bleibt
`exact_volume`. Es wird weder Sampling anstelle von Geometrie verwendet noch die
Anzahl der Gridpunkte, Zielpaare, Queries oder Antworten reduziert.

## Maschine, Eingaben und Messverfahren

Bench-i7 (`machine`, Intel i7-10700) besitzt acht physische Kerne und 16 logische
Threads. Rechnerzustand und CPU-Governor werden nicht umgestellt. Auf der Maschine
laufen ausschließlich nacheinander ausgeführte Benchmarkbedingungen, damit sich
die Bedingungen nicht gegenseitig belasten. Bibliotheksversionen, Zeitstempel,
Ausgangslast, Prozess-IDs und Quellfingerprints werden pro Lauf protokolliert.

Der primäre Datensatz besteht aus den **zehn bereits zuvor verwendeten**
gespeicherten 5×5-Matrizen und Zielgewichten. Er wird nicht je Versuch neu gezogen.
Zusätzlich wird einmalig ein separater Zehnerdatensatz mit Seed `20261012`
erzeugt und anschließend ebenfalls unverändert wiederverwendet. Dieser zweite
Datensatz soll vermeiden, Verbesserungen ausschließlich an den bekannten
Problemen zu beurteilen.

Eine Laufzeitmessung umfasst den unveränderten vollständigen `solve_problem`-
Aufruf einschließlich Initialisierung und Beendigung des jeweiligen Workerpools.
Importe im Hauptprozess vor dem Lösungsaufruf, Datensatzerzeugung und Schreiben
der Berichte liegen außerhalb dieses Intervalls. Worker-Start und dabei anfallende
Worker-Importe sind innerhalb der gemessenen Poolinitialisierung enthalten. Der gleiche leichte Messharness wird für Ausgangsversion und Varianten
verwendet. Seine bisherige Instrumentierung misst disjunkte Hauptprozessphasen
und kumulierte Workerzeiten der bestehenden Operationskategorien. Kumulierte
Workersekunden sind **keine** Anteile der verstrichenen Gesamtlaufzeit.

Separate cProfile-Diagnoseläufe untersuchen den Aufwand einzelner Funktionen.
Ihre durch Profilierung beeinflussten Laufzeiten werden nicht als Performance-
Nachweis mit leicht instrumentierten Läufen verglichen. Der besonders teure
achte Fall des primären Datensatzes wird zuerst diagnostisch profiliert.

## Ergebnisgleichheit

Der Vergleich umfasst Terminierung, Gewinner, Fragenzahl, jede gewählte Query und
Antwort, Kandidatenverläufe, Volumina, Volumenanteile, Erwartungswerte und alle
Antwortwahrscheinlichkeiten der gewählten Queries. Numerische Werte müssen
zusätzlich zur Gleichheit ohne absolute oder relative Abweichung in ihrer
`float.hex()`-Darstellung übereinstimmen, einschließlich des Vorzeichens von Null.

Neu erfasst der leichte Harness pro Wurzelentscheidung einen SHA256 über das
**vollständige geordnete ValueFunctionResult**: alle bewerteten Querykandidaten,
deren Queryquelle, Erwartungswerte, lexikografische Werte, sämtliche
Antwortzweige, Zweigwahrscheinlichkeiten und Kindzustands-Ergebnisse. Floats
werden vor dem Hashen in ihrer Hexdarstellung serialisiert. Die Zahl der
Bewertungen und Zweige wird daneben ausgewiesen. So kann eine gleiche gewählte
Query nicht verdecken, dass sich ungewählte Bewertungen verändert haben.

Für separate vertiefte Äquivalenzläufe kann `--deep-verify` zusätzlich **jede**
rekursiv ausgewertete Query in den tatsächlichen Workertasks erfassen. Die
Taskdigests werden in der ursprünglichen `executor.map`-Reihenfolge verglichen;
Worker-IDs oder deren zufällige Zuordnung gehören nicht zur Ergebnisgleichheit.
Diese zusätzliche Instrumentierung wird ebenfalls getrennt von den regulären
Laufzeitvergleichen beurteilt.

Vor der Übernahme einer Variante sind passende Tests und die vollständige Suite
erforderlich. Kleine Zeitgewinne brauchen wiederholte, möglichst wechselnde
Referenz-/Kandidatenmessungen. Ein Einzellauf reicht nicht, um einen kleinen
Unterschied als Erfolg auszugeben. Änderungen mit abweichenden Ergebnissen
werden verworfen, auch wenn sie schneller sind.

## Ablage und Reproduzierbarkeit

Die vollständigen lokalen Rohdaten befinden sich im Hauptcheckout unter:

`outputs/benchmarks/bench-i7/20261012-overnight-exact-optimization/`

Der Bench-i7-Spiegel liegt unter:

`/home/cruncher/otqa-benchmarks/20261012-overnight-exact-optimization/`

Der Ordner enthält `protocol.json`, den eingefrorenen Ausgangsquellstand samt
SHA256-Manifest, gespeicherte Eingaben, `run_benchmark.py`, `run_plan.py`,
Ausführungspläne, einzelne Ergebnisdateien, unveränderte Logausgaben und
Provenienzdateien. Jeder Versuch erhält einen eigenen Ordner mit isolierter
Quellkopie, Patch, Testnachweisen, Messungen und Entscheidung. Auch verworfene
Patches bleiben dort als historische Belege erhalten, während ihr Code aus dem
aktiven Worktree entfernt bleibt.

## Versuchsjournal

### 02:07 UTC — Referenzmessungen gestartet

Die Ausgangsversion wird dreimal auf den zehn primären Problemen gemessen.
Danach folgen die einmalige Erzeugung/Messung des zweiten Zehnerdatensatzes und
der separate cProfile-Diagnoselauf. Der Quellstand wird während jedes Laufs vor
und nach der Messung gehasht; Veränderungen würden den Versuch ungültig machen.

Parallel werden drei isolierte Kandidaten vorbereitet, zunächst ohne Übernahme
in den aktiven Algorithmus:

| ID | Ansatz | Vermiedene Arbeit | Numerische Voraussetzung |
|---|---|---|---|
| e01 | Antwortvolumen-Constraint-Keys direkt erzeugen | vollständige Systemkopien und erneute Tupelumwandlung | identische Koeffizienten, Zeilenreihenfolge und Volumenfunktion |
| e02 | kompakter interner HiGHS-Ergebnisaufbau | unbenutzte Dual-/Basisdaten und verschachtelte Ergebnisobjekte | identische kalte LP-Läufe, Prüfung, Status, Lösung und Zielfunktionswert |
| e03 | Bytevergleich bei Eckpunkt-Deduplikation | strukturierte Zeilenvergleiche in `np.unique(axis=0)` | identische gerundete Zeilenklassen und Auswahl der ersten Originalzeilen |

Lokale Mikrobenchmarks aus der Voranalyse sind ausschließlich Hinweise zur
Priorisierung. Entscheidungen beruhen auf den vollständigen Bench-i7-Läufen.


### 02:48–03:00 UTC — Wiederholungen, zweite Einzelrunde und native Diagnose

Die unveränderte Referenz löst die zehn primären Probleme mit insgesamt **75
Queries**, alle zehn korrekt. Die ersten drei Gesamtlaufzeiten betragen
126,188 / 133,962 / 134,471 s. Der erste kühle Lauf unterscheidet sich erkennbar
von den folgenden Läufen; ein Vergleich nur gegen diesen Lauf wäre irreführend.
Der separat eingefrorene zweite Datensatz benötigt 122,272 s und 73 Queries.
Alle primären Wiederholungen stimmen in sämtlichen erfassten Ergebnisbits und
Rootdigests überein (15.149 Rootbewertungen und 45.447 Antwortzweige pro Datensatz).

Die erste wechselnde Einzelrunde verwendet für alle Bedingungen denselben
Harness v1 und dieselben Eingaben:

| Bedingung | Gesamtlaufzeiten (s) | Ergebnisvergleich | Entscheidung zu diesem Zeitpunkt |
|---|---:|---|---|
| Referenz, r4 / r5 / r6 | 128,739 / 133,570 / 132,811 | exakt gleich | Kontrollen |
| e01, direkte Branch-Keys | 130,409 / 131,218 | exakt gleich | noch nicht übernommen |
| e02, kompakter interner LP-Ergebnisaufbau | 128,850 / 129,045 | exakt gleich | weitere Wiederholungen / zweiter Datensatz nötig |
| e03, Byte-Deduplikation | 129,506 / 128,340 | exakt gleich | weitere Wiederholungen / zweiter Datensatz nötig |

Die Reihenfolge steht in `plan-screen-e01-e03.json`. Hardware-Snapshots lesen
u.a. die CPU-Frequenzen, unveränderte Governors und Temperatursensoren. Diese
Abfragen dauern zusammen ungefähr sechs Sekunden zwischen den Bedingungen;
sie liegen außerhalb der Lösungszeit, können aber den Wärmehaushalt beeinflussen.
Die früheren r1–r3 haben einen älteren Plancontroller ohne diese langsamen
Abfragen. Deshalb werden sie nicht mit den aktuellen Läufen zu einer scheinbar
genaueren Vergleichsgruppe vermischt. Die in Rohreports gespeicherten
`reference_comparison`-Speedups beziehen sich auf r1 und sind keine
Übernahmeentscheidung. Maßgeblich sind zeitnahe gleich instrumentierte Kontrollen.

Der separate cProfile-Lauf des achten Problems benötigt 39,374 s und
299,984 kumulierte aktive Workersekunden; diese verlangsamte Diagnose ist kein
regulärer Performancevergleich. Von der leichten primären Referenz entfallen
788,606 von 957,631 kumulierten Workersekunden auf exakte Antwortwahrscheinlichkeiten,
117,023 s auf Ratioanalysen und 52,002 s auf andere Arbeit. In cProfile fallen
u.a. LP-Ergebnisaufbau, sparse Matrixaufbau, Geometrie und Constraint-Umwandlungen
auf. Die Funktionszeiten überlappen und dürfen nicht addiert werden. Historische
Diagnosemetadaten des ersten Harness nannten cProfile noch nicht explizit;
Ausführungsplan, Environment und pstats belegen den Diagnosemodus.

Harness v2 wurde erst nach Ende der laufenden v1-Runde eingefroren und
übertragen (SHA256 `1e5972953c3a8c6ab339b364487fbab4d27775182fb291404f713a082dd71067`).
Ein unabhängiger Review und frische Funktionstestprozesse bestätigen identische
native Modellzustände, Status, Primalwerte und numerische Resultate mit/ohne
transparenten Native-Timer. Diese Timer laufen ausschließlich in einer separaten
Diagnose; sie teilen Konstruktion, Modellübergabe, Solverlauf und Ergebnisabrufe
nach LP-Zweck auf. Ihre Zeiten sind Teilmengen der inklusiven Workerzeiten.

Vor der nächsten serialisierten Runde werden auf dem Bench-i7 alle vollständigen
Tests der Referenz und der sechs Varianten ausgeführt. Ein erster Controllerstart
mit `pytest` brach ab, weil das Benchmark-Environment pytest nicht enthält; dabei
wurde kein Benchmark ausgeführt und nichts am Algorithmus oder Environment
verändert. Der korrigierte Controller verwendet die vorhandenen unittest-Suites.
Beide Starts bleiben dokumentiert. Danach folgen native Diagnose, ein ausdrücklich
markierter Aufwärmlauf mit unveränderter voller Testkonfiguration und wechselnde
Kontrollen / Einzelmessungen von e04 (gemeinsame Ratio-Vorbereitung), e05
(bedingt vektorisierte Ratios) und e06 (direkter identischer CSC-Matrixaufbau).

### Verworfene Voranalysen ohne Remote-Laufzeitbehauptung

- **e05, unbegrenzt vektorisierte Ratios:** Das große temporäre Tensorarray war bei
  500 Eckpunkten im Mikrotest 1,63-mal langsamer als die gemeinsame Vorbereitung.
  Dieser Entwurf wurde ersetzt; Archiv und Patch bleiben erhalten. Die aktuelle
  Variante verwendet einen rein größenabhängigen Wechsel zum identischen
  vorbereiteten Verfahren und eine Speicherobergrenze, ohne Ratios auszulassen.
- **e07, `map(float, ...)` für Constraint-Keys:** Der schnelle direkte Ansatz
  würde ungewöhnliche `__float__`-Implementierungen mit StopIteration anders
  behandeln. Der sichere Wrapper erhält die Semantik, bringt jedoch nur einen
  winzigen Mikrogewinn; nach e01 ist noch weniger Arbeit betroffen. Keine aktive
  Änderung und kein behaupteter Gesamtlaufzeitgewinn.
- **Direkte Normberechnung:** Ein spezieller axis=1-Pfad könnte einen kleinen Teil
  des NumPy-Dispatch sparen. Der Defaultpfad verwendet jedoch eine andere
  native Reduktion; dessen Austausch wäre numerisch riskant. Bislang kein aktiver
  Code; die Voranalyse bleibt im Versuchsmaterial erhalten.

Alle Kandidaten bleiben zunächst isoliert. Im aktiven Worktree wurde bislang
keine der neuen Algorithmusänderungen übernommen, und es erfolgte kein Commit
oder Push.

### Native Diagnose der Referenz (separater Modus)

Auf dem schweren primären Problem 8 zeigt der transparente native Timer folgende
**kumulierte Workersekunden**, nicht verstrichene Wandzeit:

| LP-Zweck | `run`-Aufrufe | native `run`-Zeit |
|---|---:|---:|
| Geometriezentrum | 128.182 | 20,693 s |
| Antwort-Support | 78.446 | 15,813 s |
| Machbarkeit | 6.157 | 1,094 s |
| Summe | 212.785 | 37,600 s |

Die native Modellübergabe benötigt zusätzlich 2,127 s, clearModel 1,174 s
und clearSolver 0,281 s. Konstruktionen sind nur 256 erforderlich (je Prozess
maximal 32 gecachte Modelle); 212.529 native Wiederverwendungen und kein
Adapterfallback werden gemeldet. 78.180 Modellhits stehen 134.605 Misses
gegenüber. Basisabrufe sind direkt nativ billig; der bisherige Pythonaufbau
vollständiger Dual-/Residual-Ergebnisse kostet jedoch erheblich mehr. Diese
Messung begründet e02 und e06, beweist deren Gesamtnutzen noch nicht.

Der Diagnosebericht benötigt 29,100 s Wandzeit / 220,019 aktive Workersekunden
und liefert trotz Instrumentierung dieselben Rootdigests und Ergebnisbits wie
die Referenz. Eine direkte Speedupbehauptung gegenüber dem langsameren cProfile-
Diagnoselauf wäre falsch: beide Modi haben anderen Messoverhead. Geringe
Unterschiede bei LP-/Cachezählern zwischen Läufen entstehen auch durch die
Zuordnung vollständiger Aufgaben zu Prozessen; diese Zuordnung gehört nicht zum
fachlichen Ergebnis. Die vollständigen Ergebnisdigests bleiben maßgeblich.

Remote-Suites: Referenz 154, e01 156, e02 162, e03 162, e04 163, e05 175,
e06 162 Multistep-Tests sowie jeweils sechs Onestep-Tests bestanden. Die
Versionsunterschiede zwischen lokalem NumPy/SciPy und Bench-i7 sind in den
Rohreports gespeichert; insbesondere der direkte CSC-Entwurf wurde gegen die
tatsächlich installierte Remote-SciPy-Konvertierung geprüft.

### 03:00 UTC — unabhängige lokale Auswertung

`analyze_runs.py` wertet die lokal gespeicherten Originalreports unabhängig vom
Vergleichsfeld des Harness aus. Ein lesender Review und 19 positive/negative
Schemaszenarien bestanden. Der erste Bericht enthält 15 Läufe, davon 13 regulär
verwendbare und zwei getrennte Diagnosen, ohne Schemafehler. Vollständige
Konfiguration, Inputbits, CLI, Source-/Harnesshash und acht tatsächlich aktive
Worker pro Problem werden geprüft. Deep-Digests sind nur mit passender
Deep-Kontrolle vergleichbar. Der Analyzer überschreibt keine Rohdaten und
übernimmt keinen Kandidaten automatisch. Hinweise zu macOS-AppleDouble-Dateien
werden von echten Quellabweichungen getrennt, ohne Provenienz umzuschreiben.

Frische Auswertungen liegen in `analysis/<UTC-Zeitstempel>/`; `analysis-usage.md`
erklärt Wiederholung und die Bedeutung der CSV-Felder. Die nächste Auswertung
folgt nach Übertragung abgeschlossener aktueller v2-Messungen.

### e11 — LP-Eingangsvalidierung: nach vollständigem Microaudit verworfen

Der Entwurf sollte bereits validierte strukturelle LP-Arrays anhand eindeutiger
Inhaltsschlüssel wiederverwenden. Die Objective-Validierung, alle Konversions-
und Fehlerfälle, Mutationsschutz sowie Shapes/Dtypes/Strides müssen dabei
erhalten bleiben. Im tatsächlichen überlaufenden 32er-LRU-Mix spart die sichere
Version nur 0,67–1,85 µs je LP; beim Miss steigen die Kosten um etwa 25–35 %.
4.032 Arraybit-/Metadatenvergleiche, 4.000 Mutationen und 18 Originalfehler
stimmen überein. Der erwartete kleine Gesamtnutzen rechtfertigt die zusätzliche
Validierungs- und Cachekomplexität nicht. Es gibt **keine Produktionsänderung**
und keinen Remote-Speedupclaim. Drei rohe Micros, Guardaudit, Begründung und
Entscheidungsdatei liegen unter `experiments/e11-lp-input-cache/`. Alle 76
archivierten Pythondateien dieses Audits bleiben gegenüber der Referenz
unverändert.

### 03:16 UTC — zweite Einzelrunde vollständig, Kombination gestartet

Alle zehn Probleme und 75 Queries sind in jeder Bedingung korrekt und in allen
erfassten Ergebnisbits/Rootdigests identisch. Diese Runde verwendet nur den
eingefrorenen Harness v2; sie wird nicht mit v1-Laufzeiten zusammengemischt.

| Bedingung | Laufzeiten (s) | Median (s) | Einordnung |
|---|---|---:|---|
| Referenz r7/r8/r9 | 132.500 / 132.451 / 133.550 | 132.500 | Kontrollen |
| e04 gemeinsame Ratio-Vorbereitung | 132.843 / 132.454 | 132.649 | kein belastbarer Gesamtgewinn; nicht übernommen |
| e05 hybride Vektorisierung | 131.084 / 131.329 | 131.207 | kleiner wiederholter Vorteil, Gegenprüfung nötig |
| e06 direkter CSC-Aufbau | 127.536 / 128.178 | 127.857 | wiederholter Vorteil, kombinierte Gegenprüfung nötig |

Der Medianvergleich mit den drei Kontrollen entspricht grob +0,11 % längerer
Zeit für e04, etwa 0,98 % kürzerer Zeit für e05 und etwa 3,50 % für e06. Diese
kleinen Einzelgewinne bleiben vorläufig; zeitnahe Vor-/Nachkontrollen und alle
Einzelprobleme stehen in der unabhängigen Auswertung. Das gesunkene Ratio-
Zeitintervall allein wäre kein Grund, e04 zu behalten, wenn die Gesamtlaufzeit
nicht sinkt. e05 enthält die gemeinsame Vorbereitung als großen Geometrie-
Fallback und muss als gesamte Variante geprüft werden.

S01 kombiniert isoliert exakt e01+e02+e03+e06. Die überlappenden Funktionen
wurden gegen die Originalkomponenten per AST geprüft; der integrierte Patch
lässt sich auf einer frischen Referenz zu identischen Pythonmanifesten
reproduzieren. 180 Multistep- und sechs Onestep-Tests bestehen lokal und remote.
Beim ersten Remote-Auspacken war das Archiv eine Verzeichnisebene zu hoch
extrahiert; der Testcontroller brach vor jeder Rechnung mit einem fehlenden
repo/-Pfad ab. Unverändertes Archiv korrekt in repo/ entpackt, Fehler separat
gespeichert, keine Quelländerung dadurch.

`plan-stack-s01.json` umfasst je drei reguläre primäre/zweite Datensatzläufe
zwischen vier Kontrollen, anschließend passende getrennte Deep-Kontrollen
und S01-Deep-Läufe auf beiden Datensätzen sowie native/cProfile-Diagnosen des
schweren Falles. Danach werden die isolierten e01/e02/e03 auf dem zweiten
Datensatz und mit weiterem primären Wiederholungslauf bestätigt
(`plan-confirm-e01-e03.json`). Bis zu den vollständigen Ergebnissen bleibt
S01 ein isoliertes Messobjekt; canonical bleibt auf dem ursprünglichen Stand.

### 03:30 UTC — kombinierter Vorteil wiederholt, weitere Verifikation läuft

S01 benötigt im primären Zehnersatz 116,733 / 117,549 / 118,178 Sekunden.
Die zwischenliegenden Kontrollen r13/r14 liegen bei 133,067 / 133,017 Sekunden;
r12 nach der Testsuite war mit 129,201 Sekunden kühler und bleibt ebenfalls
sichtbar. Sämtliche 75 Queries, Ergebnis-Floatbits und vollständigen Rootdigests
stimmen überein. Dieser wiederholte deutliche Vorteil ist noch keine Übernahme:
zweiter Datensatz, rekursive Digests und Diagnosen laufen im selben seriellen
Plan weiter. Es wurden keine Produktionsänderungen nach canonical übertragen.

e13 (Wiederbefüllen eines nativen `HighsLp`-Templates) wurde nach dem Audit
verworfen. Die Konstruktion kostet netto nur etwa 0,272 µs; selbst bei allen
134.605 Modellcache-Misses wären das lediglich 0,037 kumulierte Workersekunden.
Der optimistische Vergleich des vollständigen Befüllens liegt bei 16,418 gegen
15,860 µs, also höchstens etwa 0,075 Workersekunden für alle Misses. Tatsächlich
gibt es weniger Wiederverwendungsereignisse. Zudem sind native `scale_`/`mods_`
undurchsichtig: ein vollständiger Reset auf den Ausgangszustand ist nicht
belegt. Kein Produktionskandidat, kein Remote-Speedupclaim; Rohmessungen und
Entscheidung stehen in `experiments/e13-template-reuse/`.

Der nächste größere Ansatz prüft, ob beim vorhandenen geometrischen Support-
Test jeweils nur die unsichere der beiden Grenzen per LP bestimmt werden muss.
Dies darf ausschließlich mit unveränderter konservativer Eckpunktprüfung und
Toleranz erfolgen; bei fehlender oder vollständig unsicherer Geometrie bleiben
beide bisherigen LPs erforderlich. e14 wird zunächst isoliert gegen die
unveränderte Referenz geprüft. Messbare Ergebnisse gibt es dafür noch nicht.

### 03:47 UTC — S01 regulär auf beiden Datensätzen bestätigt, Deep-Prüfung gestartet

Die vollständigen regulären S01-Wiederholungen sind abgeschlossen. Alle zehn
primären Fälle (75 Queries) und alle zehn unabhängigen zweiten Fälle (73
Queries) stimmen bei jeder Wiederholung in allen erfassten Ergebnisbits und
vollständigen Rootdigests mit 074 überein.

| Datensatz | Referenzkontrollen (s) | S01 (s) |
|---|---|---|
| Primär | 129,201 / 133,067 / 133,017 / 132,779 | 116,733 / 117,549 / 118,178 |
| Zweiter Datensatz | 121,255 / 121,514 / 121,550 / 122,261 | 107,160 / 107,081 / 107,014 |

Alle Kontrollen bleiben sichtbar, einschließlich des kühleren primären ersten
Laufs nach der Testsuite. Die groben Medianvergleiche ergeben etwa 11,55 % bzw.
11,89 % kürzere Zeit. Dies ist weiterhin ein vorläufiger Zwischenstand aus
serieller Remote-Ausgabe; Übertragung der unveränderten Rohreports und die
unabhängige Gesamtauswertung folgen nach dem laufenden Plan. Die passenden
Baseline/S01-Deep-Läufe für beide Datensätze haben begonnen. Ihre zusätzliche
Hasharbeit gehört nicht in den regulären Performancevergleich.

e14 ist isoliert lokal fertig und unabhängig geprüft. 162 Multistep- und sechs
Onestep-Tests bestehen. Ein echter 074-Replay umfasst 2.688 Outcomes, davon
2.592 erfolgreiche Wahrscheinlichkeitsfälle mit 7.776 identischen Floathex-
Werten sowie 96 identische Fehlerfälle; 744 Support-LPs entfallen. Weitere
unabhängige Proben umfassen 1.260 Klassifikationen und 600 volle nachfolgende
LP-Ergebnisvergleiche trotz Modellcache-Eviction. Beide Richtungen behalten die
vollständige vierstufige Retrykette für die unsichere Grenze. Die neue separate
Statistik zählt partielle Support-Versuche und vermiedene Optimierungen, keine
nativen Retryversuche; die acht alten Vertexcache-Felder bleiben erhalten.
Die Margin ist der vorhandene numerische Schutz der Referenz und kein neuer
formaler Genauigkeitsnachweis. Es gibt noch keinen Remote-Gesamtlaufzeitclaim
für e14. Rohdaten und Review: `experiments/e14-partial-support/`.

S02 wird isoliert als exakt S01+e14 vorbereitet. Zusätzlich sind die geprüften
Kandidaten e09(size2/size4), e10 und e12 samt festem v2-Screen-/Deep-Plan
paketiert. e08 bleibt wegen des winzigen prognostizierten Gesamtnutzens im
optionalen Plan, ausdrücklich ohne Behauptung eines Remote-Verlusts. Der
unabhängige Packaging-/Testreview liegt in
`pending-review/20261012T033458.318190Z/`.

Für vergleichbare cProfile-Paaranalysen werden Quellenmanifest/-fingerprint,
vollständiges Ergebnisschema, feste Konfiguration/Inputbits, verfügbare
Harnessbytes und die exakte Zuordnung der acht `.pstats`-PIDs zu den
aufgezeichneten aktiven Workern geprüft. Der lesende Review hatte fehlende
Guards entdeckt; diese wurden vor jeder produktiven Paaranalyse ergänzt.
`compare_cprofile.py` summiert exklusive Callsite-Workersekunden und kennzeichnet
inklusive Zeilen ausdrücklich als überlappend. Die Paaranalyse ist eine
getrennte Diagnose und entscheidet nicht automatisch über Übernahmen.

### 04:00 UTC — S01 lokal übernommen, Schleife wird fortgesetzt

Die abgeschlossenen Rohreports, Logs und Provenienzen wurden unverändert und
SHA256-geprüft lokal gesichert. Die unabhängige Auswertung
`analysis/20261012T035729.575086Z/` hat keine Lese-/Schemafehler. Beide vollständigen
Deep-Vergleiche sind exakt: primär 415.151 rekursive Queryauswertungen und
15.149 Rootauswertungen, zweiter Datensatz 400.494 rekursive Auswertungen und
14.222 Rootauswertungen. Die separat aufgezeichneten Worker-Taskzahlen betragen
15.155 bzw. 14.224; diese sind nicht mit den finalen Rootauswertungen gleichzusetzen.

S01 (e01+e02+e03+e06) wurde daher als **erste Overnight-Übernahme lokal** auf die
kanonische Worktree angewendet. Alle 80 Pythondateien sind bytegenau identisch
zur gemessenen S01-Kopie; Fingerprint
`8abc41e4e780f50fc8b49f8d5fdd8ae86bd985b4e9abd0bb0a19cd7d7eb439d2`.
Die erneuten kanonischen 180 Multistep-/sechs Onestep-Tests und
`git diff --check` bestehen. HEAD bleibt unverändert auf 074. Es wurde weder
committet noch gepusht. Entscheidung mit allen acht Kontrollläufen, sechs
Kandidatenläufen, vollständigen Deep-Zahlen und Grenzen:
`experiments/s01-core-stack/decision.json`; Übernahmeprüfung und Testlogs im
selben Ordner.

Der akzeptierte Medianvergleich umfasst ausdrücklich alle zeitnahen Kontrollen:
primär 132,897758 → 117,548545 Sekunden (**11,5496 % weniger Zeit**), zweiter
Datensatz 121,532108 → 107,081482 Sekunden (**11,8904 % weniger Zeit**). Die
alten eingebauten Quotienten gegen den einzelnen kühlen r1-Lauf und die
separaten Profilerzeiten bilden keine Übernahmegrundlage. Dies sind feste
Bench-i7-/Software-/Inputvergleiche, kein universeller numerischer Beweis oder
reiner Hardwarevergleich.

Die einmalige Rohdatenübertragung begann um 03:56:47 UTC während des ersten
Kontrollfalls der anschließenden Komponentenbestätigung. Dabei lief kein
zweiter Solver; es wurden kleine Reportdateien gelesen/kopiert. Dieser
Übertragungszeitpunkt bleibt als Kontext sichtbar und betrifft nicht die
bereits abgeschlossenen S01-Übernahmeläufe. Weitere Übertragungen größerer
Profilpakete erfolgen nach der aktuell laufenden Serie.

S02 ist als exakt S01+e14 vorbereitet und lokal erneut mit 188+6 Tests geprüft.
Eine neu hinzugefügte Leerzeile am Dateiende wurde vor jeder Remote-Messung
entfernt; der ursprüngliche Stand inklusive Patches, Archive und Rohbelege
bleibt unter `preclean/` erhalten. Der finale Sourcefingerprint lautet
`429539ced0e3168791991de9d36a524fbb2616a78475ca46afa8851ecda61158`.
Alle mathematischen ASTs sind unverändert durch diese Ein-Byte-Bereinigung.
Die aktuelle Quelle besteht die Whitespace-Prüfung; unabhängiger finaler
Hash-/Integrationsreview läuft. Die 32 Zeilen des vorbereiteten
`plan-partial-support-v2.json` vergleichen e14 gegen 074 und S02 gegen den
jetzt akzeptierten S01, jeweils auf beiden Datensätzen mit Kontrollen,
Wiederholungen, Deep- und getrennten Native/cProfile-Diagnosen.

e15 simuliert rein beobachtend größere native Modellcaches bei unverändertem
echtem 32er-/4-MB-Cache. Lokal ergeben 512 statt 32 Einträge auf zwei echten
acht-Worker-Strömen nur 85 bzw. 139 zusätzliche Hits (0,081 bzw. 0,0995 % der
Misses). Keine Produktionsänderung. Die Probe weist ausdrücklich eine andere
Mac-/Software-Trajektorie als Bench-i7 aus; daraus werden keine Remote-LPzahlen
oder Remote-Speedups abgeleitet. Ein optionaler gleichkonfigurierter
Bench-i7-Beobachter liegt unter `experiments/e15-model-capacity-audit/`.

### 04:17 UTC — weitere Kandidaten vorbereitet, noch nicht übernommen

Der finale unabhängige S02-Review bestätigt die vollständige Byteidentität aller
182 Archivdateien bei frischem Patch-Replay. S03 ist lokal als **exakt S02 plus
unverändertes e05** vorbereitet; Sourcefingerprint
`ad7b110a9e5fb69fcc09c3e8b11d2b4a83b20396d1978388df70bf11e944bb7b`.
209 Multistep-/sechs Onestep-Tests bestehen. Ein zusätzlicher tatsächlicher
074-Einzelpairvergleich umfasst 10.264 Arrays, 208.580 vollständige Intervalle
und 17.648 rohe Min/Max-Werte ohne Floathex-Abweichung. Er enthält Nullvorzeichen,
Subnormals, große Werte, Nennerlöcher, Strides und die Tensorgrenze 79/80/81.
Keine weiteren Guards oder Toleranzänderungen waren nötig. Die volle
Remote-Laufzeitwirkung von S02 und S03 ist noch unbekannt. Quellen, Proben und
replaybare Patches liegen unter `experiments/s02-partial-support-stack/` bzw.
`experiments/s03-ratio-stack/`; vorläufige Übernahmebehauptungen werden vermieden.

Der e16-Audit prüft zusätzliche Allokations-/Norm-/Validierungswiederverwendung.
Die affine Zerlegung selbst wird bereits fast vollständig wiederverwendet
(240.807 Zugriffe, acht echte Berechnungen; rund 0,0097 kumulierte Workersekunden).
Ein weiterer Kontextcache verspricht daher kaum Gewinn. Die untersuchten
halbseitigen `column_stack`-Arrays enthalten unterschiedliche letzte Spalten;
es handelt sich nicht um dieselbe Berechnung. Erstmalige Mikroproben zum
Recycling frischer Validierungsarrays sind mit vollständigen Guards langsamer.
Ein abschließender unabhängiger Systemvalidierungs-Audit läuft noch. Diese
Befunde sind Diagnosen, keine angeblich gemessenen Remote-Verschlechterungen.

Die optionalen e15-Beobachtertools werden vor einem möglichen Remote-Diagnoselauf
zusätzlich auf eindeutige PID-Zuordnung, Zählerkonsistenz und passende
Host-/Runtime-/Quellenreferenzen geprüft. Regulärer eingefrorener v2-Harness,
aktuelle Messkopien und produktiver 32er-Modellcache bleiben unverändert.

### 04:30 UTC — Komponentenbestätigung und nächstes Profiling-Paar

Die v2-Komponentenbestätigung ist vollständig und unverändert lokal gesichert.
`analysis/20261012T042903.860058Z/` liest 65 Messreports, davon 50 reguläre,
mit null Lese-/Schemafehlern. Alle 15 Läufe des Bestätigungsplans sind exakt.
Der zweite Problemsatz hat drei Kontrollläufe (121,726 / 122,309 / 123,638 s)
und zwei Wiederholungen pro Änderung. Medianwerte: e01 119,038 s (2,67 % weniger),
e02 118,317 s (3,26 % weniger), e03 117,744 s (3,73 % weniger). Im primären Satz
enthält diese neue v2-Runde nur eine Kandidatenwiederholung pro Änderung und
zwei Kontrollen; sie ergänzt frühere v1-Proben, deren Zeiten nicht mit v2
vermischt werden. Die S01-Übernahme beruht weiterhin auf ihrer stärkeren
kombinierten Drei-Wiederholungen-Evidenz auf beiden Sätzen.

Die nativen S01-/074-Diagnosen zeigen identisch 78.446 Support-LP-Runs und
6.157 Feasibility-Runs. Durch unterschiedliche Taskzuweisung auf die Worker
entstehen 128.182 bzw. 128.186 Geometrie-Runs (vier zusätzliche Cache-Misses),
während sämtliche funktionalen Ergebnisbits exakt bleiben. Native Cachezähler
sind somit Ausführungsdiagnosen und kein Resultatgleichheitskriterium.
S01 ruft für die kompakten internen Resultate `getBasis` nicht mehr auf;
`getSolution`, Statusprüfung, kalte Modell-/Solverinitialisierung und LP-Optionen
bleiben erhalten. Die kumulierten Nativezeiten sind weder reine Solver-CPUzeit
noch ein addierbarer weiterer Walltime-Anteil.

Das passende v2-cProfile-Paar ist abgeschlossen: 074 35,828 s, S01 30,978 s
mit 16 Fragen beim ursprünglichen achten Problem. Diese Profilerzeiten sind
**keine neue reguläre Performanceentscheidung**. Die SHA-geprüfte Übertragung
und exklusive Funktionspaaranalyse folgen nach dem optionalen e15-Beobachter.
Alle sechs nächsten Kandidatenarchive (e14, S02, e09/2, e09/4, e10, e12) wurden
auf Bench-i7 übertragen, ausgepackt und erneut gegen ihre lokalen Source-
Fingerprints geprüft; vollständiger Transportnachweis unter `source-transfers/`.
Der globale v2-Harness wurde dabei nicht verändert.

E16 ist endgültig dokumentiert verworfen: vollständige lokale
Validierungsprototypen mit identischen Bits/Layouts und allen Guards sind
ungefähr 1–8 % langsamer; die kleine einmalige Zeilenmaskenwiederverwendung
spart nur etwa 0,06 kumulierte Workersekunden. Es gibt keinen Produktionspatch
und keinen behaupteten Remote-Verlust. Unabhängiger Bericht und Rohproben:
`experiments/e16-geometry-allocation-audit/`.

### 04:33 UTC — passende Funktionsdiagnose gesichert; Support-Runde gestartet

Das SHA-geprüfte passende v2-Profiling-Paar ist ausgewertet unter
`cprofile-analysis/074-to-s01-paired-v2/`. Quellen-/Eingabe-/Konfigurations-/
Host-/Harnesschecks und die genaue Zuordnung der acht PIDs zu ihren Pstats
bestehen; die Ergebnisbits sind identisch. Die gesamten exklusiven Python-
Callsitezeiten betragen 266,861 bzw. 231,503 kumulierte Workersekunden. Darin
liegt auch von Python aufgerufene native Arbeit. Inklusive Funktionszeiten
überlappen und werden nicht addiert. Der wesentliche Resultataufbau sinkt
exklusiv von 13,015 auf 6,674 Workersekunden; Constraint-Kopien und Schlüssel-
Generierungen entfallen zu großen Teilen, strukturierte Unique- und allgemeine
Sparse-COO-/CSC-Aufbereitung werden durch die gemessenen kürzeren Pfade ersetzt.
Neue Helfer besitzen entsprechend eigene Zeiten; ein positiver neuer
Funktionswert bedeutet nicht automatisch einen gesamten Verlust.

Der zusätzliche e15-Beobachter auf **Bench-i7 selbst** ist abgeschlossen,
streng ergebnisgleich und unverändert lokal gesichert (13 Dateien samt
Hashmanifest unter `observer-transfers/`). Virtuell 512 statt 32 Einträge würden
130 zusätzliche Hits von 134.606 ursprünglichen Misses vermeiden, also
**0,0966 %**. 64/128/256 Einträge ergäben 49/69/99 weitere Hits. Kein Byte-Limit-
Eviction, keine Fallbacks, keine echten Cachegrößenänderungen. Das 4-MB-Limit
bezieht sich weiterhin auf das ursprüngliche Array-Accounting, nicht auf die
vollständige Python-/Native-RSS. Dieser Befund rechtfertigt keinen größeren
Produktionscache und ist kein hypothetisch gemessener Speedup.

Jetzt laufen die vollständigen E14-/S02-Suites und anschließend der unveränderte
32-Zeilen-Plan für die partielle Support-Prüfung, mit voller Aufwärmrunde,
zeitnahen Kontrollen, Wiederholungen, beiden Datensätzen sowie getrennten
Deep-/Native-/cProfile-Läufen. Es läuft kein zweiter Remote-Solver. S03 ist
unabhängig nochmals geprüft (209+6 Tests und Original-Replay reproduziert);
der vorbereitete Folgeplan mit acht S02-Kontrollen/sechs S03-Regulärläufen
bleibt bis zu einer tatsächlichen Übernahme des exakten S02-Quellstands gesperrt.

### 04:56 UTC — erste direkte S02-Vergleiche, weitere Vorbereitung

E14 einzeln benötigt primär 125,934 / 126,349 s, die drei zeitnahen
074-Kontrollen 133,279 / 133,208 / 133,265 s. Der direkte kombinierte
S01→S02-Vergleich beginnt mit S01 116,911 / 117,282 s und S02 111,847 /
111,589 s. Alle zehn Fälle haben weiterhin 75 Fragen und exakt identische
funktionale Bits und Rootdigests. Diese ersten Werte sind vorläufige
Zwischenstände, **keine Übernahmeentscheidung**; die zweite Kohorte sowie
vollständige Deep-/Native-/cProfile-Läufe stehen noch aus. Der aktive lokale
Algorithmus ist weiterhin exakt S01; HEAD bleibt 074.

S04 ist lokal als exakt S03+e10+e12 vorbereitet (220 Multistep-/sechs Onestep-
Tests); der unabhängige Review bestätigt die unveränderten übrigen S03-
Dateien, AST-/Byte-Inversion und strikte Patch-Replays. Alle bisherigen e03-
Deduplikations-/e06-CSC-Blöcke, e02-Kompaktflag, e05-Ratios und e14-Support
bleiben erhalten. Kein Prozesschunking, keine Ergebnisprüfungsänderung in S04.
Sourcefingerprint `7b270b00af91cca1318585c16f488b154b550b22250925bfb7ea5e8d18a4fe0c`.
Die Messung ist an einen tatsächlich übernommenen S03-Stand und positive
isolierte e10-/e12-Laufzeitbefunde gebunden; sonst bleibt der Versuch lediglich
historisch vorbereitet. Es gibt noch keine solche Remote-Evidenz.

E17 entsteht aus dem **frischen S01-v2-Profil**: 212.785 interne Checker-
Aufrufe werten bei `integrality=None` eine immer-falsch maskierte Sonder-
Allowance für semi-kontinuierliche Variablen aus. Der private Spezialpfad
vermeidet nur diese tote Operation und behält sämtliche tatsächlichen
Primalprüfungen, Rechenreihenfolgen, Toleranzen, Status-/Meldungslogik und
Retries. Öffentliche vollständige und direkte kompakte Adapteraufrufe bleiben
unverändert. Der Pfad benötigt einen ausdrücklichen internen Opt-in, passende
native Arrays und die exakt geprüfte installierte SciPy-Checkerquelle;
Quellabweichungen oder Overrides fallen auf den tatsächlichen Originalchecker
zurück. Die Checkerquelle der lokalen SciPy 1.17.1 und Bench-i7-SciPy 1.18.1
ist bytegleich. Die Utility-Datei wurde lediglich lesend per stdlib übernommen,
mit dokumentiertem Zeitpunkt während des ersten Kontrolllaufs (kein SciPy-
Import und kein zusätzlicher Solver; der Leseaufruf dauerte lokal rund 0,03 s).

Lokale vollständige Resulthelper-Mikros deuten auf rund 8,4 µs weniger Arbeit
pro passendem Aufruf; daraus wird kein gesamter Remote-Speedup berechnet.
352 öffentliche volle Resultate (inklusive Marginals/Nit), 352 private kompakte
Resultate und 1.200 nachfolgende LP-Records sind exakt; eigene Randfall-/
Override-/Stride-/Fehlerprüfungen bestehen. Ein zunächst übersehener negativer
Stride eines formal C-kontiguierlichen Einzelelements wurde durch einen
zusätzlichen Fallback abgesichert; der ursprüngliche fehlgeschlagene Testlog
bleibt erhalten. Die noch ungemessenen E17-/S05-/S06-Pakete bekommen vor
jeder Remote-Messung eine präzisierte private Docstring und zusätzliche
In-place-Code-/SciPy-np-Alias-Overrideguards; vorherige Pakete bleiben unter
`preclean/`. S05 bedeutet S02+E17, S06 S03+E17. Kein zweiter sinnvoller neuer
Hotspot-Kandidat ergab sich aus diesem Audit. Nachweise:
`experiments/e17-current-hotspot-audit/` sowie die jeweiligen isolierten Stacks.

### 05:06 UTC — dritte S02-Wiederholung und finale Checkerpakete

Die dritte primäre S02-Wiederholung beträgt 111,700 s; die vier zugehörigen
S01-Kontrollen 116,911 / 117,282 / 117,348 / 118,019 s. Weiterhin alle
75 Queries und sämtliche funktionalen Floatbits/Rootdigests identisch.
Die Holdout-Reihe läuft; S02 ist weiterhin nicht übernommen.

E17/S05/S06 sind mit korrigierter eigener Docstring und zusätzlichen
In-place-Code-/SciPy-np-Alias-Overrideguards frisch verpackt. Lokale Suites
189+6 / 197+6 / 218+6 bestehen; komplette Resulthelper-Mikros sparen noch
8,39–8,47 µs pro geeignetem Aufruf. Die früheren Pakete sind unter preclean
gesichert, die neuen Quellen werden vor jeder Remote-Arbeit unabhängig
geprüft. Suitepläne wiederholen ausschließlich neue Kandidaten, bereits
geprüfte unveränderte Basen werden nicht unnötig nochmals getestet.

### 05:14 UTC — Abschluss lokaler Vorbereitung und unabhängiger Zusatzsatz

Finale Checkerpakete und alle sechs strikten Voll-/inkrementellen Patchreplays
sind unabhängig bestätigt; E17/S05/S06-Fingerprints stehen in state.json
und ihren implementation.json. Nachweise: E17 reviewer.md und
pending-review/continuous-stack-review/plan-review.json. Beide alternativen
Benchmarkpläne enthalten 19 Zeilen; nur der zur tatsächlich akzeptierten
Basis passende Plan wird ausgeführt. Die korrigierten Suitepläne testen nur
den neuen Kandidaten. S03/S04-Suitepläne sind ebenso reduziert. Im Pending-
Suiteplan wurde der nicht hochgeladene zurückgestellte E08-Eintrag entfernt;
vier tatsächliche Kandidaten bleiben. Alte Pläne/Hashes sind bytegenau unter
pending-review/suite-plan-maintenance/historical gesichert. Benchmarkpläne,
Quellen, eingefrorener Harness und Runner blieben unverändert.

input-final-validation.json enthält zehn vorab eingefrorene zusätzliche Fälle
(Seed20261013, unverändert5x5, Grid10+Ratios-Konfigurationsfingerprint); erzeugt
mit Original074-Generator auf dem Mac und danach zwingend exakter Floatreplay.
SHA256 61b524027493331ce0a1e3a26999d3ac84578872c6492e09fb9404e0d1a1f935.
Alle300 Eingabefloats sind im eigenen Manifest als float.hex gespeichert.
Kein Solver, keine Eingabenfilterung, kein Remote-Upload oder Lauf bisher.
Dieser Satz ist für eine unabhängige spätere Schlussvalidierung reserviert,
nicht Teil bisheriger Laufzeitentscheidungen.

### 05:21 UTC — Holdout-Zwischenbefund, Dokumentationsaudit

E14 einzeln holdout116,570/116,749 s; zeitnahe074-Kontrollen
121,813/121,871/121,587 s. Kombiniert S02 bislang103,204/102,689 s,
S01-Kontrollen106,899/107,156/107,076 s. Dritte S02-Wiederholung und
Deepprüfungen stehen noch aus; keine vorgezogene Übernahme.

Unabhängiger Audit des Journals:56 Belegprüfungen und149 gehashte Dateien,
keine sachlichen Findings; kanonische Pythonquellen weiterhin exaktS01.
Aktive Remote-Zwischenwerte waren bewusst noch nicht lokal übertragen und
sind ausdrücklich als Auditgrenze vermerkt. Bericht unter
pending-review/journal-audit/REVIEW.md. Transportvorbereitung prüft alle
vier finalen S03/S04/S05/S06-Archive vollständig gegen ihre lokalen Quellen;
noch kein Upload. S07 wird ausschließlich als weitere historische
Kombinationsvorbereitung erzeugt, um einen eventuell akzeptierten Checker
mit eventuell tatsächlich nützlichen E10/E12 gemeinsam prüfen zu können.

### 05:40 UTC — S02 übernommen, S03-Runde gestartet

S02 ist nach Abschluss des festen Plans akzeptiert: sämtliche14 regulären
S01/S02-Läufe, beide vollständigen rekursiven Kohortenprüfungen gegenactual074
zusätzlich zu ihren S01-Planreferenzen, SHA-/Config-/Harness-/Inputs-/Runtime-
Checks und Remote188+6 bestehen. Alle Kandidatenzeiten liegen unter jeder
der vier zugehörigen Kontrollen. PrimärKontrollmedian117,315 s gegenüber
111,700 s (4,7866 % weniger), Holdout107,105 gegenüber102,717 s
(4,0968 % weniger). Beide vollständigen Float-/Root-/Deepvergleiche exakt.
Entscheidung und strikte Auswertung: experiments/s02-partial-support-stack/
decision.json bzwpaired-analysis-v2/comparison.json.

Inkrementeller S01-Patch ist im kanonischen Arbeitsverzeichnis übernommen;
alle81 Pythondateien entsprechen der gemessenen Quelle bytegenau. Dort
188 Multistep-/6 Onestep-Tests undgitdiffcheck bestanden, HEAD unverändert074.
Nachweiscanonical-adoption-checks.json. KeineCommits/Pushes.

Der passende S01→S02-Funktionsvergleich untercprofile-analysis/s01-to-s02-v2
besteht; exklusive Worker-Callsitezeiten231,503→223,296 s, Diagnosezeiten
nicht als reguläre Gewinne verwendet. Der Nativecase8Workerbefund zählt
78.446→44.221 Supportruns (34.225 vermiedeneLPs), bei unverändert106.864
Queryauswertungen/320.592Zweigprüfungen/6.157States/6.157Feasibilityruns.
Geometrie128.186→128.184 Runs durch2andereCache-Doppelmisses; keine neue
Ergebnisabweichung. Instrumentierte Supportrunzeiten17,057→10,244 kumulierte
Workersekunden sindnativeDiagnose, keineadditivenWalltimeanteile.

Ein unabhängiger Nachreview des Auswerters fand einen künstlich reproduzierten
Command-Seed-Provenanceguard-Mangel. Alle18 echten S01/S02-Kommando-/Plan-/
Raw-Seeds wurden zusätzlich manuell geprüft und sind korrekt; deshalb keine
Algorithmus- oder Adoptionsänderung. Prüffilemanual-command-seed-check.json.
Auswerter wird separat abgesichert, alter freigegebener Toolstand historisch
gesichert; Rohmessungen/Quellen/Harness bleiben unverändert.

Nach voller Quellen-/Archiv-/Modusprüfung wurden vierfinaleStagingrepos und
achtPläne im Idle übertragen (source-transfers/20261012T053757.562361Z.json).
S03+Ratio läuft jetzt direkt gegen akzeptiertesS02, mit nur neuer Suite,
Warmup, vierKontrollen/dreiWiederholungenjeKohorte und getrenntenDeep-/
Native-/CP-Läufen. Die übrigen Kandidaten bleiben reine Vorbereitung.

### 05:54 UTC — primäre Ratio-Wiederholungen und Reporterabschluss

S03primär108,663/109,441/108,891 s, zeitnaheS02-Kontrollen bislang
111,442/111,000/111,238 s; die vierteKontrolle, Holdout undDeep folgen.
75Fragen undsämtlichegespeicherteFloat-/Rootwerte bleibenexakt.
Das istweiterhin einvorläufiger Befund, keineS03-Übernahme. Dergrößenabhängige
Ratio-Spezialpfad verwirftkeineEckpunkte oderQueries; großeMengen werden mit
dem separat exaktgeprüften sequentiellen/batchweisenPfad bearbeitet.

DerAuswerterfix ist unabhängig abgeschlossen: aktuellecompare_stack.py-SHA
4c4426665c4446fd3c008c3ccd58eba93b91cceeffcc66b476fb933bee14e331,
42Guardfälle plus separateManipulationsproben bestanden. Gesamte kanonische
Controllerargv inklSeed/defaults/fixenParametern/Flags wird geprüft; doppelte,
unkanonische oder widersprechendeOptionen abgewiesen. NeueS01/S02-Reports
sind exakt rekonstruiert, S02Timing/Semantik/Source/Inputbits identisch zum
ursprünglichen Bericht. OriginalTool/Report bleiben historisch erhalten.
S02-Nachweis supplemental-command-validation.json; unabhängigeReviews unter
pending-review/compare-stack-independent-review/ undcompare-stack-review/.
KeinAlgorithmus-/Harness-/Controller-/Rohdatenedit durch diesenToolfix.

S07 istunabhängig finalvorbereitet (229+6): exaktS04 plusfinalE17, bei
beidenModulinversionen E12-MaskenundE17-Checker erhalten, vierPatchrouten in
beideRichtungen strict byte-/modusgleich. Fingerprint51d2234fe9ab12b27f949644d6a1dfa0ae369ce9f4ea91b5cadafc57ea3a92c7.
Beidealternativen19+1Pläne sindgegenFinalplan collisiongeprüft; nochkeine
Remoteausführung, GatesbeiderisolierterE10/E12-Gewinne bleiben bestehen.

Aktuelles Statusregister experiment-ledger.json trennt alle17Ansätze nach
tatsächlichem Befund; alteStatebytes sind in state-history gesichert.
Unverändert einzig S01undS02 akzeptiert, HEAD074, keineCommits/Pushes.

### 06:08 UTC — Ratio-Holdout und nächste Prüfschritte

Die drei S03-Holdout-Wiederholungen benötigen 100,673 / 101,398 / 100,403 s.
Die bisherigen zeitnahen S02-Kontrollen liegen bei 103,562 / 103,214 /
102,927 s; die vierte Kontrolle läuft noch. Sämtliche 73 Queries und alle
gespeicherten Float-/Root-Digests stimmen exakt überein. Das sind laufende,
noch nicht lokal übertragene Messungen, keine abgeschlossene S03-Entscheidung.
Beide vollständigen rekursiven Vergleiche und die getrennten Diagnoseprofile
folgen im unveränderten Plan. Der aktive kanonische Stand bleibt S02.

Für spätere Übernahmen wird ein lokaler Helfer adopt_verified_stack.py
vorbereitet. Er trifft keine Laufzeitentscheidung: Root muss zunächst den
kompletten Vergleich beurteilen. Der Helfer prüft die exakte Ausgangsquelle,
den freigegebenen Reporter, die Beleg-Hashes, den inkrementellen Patch und
die komplette gemessene Zielquelle; danach führt er die beiden kanonischen
Testsuiten und git diff --check aus. Keine Commit-/Push-Funktion. Noch nicht
für eine Übernahme ausgeführt; der bisherige Übernahmenachweis bleibt erhalten.

### 06:16 UTC — S03 übernommen; S06 als nächste Messrunde

S03 ist akzeptiert. Der strikte Bericht enthält sämtliche 14 regulären
Läufe sowie beide vollständigen rekursiven Vergleiche gegen tatsächliches074.
Primär: S02-Median 111,333 s, S03-Median 108,891 s, 2,1933 % weniger Zeit.
Holdout: S02-Median 103,197 s, S03-Median 100,673 s, 2,4458 % weniger Zeit.
Alle drei Kandidatenwerte liegen jeweils unter allen vier Kontrollen; auch
sämtliche Vergleiche gegen die benachbarten Kontrollen fallen positiv aus.
Alle gespeicherten Floatbits, vollständigen Root- und rekursiven Digests
sind exakt gleich. Roh-/Suite-/Profiltransfers 06:13 sind SHA-verifiziert.
Entscheidung: experiments/s03-ratio-stack/decision.json; Auswertung:
paired-analysis-v2/comparison.json. Diagnosezeiten sind weiterhin ausgeschlossen.

Das Profil erklärt die Wirkung: 311.060 einzelne Ratio-Berechnungsaufrufe
werden durch 15.553 gebündelte Aufrufe ersetzt. Alle geordneten Zielpaare
bleiben erhalten; das tatsächliche Fünf-Ziele-Limit für den Tensorpfad bleibt
80 Eckpunkte, mit unveränderter numerisch identischer Ausweichberechnung.
Exklusive kumulierte Worker-Callsitezeit im Diagnosefall 223,296 -> 218,531 s.
Fundstelle cprofile-analysis/s02-to-s03-v2/comparison.md. Native/CP-Zeiten
sind keine additiven Walltimeanteile und kein Ersatz für reguläre Wiederholungen.

Der inkrementelle S02-Patch ist übernommen. Alle 84 kanonischen Pythondateien
sind bytegleich zur gemessenen S03-Quelle, 209 Multistep-/6 Onestep-Tests
und git diff --check bestehen; HEAD bleibt074. Der Übernahmehelfer wurde
unabhängig geprüft; zwei Vorabfindings (unter -O entfallende asserts und
zu spät geprüfte Logexistenz) wurden vor seiner ersten Ausführung behoben.
Check-only und echte Übernahme bestanden. Kein Commit/Push.

E18 ist als negativer Beobachtungsaudit abgeschlossen: frische S02-Profile,
Native-Daten und sämtliche 182 Paketdateien unabhängig geprüft; keine weitere
sicher bitgleiche neue Hypothese mit mindestens ungefähr einem Prozent
Gesamtpotenzial etabliert. Drei kleine lokale Mikrohypothesen sind bewusst
nicht zu Produktionsänderungen promoviert. Kein Remote-Benchmark oder
Laufzeit-Misserfolg erfunden. Bericht experiments/e18-s02-hotspot-audit/AUDIT.md.

Als nächstes wird S06 = exaktes S03 + unveränderter finaler E17-Checker
direkt gegen S03 geprüft. Die tatsächlichen Zulässigkeitsprüfungen, Toleranzen
und Solver-Retryfolgen bleiben erhalten; der Spezialpfad vermeidet nur den
für rein kontinuierliche Probleme toten Semi-Continuous-Anteil. Noch keine
S06-Übernahme. Alle übrigen Kandidaten bleiben auf ihren dokumentierten Gates.

### 06:37 UTC — laufender S06-Vergleich und Schlussvalidierungsreserve

S06-Remote-Suiten bestehen mit 218 Multistep-/6 Onestep-Tests. Die drei
primären Wiederholungen benötigen 106,013 / 105,007 / 105,524 s; bisherige
zeitnahe S03-Kontrollen 109,705 / 109,262 / 108,930 s. Vierte Kontrolle und
Holdout folgen, dann beide vollständigen rekursiven Prüfungen und getrennte
Native-/cProfile-Diagnosen. Ergebnisse und 75 Queries bislang exakt gleich.
Diese noch nicht lokal gesicherten Zwischenwerte sind ausdrücklich kein
S06-Übernahmeentscheid. Der aktive Quellstand bleibt S03.

Der aktive S06-Launcher wurde vor Ausführung unabhängig gelesen; lokale
Quellen, beide Planhashes, 80 neue Ausgabepfade und acht externe Referenz-Raws
geprüft. Der Remote-Preflight prüfte Idle, Quellen und Transporthashes, bevor
Suite und Benchmark seriell starteten. Remote-Nachweis launch-preflights/
20261012T062252.731595Z.json wird nach der Runde gesichert. Der laufende
Launcher, Runner, Harness, Pläne und Quellen bleiben unverändert.

Die vier tatsächlichen Pending-Suiteeinträge sind jetzt auf Bench aktualisiert
(SHA d8df7969...), die früheren fünf Planzeilen bleiben bytegenau lokal
erhalten. Unabhängiger Schlussdatensatz Seed20261013 und unveränderter
21-Zeilen-Finalplan sind ebenfalls SHA-verifiziert übertragen; keine
Schlussquellenkopie oder Solverausführung bisher. Transportnachweis
source-transfers/20261012T061841.737556Z-final-input-pending-plan.json.

Die unabhängige E09-Kompositionsvorbereitung ist abgeschlossen: alle acht
temporären S02/S03/S06/S07 × Chunk2/4-Patchrouten und 17 Guardgruppen
bestehen; noch keine echte spätere Kombination, kein reservierter Lauf und
kein Remotegewinn. Root wählt erst anhand der isolierten Messungen eine
konkrete Chunkgröße und die dann tatsächlich akzeptierte Ausgangsquelle.

Zeitplanung: Nach S06 folgen die unveränderten vier isolierten Vollscreens.
Nützliche unveränderte Deltas werden gegen den jeweils aktuellen akzeptierten
Stand im vollständigen 19-Zeilen-Plan geprüft. Jede Übernahme erfordert
weiterhin beide vollständigen Deep-Vergleiche gegen tatsächliches074. Die
zusätzlichen optionalen isolierten Deep-Läufe werden nur ausgeführt, wenn sie
über den ohnehin erforderlichen vollständigen Kombinationsnachweis hinaus
Nutzen bieten; sie sind Diagnosevalidierung, keine Laufzeitkohorte. Der
Schlussvergleich erhält seine feste Reserve ab ungefähr 09:10 UTC. Weder
Problemkonfiguration noch reguläre Wiederholungen oder Übernahmeguards
werden zur Einhaltung der Zeit verkleinert.

Ein lesbarer, unabhängig beleggeprüfter S03-Snapshot liegt unter
pending-review/readable-report/report.md und report.html (236 Belegprüfungen,
619 Datei-SHAs, 42 funktionierende Beleglinks). Er markiert S06 und die
Schlussvalidierung bewusst offen; Root aktualisiert später einen separaten
Abschlussbericht statt diesen historischen Snapshot umzuschreiben.

## Ergebnis nach Kapazitätsunterbrechung (2026-10-12T11:12:54.641405+00:00)

Die Agentenausführung wurde nach ungefähr fünf Stunden durch den vom Nutzer gezeigten Fehler „Selected model is at capacity“ unterbrochen. **Die angeforderten acht Stunden und der abschließende direkte Gesamtvergleich wurden nicht vollständig ausgeführt.** Bei der Rückmeldung des Nutzers um etwa 11:05 UTC war die Frist 09:55:53 UTC bereits verstrichen. Zur Wiederherstellung wurden keine neuen Remote-Benchmarks oder Optimierungsschleifen gestartet.

Die zuvor gestartete S06-Messserie war auf Bench-i7 vollständig beendet (2026-10-12T06:55:23.268429+00:00). Ihre Rohdaten, Provenienzen, Suite-Logs, acht Workerprofile und Launch-Preflight wurden hashgeprüft lokal gesichert. Nach vollständiger Auswertung ist S06 lokal übernommen: jeweils vier Kontrollen und drei Kandidatenläufe auf beiden eingefrorenen Zehner-Problemsätzen, alle Kandidaten schneller als sämtliche Kontrollen, beide vollständigen rekursiven Vergleiche zu actual074 exakt, 218 Multistep- und 6 Onestep-Tests sowohl auf Bench als auch nach lokaler Übernahme bestanden. Der ergänzende Controller-/Zeitfenster-Audit bestätigt alle 27 verwendeten Läufe. Ein unabhängiger Reviewer bestätigt zusätzlich alle 311 Report-Evidenzhashes, Rohdaten, Mediane, Digests und den kanonischen Stand.

| Behaltener Schritt | Erster Problemsatz: Median vorher → nachher | Weniger Zeit | Zweiter Problemsatz: Median vorher → nachher | Weniger Zeit |
|---|---:|---:|---:|---:|
| Weniger LP- und Constraint-Verwaltungsaufwand | 132.898 → 117.549 s | 11.550% | 121.532 → 107.081 s | 11.890% |
| Bekannte Supportgrenzen wiederverwenden | 117.315 → 111.700 s | 4.787% | 107.105 → 102.717 s | 4.097% |
| Ratiointervalle gemeinsam berechnen | 111.333 → 108.891 s | 2.193% | 103.197 → 100.673 s | 2.446% |
| Prüfung kontinuierlicher LP-Ergebnisse spezialisieren | 109.096 → 105.524 s | 3.274% | 100.715 → 98.053 s | 2.643% |

Die Zeiten sind die Summe der zehn `solve_problem`-Laufzeiten einschließlich Poolstart und -ende; Diagnosezeiten sind ausgeschlossen. Die Zeilen stammen aus getrennten kontrollierten Messserien gegen den jeweils vorherigen akzeptierten Stand. Prozente werden weder addiert noch als abschließend gemessener Gesamtgewinn ausgegeben. Der geplante direkte Gesamtvergleich fehlt.

Unverändert: 5 Ziele, 5 Alternativen, Grid10 plus Ratios, Tiefe2, acht Worker und exakte geometrische Volumenwahrscheinlichkeiten. Toleranzen, Kandidatenabdeckung und Abbruchbedingungen wurden nicht reduziert. Pro Zehnersatz bleiben 75 beziehungsweise 73 Queries. Gespeicherte Gleitkommabits, ausgewählte und nicht ausgewählte Rootbewertungen sowie vollständige rekursive Digests stimmen in allen Übernahmeprüfungen mit actual074 überein. Dies gilt für die geprüften Instanzen und ist kein universeller Gleitkommanachweis.

Das separate passende S03→S06-cProfile zeigt 218.531 → 209.247 Sekunden exklusive kumulierte Workerzeit. Der spezialisierte private Checker vermeidet bei kontinuierlichen LPs nicht benötigte Ganzzahligkeits-/Semicontinuous-Verarbeitung; tatsächliche Primal-, Status- und Toleranzprüfungen bleiben erhalten. Diagnosezeiten sind keine regulären Laufzeitmessungen, inklusive Kategorien dürfen nicht addiert werden.

E09 (Chunkgröße2/4), E10 (Normwiederverwendung) und E12 (Infinity-Masken) bleiben **vorbereitet, aber auf Bench nicht gemessen**. Sie wurden weder übernommen noch als schlechte Optimierungen eingestuft. E08 bleibt lokal zurückgestellt. Kombinationen S04/S07 und der unabhängige dritte Schluss-Problemsatz wurden nicht ausgeführt. E01–E18 einschließlich verworfener und beobachtender Ansätze sind im bisherigen Journal und im Ledger dokumentiert.

Aktueller lokaler Quellstand: exakt gemessenes S06 `a83fed3d865341162b15865536f3d87e753a06bb5bd567ce913b0597367b9571`; Git-HEAD bleibt `074fbf9d3c09ae1318155a15f549833df661a9b3`. **Nichts committet oder gepusht.**

Belege im Artefaktroot: `recovery-summary-20261012.json`, `experiments/s06-ratio-continuous-stack/decision.json`, dort `paired-analysis-v2-recovery/comparison.json` und `canonical-adoption-checks.json`, `cprofile-analysis/s03-to-s06-v2-recovery/comparison.md`, `pending-review/controller-window-audit/20261012T110738.585467Z.json`. Der lesbare S03-Snapshot bleibt als historischer Zwischenstand unverändert.


## Neue Messrunde: drei Vorschläge gegen angenommenes S06 (2026-10-12T11:24:17.595408+00:00)

Der Nutzer autorisiert jetzt den tatsächlichen Test der drei offenen Vorschläge und ein zusammenfassendes Fazit aus Nacht und neuer Runde. Die neue Messbasis ist exakt das angenommene S06, nicht der frühere074-Stand. Normwiederverwendung, Chunkgröße2, Chunkgröße4 und Infinity-Masken werden als vier isolierte Varianten vorbereitet. Nichts wird committet oder gepusht.

Der feste Plan enthält 32 reguläre Läufe: je Problemsatz vier gemeinsame S06-Kontrollen und jeweils drei Wiederholungen jeder Variante. Kandidatenreihenfolgen wechseln. Die acht gemeinsamen Kontrollen werden transparent geteilt; sie sind keine 32 unabhängigen Kontrollen. Jeder Vergleich zeigt sämtliche Werte und den tatsächlichen zeitlichen Abstand zur Kontrollklammer. Ein Warmup, acht vollständige Deep-Vergleiche sowie zehn separate Native-/cProfile-Diagnosen sind von Laufzeitwerten ausgeschlossen. Alle zehn Probleme jedes festen Eingabesatzes bleiben 5×5, Grid10+Ratios, Tiefe2 und acht Worker; sämtliche Toleranzen und Abbruchregeln bleiben unverändert.

Plan: `plan-daytime-single-on-s06-v2.json` (9bd1029083eb7e06f26870b211408fcbfabd0b3f50011a45c4df5a22c7e7723f); Protokoll `daytime-protocol-20261012.json`. Alle Pakete enthalten ausschließlich die jeweils genannte Änderung auf S06. Ganze lokale und Bench-Suites sind je neuer Quelle einmal vorgesehen. Bereits getestete unveränderte S06-Suites werden nicht erneut ausgeführt. Ein anschließender direkter074→Endstand-Vergleich soll den bisher fehlenden Gesamtgewinn messen.


## Ergebnis der neuen Vorschläge (2026-10-12T13:04 UTC)

Die 51-Zeilen-Messserie ist vollständig beendet (11:36:25–12:57:28 UTC, Exit0). Alle vier Varianten wurden isoliert auf S06 geprüft; acht gemeinsame Kontrollläufe werden nicht als 32 unabhängige Kontrollen gezählt. Pro Problemsatz vier Kontrollen, pro Variante drei vollständige Wiederholungen. Alle acht rekursiven Vergleiche gegen tatsächliches074 und sämtliche gespeicherten Gleitkommabits/Rootbewertungen sind exakt. Mac- und Bench-Suites bestehen je Quelle. Der unabhängige vollständige Rohdaten-, Paket-, Archiv-, Rechte- und Profil-Audit hat keine offenen Befunde.

| Variante auf S06 | Weniger Zeit erster Satz | Weniger Zeit zweiter Satz | Entscheidung |
|---|---:|---:|---|
| E10: gleiche Koeffizientennormen lokal wiederverwenden | 0,1498% | 0,4141% | Verworfen: Wertebereiche überlappen, kein robuster Gesamtgewinn |
| E09: Aufgabenbündel mit Größe2 | −0,2946% | −0,1055% | Verworfen: langsamer, überlappende Messwerte |
| E09: Aufgabenbündel mit Größe4 | −2,3331% | −1,9043% | Verworfen: alle Kandidaten langsamer als alle Kontrollen |
| E12: float64-Infinity-Masken vereinfachen | 0,8856% | 0,4176% | Behalten: alle drei Kandidaten unter allen vier Kontrollen in beiden Sätzen |

Das vor Messbeginn eingefrorene Entscheidungskriterium steht in `daytime-decision-policy.json`. Einzelentscheidungen einschließlich aller Werte und Kontrollklammern liegen unter `experiments/daytime-20261012/*/decision.json`; die gemeinsame Entscheidung in `daytime-root-decision.json`. Berichte enthalten die tatsächlich wechselnden Abstände zu den gemeinsamen Kontrollen. Kleine Mediandifferenzen allein begründen keine Übernahme.

E12 allein wurde zusätzlich auf den kanonischen S06-Stand übernommen. Vollständige 224 Multistep- und 6 Onestep-Tests bestehen nach Integration; `git diff --check` besteht. Kanonischer Fingerprint: `d1a36969125c2ebcbb84d4bdcebf62da91e94e76d3581f029261e1b86583c60d`. HEAD bleibt074. Keine Commit-/Push-Aktion. E09/E10-Dateien sind ausschließlich in eingefrorenen Experimentpaketen erhalten.

Die getrennte cProfile-Diagnose erklärt lokale Effekte, liefert aber keine regulären Laufzeitwerte: E10 reduziert Normaufrufe von897386 auf640890, ohne robusten Gesamtnutzen. E12 entfernt jeweils538484 `np.isposinf`/`np.isneginf`-Aufrufe; direkte float64-Vergleiche erhalten dieselben Masken. Kleinere aufsummierte Workerzeiten bei Chunk4 verbessern dessen tatsächliche Laufzeit nicht; stärkere Bündelung kann die Verteilung und den kritischen letzten Worker verschlechtern. Diese Diagnose ist keine additive Kausalauswertung.

Der ausstehende direkte Vergleich des vollständigen Endstands gegen074 wird nun separat durchgeführt. Seine Werte ersetzen keine der isolierten Messungen; inkrementelle Prozentwerte werden nicht addiert. Ein unabhängiger dritter Zehner-Problemsatz folgt als zusätzliche vollständige rekursive Validierung.


### 13:16 UTC — eingefrorener Endstand und direkter Gesamtvergleich

`final-daytime/repo` ist eine vollständige byte- und rechtegleiche Kopie des übernommenen E12-auf-S06-Pakets (189 Dateien, 87 Pythondateien). Originalarchiv ebenfalls bytegleich, SHA `5e9efb796846805db0ab81e7596e6089a9f335a31308dcf4b053875f6e0f1b8f`. Neue Algorithmusänderungen und zusätzliche Wiederholungen unveränderter Test-Suites: null. Die vollständigen Mac-, kanonischen Mac- und Bench-Testbelege werden unverändert geerbt.

Vor dem Einfrieren wurden zwei vorhandene Formatannahmen ausschließlich im Prüfhelfer korrigiert: Das Quell-Dateimanifest speichert SHA plus Rechte statt nur SHA; das literal SHA-gepinnte Legacy074-Archiv enthält zulässige Verzeichniseinträge. Beide exakten Manifestformate sind geprüft, Legacyverzeichnisse nur für dieses gepinnte Archiv erlaubt. Kandidatenarchive bleiben strikt nur reguläre Dateien. 14 fokussierte Guards und der vollständige echte Check-only bestehen. Historischer Helfer erhalten; enger Diff/Review in `pending-review/daytime-final-freeze-helper/schema-correction-review/`. Kein Algorithmus-, Paket-, Harness-, Plan- oder Messdatenbyte geändert.

Transport bestätigt sämtliche Quellenbytes/Rechte, Archive und drei feste Pläne. Nach Idle-/SHA-/Quellen-/Frischeprüfung startet Job `jobs/daytime-final-paired-v2` um13:16:45 UTC (PID180587). Plan enthält21 Zeilen: ein ausgeschlossener Warmup, pro Problemsatz vier neue074-Kontrollen und drei Endstandläufe, zwei vollständige rekursive Endstandprüfungen sowie vier getrennte Native-/cProfile-Diagnosen. Der dritte Schlussdatensatz ist als eigener anschließender Zwei-Zeilen-Plan vorbereitet. Die Tagesentscheidungen werden während des Endvergleichs nicht verändert.


### 14:00 UTC — unabhängiger dritter Schlussdatensatz vollständig berechnet

Dritter Problemsatz Seed20261013 unverändert5×5: beide Quellen lösen alle10 Probleme mit90 Queries. Die beiden vollständigen rekursiven Rohdatensätze sind fertig, Job `jobs/daytime-final-third-v2` Exit0 um14:00:41 UTC. Diese einmaligen instrumentierten Prüfläufe bilden keine reguläre Performancekohorte; aus ihnen wird kein zusätzlicher Prozentgewinn geschätzt.

Die erste unveränderte Third-Reader-Ausführung verweigert den Nachweis ausschließlich wegen einer bisherigen Auditorannahme: Fall10 hat0 Queries und keinen Workerpool, der Auditor verlangt dennoch8 aktive Worker-PIDs. Beide Rohdaten bestätigen unveränderte `volume_confidence`-Terminierung schon nach der initialen Zustandsanalyse, Anteil exakt `0.9944993195247578` ≥ unverändert0.99, zwei Kandidaten[0,1], richtiger Gewinner1, leere Query-/Root-/Recursive-Listen,0 Planungscalls und ein vollständig leeres/Null-Workerprofil. Die Fälle1–9 haben jeweils8 aktive PIDs. Die gespeicherten Ergebnisse sind nicht geändert. Der fehlgeschlagene ursprüngliche Audit liegt unverändert unter `pending-review/daytime-final-third-audit/actual-20261012T140220.836243Z/`.

Eine getrennte enge Readerkorrektur prüft diesen vollständig belegten Fall ohne Workerarbeit ausdrücklich; keine erfundenen PIDs, keine verworfenen Instanzen, kein Rerun und keine Änderung an Algorithmus, Messharness, Parametern oder originalem Analyzer. Alle übrigen Quellen-/Input-/Konfigurations-/Ergebnisguards bleiben erforderlich. Alter Reader und zugehöriger Reportbuilder bleiben als historische Artefakte erhalten; die korrigierte Route wird separat nachgewiesen.


## Abgeschlossenes Gesamtfazit: Nacht plus neue Vorschläge

Der frische direkte Vergleich actual074→Endstand ist vollständig und unabhängig geprüft. Ein Warmup und sämtliche Deep-/Native-/cProfile-Diagnosen sind ausgeschlossen. Pro festem Zehner-Problemsatz vier neue Kontroll- und drei neue Endstandläufe.

| Vollständiger Problemsatz | Ausgangsmedian074 | EndstandmedianS06+E12 | Weniger Zeit | Queries |
|---|---:|---:|---:|---:|
| Seed20260902, zehn5×5-Probleme | 132,937096s | 104,871660s | 21,111816% | 75 |
| Seed20261012, zehn5×5-Probleme | 121,629506s | 97,920330s | 19,492948% | 73 |

Alle drei Endstandläufe sind in beiden Sätzen schneller als jede der vier Kontrollen. Erster Satz Kontrollwerte132,925853/133,005838/132,691859/132,948338s; Endstand104,708590/104,871660/106,770666s. Zweiter Satz Kontrollen121,528413/123,249693/121,418476/121,730600s; Endstand97,920330/97,555678/98,144541s. Mediane werden über vollständige Gesamtlaufzeiten gebildet; nicht als Summe von Problemmedianen oder inkrementellen Gewinnen. Die Beobachtung ist keine statistische Signifikanzbehauptung oder universelle Performancegarantie.

Quelle und Einstellungen bleiben exakt5 Ziele,5 Alternativen,Grid10+Ratios,Tiefe2,8Worker,exakte geometrische Volumenwahrscheinlichkeiten. Kein Sampling, keine geringere Kandidatenabdeckung, keine angepassten Toleranzen/Abbruchgrenzen. Zwei volle Endstand-Deep-Vergleiche gegen074, alle Rootbewertungen und gespeicherten Gleitkommabits stimmen. Unabhängiger21-Zeilen-Audit: `pending-review/final-paired-daytime-independent-review/actual-20261012T135528.445910Z/final-verification.json`; direkter Bericht `final-daytime/paired-analysis-v2/comparison.json`; Controllerfenster `pending-review/controller-window-audit/20261012T135442.951671Z.json`.

Der separate dritte Zehner-Satz Seed20261013 ist ebenfalls vollständig geprüft:300 originale Eingabe-Floathexwerte,1746 Ergebnis-Floatwerte,90 Root- und16917 rekursive Datensätze exakt,90 Queries in beiden Quellen. Der strikte neue Reader `audit_final_third_v2.py` (SHA5459799809462f8ce29cb61f8b926d730abc26d75d7f7a96d0b358c46a45b040) belegt für Fall10 die tatsächliche Konfidenzterminierung ohne Query/Workeraufgabe; Fälle1–9 jeweils8Worker. Die Ausnahme ist ausschließlich für den eingefrorenen Fall10 erlaubt, alle anderen originalen Row-, Input-, Reihenfolge-, Source-, Runtime- und Vergleichsguards bleiben erhalten. Originalreader/-Analyzer/-Reportbuilder unverändert erhalten. Der erste6d-Wrapperlauf bleibt historisch; endgültiger SHA545-Nachweis in `pending-review/daytime-final-third-audit/actual-v2-final-20261012T141750.849404Z/third-audit.json`. Die Veränderung betrifft nur den dokumentarischen Reader, keine Berechnung.

Passende Gesamtdiagnose für das schwere primäre Problem8: Queryplanung inkl.Pool/Warten30,669556→23,922442s; initiale/laufende Zustandsanalyse0,133745→0,109856s; restliche Wallzeit0,113688→0,110208s. Insgesamt30,916989→24,142507s in dieser separaten Native-Diagnose, nicht reguläre Performancekohorte. Query-Bewertungen106864 und Antwortzweigprüfungen320592 bleiben gleich; Support-LP-Aufrufe78446→44219. Geometriezentrum-LPs128180→128198 schwanken mit Worker-/Cacheverteilung; nicht alle LP-Zähler sind konstant. Kumulierte Workeroperationszeiten: exakte Volumenwahrscheinlichkeiten196,438784→150,796141s, Ratiointervalle24,511404→18,393488s. Diese Werte sind Worker-Summen, keine additiven Walltimeanteile.

Matched cProfile-Gesamtdiagnose `cprofile-analysis/074-to-final-daytime-v2/comparison.md`:284,236909→206,298545s exklusive kumulierte Worker-Callsitezeit. Die größten lokalen Callsiteeinsparungen betreffen `_solve`, kompakte `_result`-Verarbeitung, den kontinuierlichen Checker und gebündelte Ratiointervalle; Solver-/inklusive Zeiten überlappen und werden nicht addiert. Quellen- und Ergebnisbits sind gepaart identisch.

Die vier nächtlichen Stufen und heutigesE12 bleiben erhalten; E10/E09-Chunk2/Chunk4 sind verworfen und außerhalb des kanonischen Codes vollständig dokumentiert. Die heutige Zusatzreduktion von0,8856%/0,4176% ist separat gegenS06 gemessen. Der Gesamtgewinn21,1118%/19,4929% stammt ausschließlich aus dem direkten Endvergleich. Alle18 experimentellen Ideen behalten ihren tatsächlichen Status im Ledger; vorbereitet/beobachtet gilt nicht als remote gemessener Misserfolg.

Kanonischer Endstandd1a36969125c2ebcbb84d4bdcebf62da91e94e76d3581f029261e1b86583c60d. Vollständige224 Multistep-/6 Onestep-Tests aufMac undBench sowie nach kanonischerIntegration bestehen; Quellen danach unverändert und `git diff --check` bestanden. HEAD074fbf9d3c09ae1318155a15f549833df661a9b3 bleibt unverändert. Alles lokal, keine Commit-/Push-Aktion. Die ursprüngliche8-Stunden-Nacht wurde nach etwa5 Stunden durch Modellkapazität unterbrochen; die hier abgeschlossenen Tagesmessungen folgen auf die neue ausdrückliche Nutzeranweisung.
