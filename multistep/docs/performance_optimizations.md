# Laufzeitoptimierungen des Multi-Step-Algorithmus

## Ziel und Messfall

Die Optimierungen reduzieren die Kosten der exakten Auswertung, ohne die
Standardsemantik des Algorithmus zu ändern. Als reproduzierbarer Messfall dient
`onestep/data/a5_a10_case.json` mit Tiefe 2, 400 Samples, Grid-Größe 21 und
Seed 1.

```bash
python multistep/scripts/benchmark_a5_a10_optimized.py \
  --mode optimized-all \
  --samples 400 \
  --burn-in 200 \
  --thinning 5 \
  --seed 1 \
  --grid-size 21 \
  --max-query-value 100 \
  --workers 1
```

Der Benchmark gibt neben Laufzeit und Ergebnis auch interne Zähler sowie
Zeitanteile aus. Mit `--workers 1` umfasst das Profil auch die rekursiven
Kindzustände; bei mehreren Prozessen enthält es nur die Arbeit des
Hauptprozesses.

## Abendoptimierungen und vier gegen acht Worker (2026-10-12)

Die erste Implementierung des Abends aus Commit
`6c48de1b9868e87eacc0a2caa4905a9e268cdc2e` wurde mit dem aktuellen erfolgreichen
Stand verglichen: Eckpunkt-Support-Cache, Cache der affinen Zerlegung und
Wiederverwendung des LP-Setups mit kaltem Solverzustand. Die beiden unten
dokumentierten verworfenen Caches waren vollständig entfernt.

Der schnelle LP-Pfad nutzt die in SciPy gebündelten privaten HiGHS-Bindings.
Geprüft wurde er lokal mit SciPy 1.17.1 und auf Bench-i7 mit SciPy 1.18.1.
Bei fehlender Kompatibilität oder nicht unterstützten Optionen verwendet die
Implementierung weiterhin das öffentliche `scipy.optimize.linprog`. Die Tests
des schnellen Backends setzen kompatible private Bindings voraus.

Alle vier Varianten wurden frisch und nacheinander auf **Bench E7 / bench-i7**
(Intel Core i7-10700, acht physische Kerne, 16 logische Threads) ausgeführt.
Es waren dieselben ersten fünf gespeicherten Probleme des ursprünglichen
5D-Benchmarks: fünf Ziele, fünf Alternativen, Tiefe 2, Grid-Größe 10 plus
Ratio-Queries an der Wurzel, Ratio-Queries auf Tiefe 1, exakte
Volumenwahrscheinlichkeiten und Abbruch bei einem einzelnen Kandidaten oder
mindestens 99 Prozent Volumenanteil. Jede numerische Bibliothek nutzte einen
Thread pro Prozess. Die tatsächlich arbeitenden Prozesse wurden anhand ihrer
PIDs geprüft: vier bzw. acht Worker in jedem Fall.

| Quellstand | Worker | Gesamtlaufzeit für fünf Probleme | Queries | Korrekte Auswahl |
| --- | ---: | ---: | ---: | ---: |
| Erste Variante des Abends | 4 | 220,454 s | 27 | 5/5 |
| Erste Variante des Abends | 8 | 138,853 s | 27 | 5/5 |
| Erfolgreich optimierter Stand | 4 | 80,899 s | 27 | 5/5 |
| Erfolgreich optimierter Stand | 8 | 50,224 s | 27 | 5/5 |

- Die Optimierungen reduzierten die Laufzeit bei vier Workern um **63,30 Prozent**
  (Faktor 2,73), bei acht Workern um **63,83 Prozent** (Faktor 2,76).
- Acht statt vier Worker reduzierten die Laufzeit der ursprünglichen Variante
  um **37,02 Prozent**, die des optimierten Stands um **37,92 Prozent**.
- Der optimierte Stand mit acht Workern benötigte gegenüber der ursprünglichen
  Variante mit vier Workern **77,22 Prozent weniger Zeit** (Faktor 4,39).

Alle drei direkten Vergleiche gegen die frische ursprüngliche Vier-Worker-Stufe
hatten identische Query- und Kandidatenverläufe, Auswahl und Abbruchbedingungen.
Je Vergleich waren **492 von 492 gespeicherten numerischen Werten** einschließlich
**81 von 81 Antwortwahrscheinlichkeiten der gewählten Queries** auch in der
`float.hex()`-Darstellung identisch. Absolute und relative Toleranz waren null.
Nicht gewählte rekursive Auswertungsbäume wurden nicht vollständig verglichen.
Die eingefrorenen Quellen wurden vor und nach den Messungen über SHA-256 geprüft.

**Folgerung für diesen Benchmark:** Acht Worker sind gegenüber vier Workern
vorzuziehen; für weitere vergleichbare 5D-Läufe `--workers 8` verwenden. Diese
Aussage beruht auf einem Durchlauf je Fall und Variante. Normale Laufzeitvariation
bleibt enthalten; ein statistischer Signifikanznachweis wurde nicht geführt.
Die Zeiten umfassen den Lösungsaufruf einschließlich Poolstart und Poolende.
Imports im Hauptprozess vor diesem Aufruf, Eingabebereitstellung und Berichtsdateien
sind ausgeschlossen; Worker-Initialisierung ist enthalten. Kumulierte Workerzeiten
werden im Bericht separat ausgewiesen.

Lokale Nachweise im Projekt:
`outputs/benchmarks/bench-i7/20261012-g5-a5-evening-optimizations-workers4-vs8/`.
`report.html`, `summary.json`, `per_problem.csv` und `operations.csv` enthalten
Ergebnisse und Zeitanteile. `input.json`, `provenance.json`, `machine.json`,
Quellarchive, SHA-256-Manifeste, `run_benchmark.py` und `run_all.py` dokumentieren
Eingaben, Quellstände und Ausführung. Historische Abendzeiten wurden nicht als
frische Vergleichsmessung wiederverwendet.

Die [kompakten versionierten Benchmarkdaten](benchmarks/20261012_g5_a5_workers4_vs8.json)
enthalten außerdem die fünf gespeicherten Eingaben, Einzelzeiten,
Vergleichsergebnisse, Softwareversionen und SHA-256-Werte der getesteten Quell-
und Testdateien. Sie erhalten die wesentlichen Nachweise auch ohne die
vollständigen lokalen Archive.

## Grid-only mit gleicher Kandidaten-Vorauswahl (2026-10-13)

Der Grid-only-Modus reicht jetzt auf verbleibender Tiefe größer 1 bereits
vorhandene Ratio-Intervallmaps an die konservative Vorauswahl der
Folgekandidaten weiter. Der Guard lautet, mit ausgelassenen Aufrufargumenten:

```python
elif query_source_mode == "central" or (
    query_source_mode == "grid" and precomputed_ratio_intervals is not None
):
    ratio_intervals = _resolve_ratio_intervals(...)
    ratio_intervals_by_goal_pair = {...}
```

Dadurch entstehen weder zusätzliche Ratio-Queries noch zusätzliche
Ratio-Intervallberechnungen. `both`, der Grid-Pfad ohne vorliegende Intervalle
(insbesondere `closed_lp`) und die Standardkonfiguration bleiben unverändert.
Der kleine Patch und vier neue Regressionstests wurden auf Basis von
`8baeac827408156b97814f1210c52bbaac6b0187` als lokaler, uncommitteter Stand
geprüft und gemessen. Die Statusangaben in den Benchmarkdaten beziehen sich
auf diesen Messzeitpunkt vor dem anschließenden Folgecommit.

Auf **Bench i7** wurden dieselben zehn gespeicherten Probleme mit fünf Zielen
und fünf Alternativen frisch mit beiden Einstellungen gelöst: Grid 10,
Tiefe 2, exakte Volumenwahrscheinlichkeiten, acht tatsächlich gestartete
Worker, `forkserver`, ein nativer Thread je Bibliothek und Abbruch bei einem
Kandidaten oder mindestens 99 Prozent Volumenanteil. Jede Einstellung wurde
dreimal gemessen, insgesamt **60 Läufe**, mit neuen Prozessen und wechselnder
Reihenfolge. Einzige Konfigurationsdifferenz: `grid_depth_query_source_mode`
ist `both` oder `grid`; Tiefe 1 verwendet in beiden Fällen Ratio-Queries.

| Einstellung | Summe der zehn Problemmediane | Queries je Durchlauf | Korrekte Auswahl |
| --- | ---: | ---: | ---: |
| Grid plus Ratios, gleiche Vorauswahl | 108,406 s | 75 | 10/10 |
| Grid-only, gleiche Vorauswahl | 120,265 s | 83 | 10/10 |

Grid-only benötigte insgesamt **10,94 Prozent mehr Zeit** und acht zusätzliche
Fragen. Bei sechs der zehn Probleme war sein Median schneller; die Mehrzeit
der anderen vier überwog. Die tatsächlichen Gesamtzeiten der drei
Wiederholungen betrugen für `both` **105,154 / 108,498 / 108,609 s**, für `grid`
**117,045 / 120,159 / 120,598 s**. Deren Mediane sind 108,498 beziehungsweise
120,159 s; diese andere Aggregation ergibt 10,75 Prozent Mehrzeit für Grid-only.
Die Zeiten umfassen die Lösung einschließlich Poolstart und Poolende, jedoch
keine Imports, Eingabeparsing oder Berichtsdateien.

Alle **60 aufgezeichneten semantischen Ergebnisse** sind gegenüber dem
früheren Ergebnis derselben Eingabe und Einstellung mit `float.hex()`
identisch; nur Zeiten und Cachestatistiken wurden ausgenommen. Geprüft wurden
Queries, Antworten, Erwartungswerte, Kandidatenverläufe, Volumina, Anteile und
Auswahl. Innerhalb jeder Einstellung waren auch alle drei Wiederholungen
identisch. Zwischen den beiden Strategien unterscheiden sich dagegen alle
zehn Query-Verläufe. Nicht gewählte rekursive Zweige wurden nicht aufgezeichnet.
Die vollständige Multi-Step-Suite bestand auf Mac und Bench i7 jeweils mit
**228 Tests**, einschließlich der vier neuen Regressionstests.

Eine separate Diagnose nach den Zeitmessungen bestätigte in den zehn
Initialzuständen jeweils **20 geordnete Zielpaar-Maps mit identischen
semantischen Hashes** für beide Einstellungen. Grid-only erzeugte weiterhin
jeweils 200 Grid-Queries; `both` ergänzte 4 bis 24 Ratio-Queries und wählte in
neun Fällen eine davon als erste Query. Diese Diagnose gilt nur für die
unbeantworteten Anfangszustände, nicht für alle Suchzustände oder Fragen.

**Folgerung:** Für diese Arbeitslast bleibt Grid plus Ratios die schnellere
Standardstrategie. Der Patch stellt dieselbe Vorauswahl bereit; sein eigener
Laufzeitnutzen wurde nicht durch einen separaten frischen Vorher-Nachher-Test
isoliert. Frühere Grid-only-Zeiten stammen aus einer anderen Ausführung und
werden nicht als kausaler Optimierungsnachweis verwendet. Zehn Probleme eines
Seeds erlauben keine allgemeine Gleichheits- oder Qualitätsgarantie; die
99-Prozent-Volumenregel garantiert nicht für jedes Zielgewicht die richtige
Auswahl.

Die [kompakten versionierten Nachweise](benchmarks/20261013_grid_prefilter_parity_workers8.json)
enthalten Eingaben, vollständige Konfigurationen, Rohzeiten, Endzustände,
Testbelege, semantische Vergleiche und SHA-256. Vollständige Rohberichte und
Query-Verläufe bleiben im Projekt unter
`outputs/benchmarks/bench-i7/20261013-grid-prefilter-parity-workers8/`
und auf Bench i7 unter dem gleichnamigen Verzeichnis in
`/home/cruncher/otqa-benchmarks/` erhalten.

## Grid 5 gegenüber Grid 10 bei Grid plus Ratios (2026-10-13)

Auf ausdrücklichen Wunsch wurde die Grid-Größe als **Strategieparameter**
verglichen. Dies ist keine weitere Codeoptimierung: Grid 5 kann andere Queries
und Folgezustände erzeugen. Beide Varianten verwenden den unveränderten Commit
`03ac83fda7ed04e28f9782672f2dd49ec5700668`. Die bisherige Benchmark-
Vergleichskonfiguration bleibt **Grid 10 plus Ratios**; bestehende Code-Defaults
bleiben unverändert (Grid-Größe 21). Die einzige Konfigurationsdifferenz zwischen
den beiden gemessenen Varianten ist `grid_size`.

Auf **Bench i7** wurden dieselben zehn gespeicherten 5-Ziele-/5-Alternativen-
Probleme mit Seed 20260902 je Variante dreimal gelöst, insgesamt **60 Läufe**.
Tiefe 2, acht tatsächlich gestartete Worker, `forkserver`, ein nativer Thread
je Bibliothek, exakte Volumenberechnung, Ratio-Queries auf Tiefe 1, Wertebereich
0,001 bis 100, maximal 100 Fragen und 99-Prozent-Abbruchregel blieben gleich.
Neue Prozesse und wechselnde Reihenfolge wurden für beide Grid-Größen verwendet.

| Einstellung | Summe der zehn Problemmediane | Queries je Durchlauf | Korrekte Auswahl | Exakt gelöst |
| --- | ---: | ---: | ---: | ---: |
| Grid 10 plus Ratios | 108,428 s | 75 | 10/10 | 3/10 |
| Grid 5 plus Ratios | 55,456 s | 71 | 10/10 | 6/10 |

Grid 5 benötigte **48,85 Prozent weniger Zeit** (Faktor 1,96) und war bei
allen zehn Problemen schneller. Die drei vollständigen Durchlaufzeiten waren
für Grid 10 **104,426 / 108,446 / 108,753 s**, für Grid 5
**53,108 / 55,511 / 55,538 s**. Der beschreibende Quotient
Gesamtlaufzeit/gestellte Fragen sank von **1,446 auf 0,781 s**;
er ist kein isolierter Timer für die Bewertung desselben Suchzustands.
Die Problemmediane enthalten Session- und Poolstart sowie Poolabbau.

Beide Varianten wählten bei jedem Lauf den anhand der Zielgewichte optimalen
Gewinner; absoluter und relativer ausgewählter Nutzenverlust waren jeweils null.
Grid 5 benötigte in den Fällen 4, 6 und 7 mehr Fragen, in 2, 8 und 10 weniger.
Die neun veränderten Query-Verläufe erzeugten andere Endzustände; nur Fall 9
hatte dieselbe volle aufgezeichnete Semantik. Die drei Wiederholungen je Grid-Größe
waren mit `float.hex()` semantisch identisch, ausgenommen Zeiten und Cachewerte.
Alle 30 Grid-10-Ergebnisse entsprachen außerdem der früheren `both`-Aufzeichnung.
Frühere Laufzeiten wurden nicht zur Schätzung dieser Beschleunigung verwendet.

**Einordnung:** Grid 5 ist für diese Arbeitslast vielversprechend, ohne beobachteten
Qualitätsverlust bei der Auswahl. Zehn unterschiedliche Probleme eines Seeds
belegen keine allgemeine Gleichheit oder statistisch abgesicherte Nichtunterlegenheit.
Die Zeitwiederholungen sind keine zusätzlichen unabhängigen Qualitätsfälle;
die 99-Prozent-Volumenregel garantiert nicht für jedes Zielgewicht die richtige Auswahl.
Die bereits zuvor bestandenen **228 Tests auf Mac und Bench i7** betreffen denselben
87-Datei-Python-Fingerprint; für diesen Parametertest wurde keine Suite neu ausgeführt.

Die [kompakten versionierten Nachweise](benchmarks/20261013_grid5_vs_grid10_ratios_workers8.json)
enthalten Eingaben, beide Konfigurationen, Rohzeiten, Nutzen- und Endzustandsdaten
sowie SHA-256. Vollständige Rohberichte und Query-Verläufe bleiben unter
`outputs/benchmarks/bench-i7/20261013-grid5-vs-grid10-ratios-workers8/`
und auf Bench i7 im gleichnamigen Verzeichnis unter `/home/cruncher/otqa-benchmarks/`.

## Grid 5: Tiefe-3-Pilot mit numerischem Fehler (2026-10-13)

Ein Pilot auf Bench i7 verglich den gespeicherten Fall 1 (fünf Ziele, fünf
Alternativen, Seed 20260902) mit Grid 5 plus Ratios, acht Workern und Tiefe 2
beziehungsweise 3. Exakte Volumenberechnung, native Threads auf 1, Ratio-Queries
auf Tiefe 1 und 99-Prozent-Abbruchregel blieben gleich; die vollständigen
Konfigurationsobjekte sind identisch. Nur der Planungstiefenparameter änderte sich.
Beide Läufe verwendeten den unveränderten Commit `03ac83f` und dieselben externen
Root-Beobachter; deren nicht separat gemessener Aufwand kann mit der Aufgabenanzahl steigen.

Tiefe 2 löste das Problem in **4,487 s mit fünf Fragen**, korrekter Alternative 0
und einem einzelnen verbleibenden Kandidaten. Das aufgezeichnete Ergebnis
entspricht dem bisherigen Grid-5-Ergebnis exakt, ausgenommen Zeiten und Cachewerte.

**Tiefe 3 scheiterte vor der ersten Frage** an
`halfspace center is not strictly interior`. Der Fehler erreichte den Hauptprozess
nach **250,755 s** Planung; nach Abbau noch laufender Workeraufgaben endete der
Prozess nach **307,834 s**. Diese Fehlerzeiten sind keine Lösungszeiten.
Von 124 Root-Queries wurden 109 erfolgreich bewertet, eine erzeugte den Fehler
und 14 wurden nicht gestartet. Der 900-Sekunden-Timeout wurde nicht erreicht;
nach dem Prozessabbau waren keine lebenden Gruppenmitglieder übrig.

Die isolierte Auswertung der fehlerhaften Root-Query reproduzierte den Fehler
in **19,263 s**. Auf dem hypothetischen Pfad folgten `(3, 2, 0.001) >` und
`(4, 3, 0.001) >`, dann eine Ratio-Query mit Wert rund **1,83785 × 10⁷**.
Im vierdimensionalen affinen Raum mit acht Halbräumen meldete das Innenpunkt-LP
einen Radius von **3,03980 × 10⁻⁸**. Der tatsächliche Punkt hatte jedoch eine
minimale normalisierte Reserve von **−3,34722 × 10⁻⁶** und war nicht im Inneren.
Der Schutztest verweigerte die unsichere Volumenberechnung; dieser Befund
rechtfertigt weder die Annahme einer leeren Region noch gelockerte Toleranzen.

Weitere Fälle oder Wiederholungen wurden wegen dieses numerischen Hindernisses
nicht gestartet. Die frühere Durchsatzprognose von etwa 330–340 Sekunden galt
nur für den ersten Root-Batch und erlaubt keine Aussage über die vollständige
Lösungsdauer oder eine 30-Minuten-Grenze. Für Tiefe 3 fehlen gültige Gesamtlaufzeit,
Fragenzahl und Auswahl; ein Qualitätsvergleich ist nicht möglich.

Eine zusätzliche isolierte Diagnose mit positiver Skalierung derselben Zeilen
und öffentlichem `scipy.optimize.linprog` fand einen an den ursprünglichen
Halbräumen geprüften Innenpunkt: Radius **3,04169 × 10⁻⁸**, minimale normalisierte
Reserve **3,03789 × 10⁻⁸**. Die ursprüngliche geometrische Schnittfunktion lieferte
damit **fünf Ecken**. Das stützt den Ansatz für dieses konkrete System, ersetzt
aber keine Produktionsänderung, Regressionstests oder neue Laufzeitmessung.

Ein noch nicht implementierter nächster Ansatz ist äquivalente Normalisierung
der LP-Zeilen mit anschließender Prüfung des Mittelpunkts an denselben ursprünglichen
Nebenbedingungen. Vor einem neuen Vergleich muss diese numerische Robustheit
geprüft werden. Algorithmuscode und Code-Defaults blieben unverändert.

Die [kompakten versionierten Nachweise](benchmarks/20261013_grid5_depth3_pilot_workers8.json)
sichern Fall, Konfiguration, Ergebnis, Fehler, Diagnosesystem und Artefakthashes.
Vollständige Logs und Fortschrittsjournale liegen unter
`outputs/benchmarks/bench-i7/20261013-grid5-depth2-vs-depth3-live-workers8/`;
`pilot-report.md` beschreibt den abgebrochenen Vergleich und seine Grenzen.

## Numerisch robuster Innenpunkt durch äquivalente Zeilenskalierung (2026-10-13)

Der im vorherigen Tiefe-3-Pilot erfasste Fehler wird nun durch einen bedingten
Wiederholungsversuch des Innenpunkt-LPs behandelt. Basis ist Commit `03ac83f`;
`multistep/src/polytope_geometry.py` und dessen Testdatei sind lokal geändert.
Die 87 Python-Dateien wurden unter Fingerprint
`ad6bccef4dfbccfd19c1360b10a9fb8f07d4d7bf686c3d7521b6cf05c6aa3053` eingefroren.
Die vorherigen Fehlerdiagnosen bleiben als historische Messungen dokumentiert.

Der erste LP-Aufruf bleibt unverändert. Sein Mittelpunkt und Radius werden auf
Endlichkeit geprüft. Bei positivem Radius muss der Punkt strikt innerhalb der
projizierten Halbräume liegen; bei kleinem Radius muss er die bestehenden
Machbarkeitstoleranzen erfüllen. Gültige ursprüngliche Mittelpunkte werden
unverändert weiterverwendet, einschließlich ihrer bisherigen Rundung.

Nur bei ungültigem Punkt wird jede vollständige LP-Zeile durch die positive
Norm ihres projizierten Normalenvektors geteilt. Aus
`A_i c + ||A_i|| r <= b_i` wird `A_i/||A_i|| c + r <= b_i/||A_i||`.
Dies verändert weder die zulässigen `(c, r)` noch das Radiusziel. Bounds,
Solveroptionen und Toleranzen bleiben gleich. Der reparierte Punkt wird erneut
an der ursprünglichen Geometrie geprüft. Ungültige oder fehlgeschlagene
Wiederholungen ergeben weiterhin einen Fehler; es gibt keinen Nullvolumen-Kurzschluss.
Grid, Ratio-Queries, exakte Volumenberechnung und Code-Defaults bleiben unverändert.

Die vollständige Multi-Step-Suite bestand auf Mac und Bench i7 jeweils mit
**235 Tests**. Sieben neue Regressionstests sichern den erfassten Fehler,
Unverändertheit gültiger Mittelpunkte, nicht endliche oder außerhalb liegende
Punkte, fehlgeschlagene Wiederholungen und gültige niedrigdimensionale Fälle.
Für das erfasste schmale Polytop werden fünf Ecken und ein positives Volumen
berechnet; die unabhängige analytische Referenz ist
**1,4232771997898375 × 10⁻¹⁴**. Das Volumen wird relativ gegen diese Formel geprüft.

Die zuvor fehlerhafte Root-Auswertung lief in einem frischen isolierten Prozess
mit **20,756 s** ohne Fehler durch; ein ungültiger erster Punkt wurde automatisch
repariert. Diese Diagnose ist keine vollständige Lösung. Ein frischer
Tiefe-2-Kontrolllauf löste Fall 1 in **4,519 s mit fünf Fragen**, korrekter Alternative 0
und genau einem verbleibenden Kandidaten. Seine aufgezeichnete Semantik entspricht
dem vorherigen Live-Pilot exakt, ausgenommen Zeiten und Cachewerte.

Der erneute Tiefe-3-Pilot löste diesen ersten Fall vollständig und fehlerfrei.
Beide Tiefen verwendeten denselben geänderten Quellstand, Grid 5 plus Ratios,
acht Worker und dieselben externen Root-Beobachter.

| Planungstiefe | Lösungszeit | Fragen | Auswahl | Abbruch | Endkandidaten | Volumenanteil |
| --- | ---: | ---: | --- | --- | --- | ---: |
| 2 | 4,519 s | 5 | Alternative 0, korrekt | Einzelner Kandidat | `[0]` | 100 % |
| 3 | 734,258 s | 2 | Alternative 0, korrekt | 99-Prozent-Regel | `[0, 3]` | 99,934292 % |

Tiefe 3 benötigte **12 min 14,26 s**, Faktor **162,47** der Tiefe-2-Laufzeit,
und drei weniger Antworten. Beide wählten den tatsächlichen optimalen Gewinner
mit Nutzenverlust null. Die Endzustände sind unterschiedlich: Tiefe 3 lässt
noch Alternative 3 zu; Tiefe 2 hat genau einen verbleibenden Kandidaten.
Die beiden Tiefe-3-Planungen dauerten **347,860 und 386,350 s**. Alle
**249 Root-Aufgaben** (124 und 125) waren erfolgreich, einschließlich der vormals
fehlerhaften Root-Query. Innere rekursive Phasen wurden nicht separat profiliert.
Die 900-Sekunden-Grenze wurde nicht erreicht; nach Abschluss waren alle
Prozessgruppen beendet und alle 87 Python-Dateien unverändert.

Die Reparatur behandelt dieselbe Geometrie numerisch robuster; die Änderung
von Tiefe 2 auf 3 ist dagegen eine Suchstrategieänderung. Ein Problem mit je
einem Lauf belegt keine allgemeine Laufzeit oder Qualitätsüberlegenheit.
Die 99-Prozent-Volumenregel garantiert nicht für alle Zielgewichte die richtige Auswahl.
Der Mehraufwand der gemeinsamen Root-Beobachter ist nicht separat quantifiziert.

Die [kompakten versionierten Nachweise](benchmarks/20261013_normalized_center_depth3_workers8.json)
sichern Code- und Testbelege sowie vollständige Pilotdaten. Rohberichte und Live-Journale liegen unter
`outputs/benchmarks/bench-i7/20261013-normalized-center-depth3-live-workers8/`.
`pilot-report.md` dokumentiert den abgeschlossenen Pilot und seine Grenzen.

## Verworfene Optimierungsversuche

### 2026-10-12: Gemeinsamer Geometriecache und vollständiger Wahrscheinlichkeitscache

**Entscheidung: Beide zusätzlichen Caches wurden nach dem Benchmark vollständig
aus dem aktiven Code entfernt.** Sie haben im geprüften Ablauf keine
Beschleunigung gebracht, sondern die Laufzeit verlängert. Dieser Versuch soll
bei weiteren Optimierungen als bereits geprüft gelten.

#### Was wurde ausprobiert?

1. **Gemeinsamer Geometriecache:** Ratio-Intervallberechnung und exakte
   Volumenberechnung sollten bereits berechnete Polytopecken gemeinsam nutzen.
   Der Cache speicherte Ergebnisse für exakt gleiche Eingaben, mit höchstens
   32.768 Einträgen und 64 MiB je Prozess.
2. **Vollständiger Wahrscheinlichkeitscache:** Für exakt denselben Gewichtsraum
   und dieselbe Query wurden die drei Antwortwahrscheinlichkeiten (`<`, `=`,
   `>`) gemeinsam gespeichert. Ein Treffer sollte Support-Prüfungen und die
   erneute Zusammensetzung der Wahrscheinlichkeiten vermeiden. Die Grenzen
   waren 32.768 Einträge und 16 MiB je Prozess.

Die Schlüssel enthielten die unveränderten numerischen Eingaben und relevanten
Toleranzen bzw. Solver-Einstellungen. Es gab weder gerundete Schlüssel noch
Sampling als Ersatz für die exakte Volumenberechnung.

#### Messaufbau und Ergebnis

Auf **Bench E7** (Artefaktname `bench-i7`, Intel Core i7-10700) wurden dieselben
zehn gespeicherten Probleme mit jeweils **fünf Zielen und fünf Alternativen**
in drei aufeinanderfolgenden Stufen gelöst. Konfiguration: Tiefe 2, Grid-Größe
10 plus Ratio-Queries an der Wurzel, Ratio-Queries auf Tiefe 1, vier Worker,
ein Thread je numerischer Bibliothek und `answer_probability_mode="exact_volume"`.
Der Abbruch erfolgte bei einer einzelnen verbleibenden Alternative oder einem
Volumenanteil von mindestens 99 Prozent. Instanzen und Zielgewichte wurden aus
`input.json` wiedergegeben, statt sie neu zu erzeugen.

Die Referenz enthielt bereits die erfolgreichen Optimierungen
**Eckpunkt-Support-Cache, Cache der affinen Zerlegung und Wiederverwendung des
LP-Setups**. Diese bleiben erhalten. Der hier verworfene gemeinsame
Geometriecache ist eine zusätzliche Optimierung und ersetzt diese nicht.

| Variante | Gesamtlaufzeit für zehn Probleme | Gegenüber der Referenz |
| --- | ---: | ---: |
| Bisheriger optimierter Stand | 206,091 s | Referenz |
| Zusätzlich gemeinsamer Geometriecache | 217,941 s | +5,75 % |
| Zusätzlich beide neuen Caches | 226,802 s | +10,05 % |

Der vollständige Wahrscheinlichkeitscache verlängerte die Laufzeit gegenüber
der Geometriecache-Stufe nochmals um **4,07 Prozent**. Eine Variante mit nur
dem neuen Wahrscheinlichkeitscache wurde nicht separat gemessen.

Die Zeiten sind summierte verstrichene Lösungszeiten einschließlich
Prozesspool-Start und -Ende, keine kumulierten Workerzeiten. Imports im
Hauptprozess vor dem Lösungsaufruf, Instanzerzeugung und Berichtsdateien sind
ausgeschlossen; die Worker-Initialisierung ist im Pool-Start enthalten. Alle Stufen nutzten
denselben leicht instrumentierten Benchmark. Pro Problem und Stufe wurde ein
Durchlauf gemessen; daraus folgt keine Aussage über statistische Signifikanz.

#### Warum wurde der Versuch verworfen?

- Der gemeinsame Geometriecache hatte in der Geometriecache-Stufe in den
  Workern **0 Treffer bei 594.923 Anfragen ohne Treffer**.
- Der vollständige Wahrscheinlichkeitscache hatte in der kombinierten Stufe
  in den Workern **194 Treffer und 414.957 Anfragen ohne Treffer**. Das sind nur
  **0,047 Prozent Treffer** unter 415.151 Aufrufen.

Die erhoffte Wiederverwendung trat damit fast nie ein. Schlüsselaufbau,
Nachschlagen und Speichern verursachten zusätzliche Arbeit, der kaum
eingesparte Berechnungen gegenüberstanden. Die Messung spricht deshalb gegen
diese beiden zusätzlichen Caches in diesem Ablauf. Sie belegt nicht, dass
Caching für andere Eingaben generell ungeeignet wäre. Eine reine Erhöhung der
Cache-Grenzen ist durch diese Daten nicht als Verbesserung begründet.

#### Ergebnisgleichheit und endgültiger Codezustand

Alle drei Varianten lösten **10 von 10 Problemen korrekt** und stellten
insgesamt **75 Queries**. Die drei paarweisen Vergleiche der aufgezeichneten
Query-Verläufe, Kandidaten, Auswahl und Abbruchbedingungen zeigten keine
Abweichungen. Je Vergleich waren **1.280 von 1.280 numerischen Werten exakt
gleich**, darunter die **225 Antwortwahrscheinlichkeiten der gewählten
Queries**, bei absoluten und relativen Vergleichstoleranzen von null.
Nicht gewählte rekursive Auswertungsbäume wurden nicht vollständig verglichen.

Nach der Entfernung bestanden **154 Tests**. Die unabhängige Kontrolle
bestätigte, dass alle 40 Quell- und Testdateien dem erfolgreichen Stand vor
diesem Experiment bytegenau entsprachen. Die neuen Cache-Module, ihre Tests
und ihre Aktivierungsschalter sind entfernt.

**Leitlinie für weitere Optimierungen:** Diese beiden Caches nicht unverändert
erneut implementieren. Ein neuer Versuch braucht vorher Messbelege für
wesentlich mehr exakt wiederholte Eingaben oder einen konkreten Ansatz, der
die Kosten der Wiederverwendung reduziert. Die vorhandenen erfolgreichen
Caches sind davon unabhängig zu bewerten.

#### Nachweise

Die vollständigen lokalen Benchmark-Artefakte liegen im Projekt unter
`outputs/benchmarks/bench-i7/20261012-g5-a5-shared-geometry-probability-cache/`:

- `summary.json`, `per_problem.csv`, `operations.csv` und `report.html` enthalten
  Messwerte, Cache-Zähler und Ergebnisvergleiche.
- `input.json`, `run_benchmark.py`, `baseline-implementation.patch`,
  `geometry.patch`, `probability.patch` und `both-caches.patch` dokumentieren
  die Eingaben, den Messablauf und die eingefrorenen Implementierungsstände.
- `cleanup.json` dokumentiert die anschließende vollständige Entfernung.
  Ältere Angaben zum bloßen Deaktivieren der Caches in den historischen
  Berichten beschreiben einen Zwischenstand und werden durch diese Entfernung
  abgelöst.

Implementierungsbranch: `codex/vertex-support-cache`; gemeinsamer Basiscommit:
`6c48de1b9868e87eacc0a2caa4905a9e268cdc2e` plus die archivierten Änderungen.
Die wesentlichen Messwerte und die Entscheidung sind hier im Repository
festgehalten, unabhängig von der Verfügbarkeit der lokalen Artefakte.

## Korrigierter Sampler

Der Hit-and-Run-Sampler startet nicht mehr am beliebigen Machbarkeitspunkt des
LP-Solvers. Dieser Punkt war auf dem Simplex typischerweise eine Ecke und konnte
die Kette, insbesondere bei sieben Zielen, für viele Schritte festhalten.

Jetzt wird zuerst ein relativer Chebyshev-Mittelpunkt im durch die Gleichungen
definierten affinen Raum bestimmt. Optional laufen mehrere unabhängige Ketten;
`sample_points_with_diagnostics(...)` liefert dazu Anzahl eindeutiger Samples,
effektive Stichprobengröße und Split-R-hat. Die optimierte Konfiguration steuert
dies über `sampling_chain_count`, die Terminierungsanalyse verwendet
standardmäßig vier Ketten.

## Exakte Optimierungen

Diese Änderungen sind standardmäßig aktiv und verändern weder Query-Menge noch
Lookahead-Tiefe:

1. **Quotientenintervalle je Zustand nur einmal berechnen.** Die
   Kandidatenanalyse liefert ihre Intervalle an die Query-Erzeugung weiter.
   Vorher wurden dieselben LPs im selben Zustand ein zweites Mal gelöst.
2. **Geometrische Ecken-Engine.** Für jeden Kandidaten wird dessen
   Optimalitätspolytope einmal in den affinen Gleichungsraum projiziert. Aus
   seinen Ecken folgen danach sämtliche Quotientenintervalle. Degenerierte,
   unbeschränkte oder numerisch unsichere Fälle fallen automatisch auf die
   bisherige exakte LP-Engine zurück. Für A/B-Messungen kann
   `ratio_interval_engine="lp"` gesetzt werden.
3. **Sample-Punkte als exakte Machbarkeitszeugen verwenden.** Liegen Samples
   strikt auf beiden Seiten einer Query, sind keine zusätzlichen LPs zur
   Bestimmung der unterstützten Antworten nötig. Bei einem einseitigen Zeugen
   wird nur die gegenüberliegende Schranke optimiert.
4. **Zustandsraum nicht je Query neu aufbauen.** Die aktuelle
   `LinearConstraintSystem`-Instanz wird an die Query-Auswertung weitergegeben.
5. **Solver-Matrizen cachen.** Die Listen der Nebenbedingungen werden einmal in
   NumPy-Arrays umgewandelt und bis zur nächsten Mutation wiederverwendet.
6. **Sampling vektorisieren.** Die zulässigen Schrittintervalle des
   Hit-and-Run-Samplers werden mit Matrixoperationen berechnet; die Matrizen
   werden nicht mehr in jedem Schritt neu aufgebaut.
7. **Prozesspool und Zustandsresultate wiederverwenden.** Die öffentliche
   `OptimizedValueFunctionSession` hält Worker am Leben und besitzt einen
   begrenzten LRU-Cache für identische Aufrufe ohne explizit übergebene Samples.
   Zusätzlich cached `analyze_state(...)` Machbarkeit, Kandidaten und
   Quotientenintervalle unabhängig von der Lookahead-Tiefe. Semantisch gleiche
   gespiegelte Queries teilen sich einen Cache-Key.
8. **Kindzustände vorfiltern.** Bereits berechnete Elternintervalle entfernen
   Kandidaten, die eine konkrete Query-Antwort unmöglich erfüllen können. Der
   Kindaufruf übernimmt außerdem den schon aufgebauten und auf Machbarkeit
   geprüften Gewichtsraum.

## Optionale approximative Begrenzungen

Die folgenden Schalter sind standardmäßig `None`. Sie können große Fälle stark
beschleunigen, dürfen aber die gewählte Query verändern und müssen deshalb
gegen Lösungsqualität und Anzahl der Folgefragen evaluiert werden:

- `max_query_candidates_per_state=N` bewertet nur die `N` Queries mit den
  ausgewogensten Sample-Partitionen.
- `adaptive_depth_candidate_threshold=K` verwendet Tiefe 1, solange mehr als
  `K` Kandidaten verbleiben, und schaltet danach auf die angeforderte Tiefe
  zurück.

In der Terminierungsanalyse heißen die zugehörigen CLI-Optionen
`--max-query-candidates` und `--adaptive-depth-candidate-threshold`.

## Ratio-, Quantil-, Entropie- und Regret-Politiken

Posterior-Quantil-Queries sind opt-in. Beispielsweise ergänzt
`posterior_quantile_levels=(0.25, 0.5, 0.75)` für jedes kanonische Zielpaar die
drei entsprechenden Sample-Quantile zu den Ratio-Queries. Die exakte
Value-Function kann danach alle Queries bewerten.

Für größere Query-Mengen gibt es zwei experimentelle Shortlist-Ziele:

- `posterior_query_objective="entropy"` minimiert die erwartete Entropie des
  posterioren Gewinner-Kandidaten.
- `posterior_query_objective="regret"` minimiert den erwarteten posterioren
  Entscheidungs-Regret.

`posterior_query_shortlist_size=N` begrenzt dabei nur die zusätzlichen
Grid-/Quantil-Queries. Alle geometrisch erzeugten Ratio-Queries bleiben in der
Auswertung. Die Shortlist ist approximativ und sollte anhand Laufzeit,
Folgefragen und finalem Regret evaluiert werden.

## Dimension-7-Benchmark

Der neue Benchmark erzeugt deterministische 7-Ziele-/10-Alternativen-Fälle und
schreibt optional alle Messwerte einschließlich Samplerdiagnostik und
Profiling-Zählern nach JSON:

```bash
python multistep/scripts/benchmark_dimension_seven.py --preset quick

python multistep/scripts/benchmark_dimension_seven.py \
  --preset standard \
  --output-json multistep/data/benchmarks/dimension7.json
```

`quick` misst standardmäßig Tiefe 1 auf einem Problem. `standard` umfasst drei
Probleme, Tiefe 1 und 2 sowie Ratio, Ratio+Quantil, Entropie, Regret und
Grid+Ratio. Einzelne Dimensionen lassen sich mit `--depths`, `--policies`,
`--samples`, `--chains`, `--workers` und `--ratio-engine` isolieren.

## Profilierung im Code

Für gezielte Messungen kann die Kernfunktion ohne Monkey-Patching instrumentiert
werden:

```python
from multistep.optimized import collect_optimization_profile

with collect_optimization_profile() as profile:
    result = compute_value_function_optimized(...)

print(profile.counters)
print(profile.seconds_by_operation)
```

Wichtige Zähler sind `state_calls`, `ratio_interval_batches`,
`query_evaluations`, `query_support_checks`, `sampling_calls` und
`branch_checks`.

## Gemessener Effekt

Einzelmessungen auf dem Entwicklungsrechner mit dem oben beschriebenen
a5/a10-Fall:

| Variante | Vorher | Nachher | Ergebnis |
|---|---:|---:|---|
| optimiert, seriell | 7,287 s | 3,436 s | Wert 1,453; gleiche beste Query |
| optimiert, 4 Worker | 2,774 s | 1,906 s | Wert 1,453; gleiche beste Query |

Auf einer zufälligen 7-Ziele-/10-Alternativen-Instanz benötigte ein kompletter
Quotientenintervall-Batch 0,029 s mit Geometrie gegenüber 1,083 s mit LP
(37-fach). In einem kurzen Dimension-7-Tiefe-2-Lauf mit 240 Samples und vier
Workern lagen Ratio, Entropie-Shortlist und Regret-Shortlist bei 2,26 s, 2,79 s
und 3,09 s; die vollständige Ratio+Quantil-Menge benötigte 5,47 s. Diese
Einzelwerte dienen nur als Smoke-Benchmark, nicht als statistische
Qualitätsaussage.

Die Werte sind keine belastbare Hardware-Benchmarkserie, zeigen aber die
Größenordnung. Für Vergleiche sollten mehrere Wiederholungen mit
`--repeats N` genutzt werden. `--reuse-worker-pool` misst dabei den
Session-Pfad.

## Korrektheitsprüfung

Neben der vollständigen Testsuite werden geometrische Quotientenintervalle in
Zufallstests gegen direkt gelöste LPs verglichen. Der Sampler besitzt
Regressionstests für den relativen Simplex-Innenpunkt, die Exploration in
Dimension 7 und seine Diagnostik. Die approximativen Schalter besitzen eigene
Tests, bleiben aber bewusst opt-in.

## Laufende Serie: 20 weitere Tiefe-3-Probleme (2026-10-13)

Nach dem erfolgreichen Pilotfall wird auf Bench i7 eine feste Serie von
**20 weiteren Problemen, Fälle 2 bis 21**, ausgeführt. Pilotfall 1 ist nur zur
Provenienz in den Eingaben enthalten und bleibt aus der Serienauswertung
ausgeschlossen. Die Eingaben setzen die historische RNG-Folge mit Seed
`20260902` ohne Auswahl oder erneutes Ziehen fort; alle 21 gespeicherten Fälle
wurden unabhängig bitgenau reproduziert. Verwendet wird weiterhin Basiscommit
`03ac83fda7ed04e28f9782672f2dd49ec5700668` mit der oben dokumentierten
Innenpunktkorrektur; der Fingerprint der 87 eingefrorenen Python-Dateien ist
`ad6bccef4dfbccfd19c1360b10a9fb8f07d4d7bf686c3d7521b6cf05c6aa3053`.

Die feste Konfiguration lautet **fünf Ziele, fünf Alternativen, Grid 5 plus
Ratios, Tiefe 3 und acht Worker**. Auf verbleibenden Tiefen größer als 1 werden
Grid- und Ratio-Queries berücksichtigt, auf Tiefe 1 Ratio-Queries. Die
Antwortwahrscheinlichkeiten werden weiterhin mit `exact_volume` und der
geometrischen Ratio-Engine berechnet. Das 99%-Volumen-Abbruchkriterium, alle
Toleranzen und die Begrenzung auf höchstens 100 tatsächlich gestellte Fragen
bleiben erhalten. Der Nutzer hat die bisherige Laufzeitgrenze ausdrücklich
aufgehoben: `case_timeout_seconds=0` lässt jeden Fall ohne Zeitlimit bis zum
Ergebnis beziehungsweise einem dokumentierten Fehler laufen.

Der unabhängig vom AI-Run gestartete Controller speichert den Serienzustand
atomar sowie Fragen, Fortschritt und Ergebnisse in erhaltenen Verzeichnissen
je Fall und Versuch. Ein erneuter Start überspringt bereits abgeschlossene,
über ihre Hashes validierte Fälle. Ein durch Rechner- oder Prozessausfall
unterbrochener Fall wird in einem neuen Versuch am **Problemanfang** neu
gestartet; innerhalb einer laufenden LP-Berechnung wird kein Zustand
wiederhergestellt. Fehlgeschlagene oder unvollständige Versuche bleiben mit
ihren Fehlern und Zwischenständen erhalten und zählen nicht als gelöste
Probleme. Der Nenner der Serie bleibt stets 20.

Die geprüfte lokale Momentaufnahme vom **2026-10-13, 02:43:45 UTC** zeigt den
ersten zusätzlichen Fall 2 in Bearbeitung: zwei tatsächliche Fragen sind
gespeichert, die dritte Planung läuft. Acht Worker sind beobachtet worden;
die gespeicherten Journale enthalten zu diesem Zeitpunkt keine Fehler.
**Noch kein Serienfall ist abgeschlossen**, 19 weitere Fälle warten. Daraus
werden weder fertige Laufzeiten noch Aussagen zur Query-Ersparnis oder
Lösungsqualität der gesamten Serie abgeleitet.

Anleitung für Live-Status und Fortsetzung:
`outputs/benchmarks/bench-i7/20261013-depth3-series20-grid5-ratios-workers8/README.md`.
Im selben lokalen Artefaktverzeichnis werden `report.md` und
`series-summary.json` laufend neu aus den gespeicherten Daten erstellt;
`series-state.json` und `attempts/pXX/aNN/` bewahren den Controllerzustand und
die Rohdaten. Diese dynamischen Berichte zeigen den jeweils neuesten
gesicherten Stand; dieser Dokumentationsabschnitt hält ausschließlich die
oben bezeichnete Momentaufnahme fest.

## Abschluss der 20 weiteren Tiefe-3-Probleme (2026-10-13)

Die oben erhaltene Momentaufnahme wird durch diesen Serienabschluss ergänzt.
Der Controller beendete die festen Fälle 2 bis 21 mit
`completed_with_failures`: **17 von 20 Problemen wurden korrekt gelöst
(85 Prozent)**, drei scheiterten ohne gültiges Endergebnis. Unter den 17
fertigen Auswahlen gab es keine falsche Auswahl; der anhand der eingefrorenen
Zielgewichte unabhängig berechnete Nutzenverlust war jeweils null. Pilotfall 1
bleibt ausgeschlossen, der Nenner bleibt 20. Es gab keine laufenden oder
wartenden Fälle mehr.

Konfiguration und Quellstand blieben während der Serie unverändert: fünf Ziele,
fünf Alternativen, Grid 5 plus Ratios, Tiefe 3, acht Worker mit `forkserver`,
ein nativer Thread je Bibliothek, `exact_volume`, geometrische Ratio-Engine,
99-Prozent-Volumenregel und maximal 100 tatsächlich gestellte Fragen.
`case_timeout_seconds=0` ließ die Fälle ohne Laufzeitgrenze bis zum Ergebnis
oder Fehler laufen. Die 87 eingefrorenen Python-Dateien behielten den oben
angegebenen Fingerprint `ad6bccef…caa3053`.

Die folgenden Statistiken beziehen sich ausschließlich auf die **17 gültigen
Lösungen** und deren verstrichene Lösungszeit einschließlich Session- und
Workerpoolstart/-ende. Imports und Eingabeparsing im Hauptprozess sowie
Berichtsdateien sind ausgeschlossen; die externe Beobachtung ist enthalten.
Kumulierte Workerintervalle überlappen diese Zeiten und werden nicht addiert.

| Kennzahl | Erfolgreiche Lösungszeit | Tatsächliche Queries |
| --- | ---: | ---: |
| Summe | 22.403,098633 s | 171 |
| Median | 975,10 s | 8 |
| Mittelwert | 1.317,83 s | 10,06 |
| Minimum | 620,21 s | 3 |
| Maximum | 4.177,64 s | 42 |

Das 25-Prozent-Quantil der Erfolgszeiten beträgt **847,48 s**, das
75-Prozent-Quantil **1.320,12 s** und das 90-Prozent-Quantil **2.074,33 s**;
die Quantile verwenden lineare Interpolation bei `(n-1)*p`. Der frühere Pilot
mit zwei Queries liegt unter dem Serienmedian von acht und ist für den
Fragebedarf dieser Serie nicht typisch. Diese Serie enthält keinen frischen
Tiefe-2-Vergleich.

**Vier Abschlüsse hatten einen numerischen Einzelkandidaten**, 13 erfüllten
die gespeicherte 99-Prozent-Volumenregel mit verbleibenden Restkandidaten.
Die Volumenabbrüche garantieren keine mathematisch eindeutige Auswahl für
jeden zulässigen Gewichtsvektor. Auch der Einzelkandidatenstatus gilt innerhalb
der bestehenden numerischen Toleranzen. Die Prüfungen glichen gespeicherte
Geometriewerte und deren arithmetische Konsistenz ab; sie berechneten keine
LPs oder Volumina erneut und lieferten keinen symbolischen Nullvolumenbeweis.
Nutzenmaximum und tatsächliche Antwortvorzeichen wurden dagegen unabhängig
aus den gespeicherten Eingaben geprüft.

| Fehlgeschlagener Fall | Beobachtete Prozesszeit | Gespeicherte Teilantworten |
| --- | ---: | ---: |
| 7 | 3.194,1706 s | 44 |
| 12 | 4.611,3404 s | 75 |
| 16 | 3.458,2364 s | 71 |

Alle drei meldeten
`row-normalized center LP returned an invalid center or radius`.
Die gleiche Meldung belegt keine identische numerische Ursache. Die Versuche
besitzen kein gültiges Lösungsergebnis und bleiben aus allen Erfolgsaggregaten
ausgeschlossen. Ihre insgesamt **11.263,7473373 s Prozesszeit und 190
Teilantworten** werden separat erhalten; diese Zeiten sind keine Lösungszeiten.

Der gesamte Controllerlauf dauerte von **2026-10-13, 02:35:28.768852 UTC**
bis **11:56:58.999442 UTC** insgesamt **33.690,230590 s**. Das ist die
verstrichene Serienzeit mit erfolgreichen und fehlgeschlagenen Versuchen sowie
Controller- und Prozessaufwand, kein kumulierter Workerzeitwert.
Die Abschlussprüfung bestätigte **93 primäre Artefakte bytegleich auf Mac und
Bench**, unveränderte **87 Python-Dateien** und keine noch lebende der
**221 erfassten historischen Prozessidentitäten**. Controller und aufgezeichnete
Solvergruppen waren beendet. Es gab keine Solverwiederholung, keinen Retry,
Commit oder Push im Zuge dieses Serienabschlusses.

Die vollständigen lokalen Nachweise liegen im Hauptprojektarbeitsverzeichnis:
[Serien-README](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-depth3-series20-grid5-ratios-workers8/README.md),
[Einzelfallbericht](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-depth3-series20-grid5-ratios-workers8/report.md),
[terminale Zusammenfassung](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-depth3-series20-grid5-ratios-workers8/series-summary.json),
[unabhängige Serienabschlussprüfung](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-depth3-series20-grid5-ratios-workers8/final-series-independent-review.json)
und [Sicherungs- und Cleanup-Prüfung](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-depth3-series20-grid5-ratios-workers8/terminal-raw-backup-review.json).

## Gepaarter Vergleich der 17 Tiefe-3-Erfolgsfälle mit Tiefe 2 (2026-10-13)

Im Anschluss an die abgeschlossene Serie wurden alle 17 erfolgreichen Fälle
mit derselben eingefrorenen Quelle `ad6bccef…caa3053`, denselben Eingaben und
demselben Replay-Adapter erneut mit Tiefe 2 gemessen. Grid 5 plus Ratios,
Ratio-Queries auf der letzten Ebene, acht Worker, native Threads jeweils 1,
`exact_volume` und die vorhandene 99-Prozent-Abbruchregel blieben gleich.
Die Solver-Einstellungen unterscheiden sich ausschließlich in der Tiefe.

Beide Strategien wählten in allen 17 Fällen dieselbe korrekte Alternative.
Tiefe 2 benötigte insgesamt **126,277269 s und 139 Queries**, Tiefe 3
**22.403,098633 s und 171 Queries**. Die Mediane lagen bei **5,144835 s /
6 Queries** beziehungsweise **975,103562 s / 8 Queries**. Tiefe 3 benötigte
in vier Fällen weniger, in drei gleich viele und in zehn mehr Queries.
Tiefe 2 endete in zehn Fällen mit einem Kandidaten und in sieben über die
Volumenregel; bei Tiefe 3 waren es vier und 13.

Dies ist eine einmalige gepaarte Messung auf dem nach Tiefe-3-Erfolg
ausgewählten Teilkollektiv, kein Erfolgsquotenvergleich auf allen 20 Fällen.
Die Fehlerfälle 7, 12 und 16 sowie Pilot 1 bleiben ausgeschlossen. Der
Arbeitsverzeichnis-Unterschied des neuen Controllers wurde statisch geprüft;
Quell-/Eingabepfade sind explizit und die Anfangszustände bitgleich.
Beobachter- und Betriebssystemaufwand sind nicht separat quantifiziert.

Der Planungsscore minimiert erwartete Restkandidaten nach einem festen
Horizont; er minimiert nicht die gesamte Queryzahl bis zur 99-Prozent-Regel.
Nach jeder tatsächlichen Antwort wird der volle Horizont erneut geplant.
Die gespeicherten ersten Tiefe-3-Scores waren in allen 17 Fällen kleiner als
die Tiefe-2-Scores; dies garantiert keine kleinere tatsächliche Queryzahl.

Nachvollziehbare Aggregate und Rohdaten-Hashes:
`benchmarks/20261013_depth2_vs_depth3_matched_success17.json`. Alle Rohdaten,
Journale, unabhängigen Prüfnachweise und Sicherungsnachweise liegen bytegleich
auf Mac und Bench im Verzeichnis
`outputs/benchmarks/bench-i7/20261013-depth2-matched-depth3-success17-grid5-ratios-workers8`.
Die abgeschlossene Statusautomatik bleibt deaktiviert.

Dieser Stand einschließlich Innenpunktkorrektur, ihrer elf erfolgreichen
Regressionstests und der vollständigen bisherigen Benchmark-Dokumentation
wird vor dem folgenden separaten Strategieexperiment als lokaler Commit
gesichert. Das Experiment verändert die Queryauswahl; der feste bisherige
Algorithmus bleibt als Vergleichsbasis erhalten.
