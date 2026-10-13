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
