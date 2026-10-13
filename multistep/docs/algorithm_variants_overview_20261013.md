# Übersicht der Algorithmusvarianten und Messungen

Stand: 13. Oktober 2026. Bestehende Ergebnisse ausgewertet; keine neuen Benchmarks, Algorithmusänderungen, Commits oder Pushes.

Die wichtigste aktuelle Vergleichsreihe umfasst dieselben **100 Probleme, fünf Ziele, fünf Alternativen, Grid5 + Ratios, acht Worker und exact_volume**. Alle drei Strategien haben den tatsächlichen 99%-Stop. Feste Tiefe 2 ist die Referenz, adaptive Rate ist in dieser Reihe am schnellsten, adaptive Rate mit Planungsstop benötigt insgesamt die wenigsten Antworten.

**Leseregel:** Zeiten sind vollständige Problem-Laufzeiten, keine summierten Workerzeiten. Queries sind tatsächlich beantwortete Fragen. Messwerte verschiedener Problemkollektive dürfen nicht als Rangliste oder Beschleunigung gegeneinander gerechnet werden. Wiederholungen vergrößern nicht die Zahl unabhängiger Probleme.

## 1. Varianten und Charakteristika

| ID / Kurzname | Bewertung und Querysatz | Einordnung |
| --- | --- | --- |
| V01 Original D2/G10 | (E2,E1) lexikographisch; Grid10 + Ratios, letzte Ebene Ratios | Referenz vor den hier dokumentierten Codeoptimierungen |
| V02 Optimiert D2/G10 | Gleiche Strategie wie V01; Caches, LP-/Geometrie-/Ratio-Code effizienter | Codeoptimierung, geprüfte Queryverläufe unverändert |
| V03 Optimiert D2/G5 | Wie V02, aber Grid5 statt Grid10 | Konfigurationsänderung; Queryverläufe können sich ändern |
| V04 Grid-only D2/G10 | Höhere Ebene nur Grid; gleicher Ratio-map-Vorfilter; letzte Ebene Ratios | Anderer zulässiger Querysatz |
| V05 D3/G5 | (E3,E2,E1) lexikographisch; höhere Ebenen Grid5 + Ratios; letzte Ebene Ratios | Andere Planungstiefe; kostspielige Serie mit 3 Fehlerfällen |
| V06 Adaptive Rate | (C − E[C_terminal]) / E[Queries]; Horizonte 1/2 komplett bewerten | 99%-Stop im realen Ablauf, nicht in hypothetischer Planung |
| V07 Adaptive Rate + Planungsstop | (C − E[Loss_terminal]) / E[Queries]; hypothetischer 99%-Stop absorbierend | Beendeter Zustand: Loss 1; echtes E[C] bleibt separat |
| V08 Feste D1/Ratio-only | min E[C] nach einer Query; nur Ratios | Greedy Referenz, auf dem 100er-Satz häufig Querysatz erschöpft |

V01–V07 verwenden in den aktuellen Bench-i7-Serien exakte geometrische Volumenberechnung, vorhandene numerische Toleranzen und tatsächliches Ende bei Einzelkandidat oder 99%-Volumenanteil. „Exact_volume“ bezeichnet die geometrische Methode; numerische Gleitkommarechnung und das 99%-Konfidenzkriterium bleiben bestehen. Es ist kein Nachweis symbolischer Eindeutigkeit.

## 2. Aktueller direkter Strategievergleich: dieselben 100 Probleme

Grid5 + Ratios auf Ebene 2; Ratio-only auf Ebene 1; acht maximale Worker; native Threads je Prozess 1; höchstens 100 tatsächlich gestellte Fragen; kein Zeitlimit. Ein frischer Lauf pro Fall/Strategie, Reihenfolge rotiert. Beide adaptiven Horizonte und die Protokollierung zählen zur Laufzeit.

| Strategie | Gesamt s | Mittel s/Problem | Median s | Queries gesamt | Mittel Queries | Max Queries | Korrekt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V03: feste Tiefe 2 | 713,061 | 7,131 | 5,392 | 1010 | 10,10 | 34 | 100/100 |
| V06: adaptive Rate | 611,543 | 6,115 | 5,143 | 928 | 9,28 | 44 | 100/100 |
| V07: adaptive Rate + Planungsstop | 2.390,720 | 23,907 | 19,048 | 824 | 8,24 | 38 | 100/100 |

V06 gegenüber fester D2: **14,24% weniger Rechenzeit und 8,12% weniger Antworten**. V07 gegenüber V06: **11,21% weniger Antworten, aber 3,91-fache Rechenzeit**. V07 gegenüber fester D2: 18,42% weniger Antworten und 3,35-fache Rechenzeit. Das sind Kohortenergebnisse, keine Garantie für jedes Problem.

Abschlüsse Einzelkandidat/Volumen-Konfidenz: feste D2 **41/59**, V06 **90/10**, V07 **39/61**. Null-Query-Fälle bleiben im 100er-Nenner. Alle 100 Gewinner je Strategie korrekt; keine Fehler oder ungelösten Fälle.

Eine frühere Messrunde auf exakt denselben Inputs hatte 719,625 s für D2 und 621,498 s für V06; Querysummen ebenfalls 1010/928. Die neuere Reihe oben ist der maßgebliche gemeinsame Vergleich einschließlich V07. Beide Zeitrunden bleiben erhalten und werden nicht gemischt.

Belege: [20261013_query_efficiency_stop_aware99_workers8.json](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/benchmarks/20261013_query_efficiency_stop_aware99_workers8.json); [comp.md](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-query-efficiency-stop-aware99-problems100-grid5-ratios-workers8/comp.md).

### Feste Tiefe 1: erfolgreiche Fälle und Ausfälle getrennt

V08 wurde auf denselben 100 Inputs in der früheren Runde ausgeführt: **22/100 korrekt gelöst**, 77 endeten ohne verfügbare Query, Fall 61 blieb nach 100 Antworten ungelöst. Keine falsche fertige Auswahl.

Nur die 22 erfolgreichen Fälle: 11,210 s gesamt, 0,510 s/Problem, 80 Queries bzw 3,636/Problem. Das ist **kein fairer Laufzeit- oder Queryvergleich gegen die 100 erfolgreichen D2-Fälle**. Die 77 Ausfälle verbrauchten zusammen 88,725 s Prozesszeit und 580 Teilantworten; der ungelöste Fall 2,920 s Solverzeit/100 Antworten. Teilantworten sind keine gelösten Ergebnisse.

Beleg: [comp.json](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-query-efficiency-depth12-problems100-grid5-ratios-workers8/comp.json).

## 3. Grid und Tiefe: eigene passende Vergleichsgruppen

Die beiden Grid-Experimente verwenden dieselben zehn gespeicherten Probleme mit je drei Wiederholungen. Die folgende „Gesamtzeit“ ist jeweils die **Summe der zehn Fallmediane**; Queryzahlen gelten pro Wiederholung. Fünf Ziele/fünf Alternativen, D2, acht Worker, exact_volume und 99%-Stop.

| Vergleich | Variante | Summe Fallmediane s | Mittel Fallmediane s | Queries gesamt | Queries/Fall | Korrekt |
| --- | --- | --- | --- | --- | --- | --- |
| Gridgröße | V02 G10 + Ratios | 108,428 | 10,843 | 75 | 7,50 | 10/10 |
| Gridgröße | V03 G5 + Ratios | 55,456 | 5,546 | 71 | 7,10 | 10/10 |
| Queryquelle / gleicher Vorfilter | V02 G10 + Ratios | 108,406 | 10,841 | 75 | 7,50 | 10/10 |
| Queryquelle / gleicher Vorfilter | V04 G10-only | 120,265 | 12,027 | 83 | 8,30 | 10/10 |

Grid5 spart in diesem Zehnersatz 48,85% Zeit und vier Antworten; alle gewählten Alternativen sind korrekt und identisch zu Grid10, aber nur ein Queryverlauf ist identisch. Grid-only mit fairer Vorfilterung benötigt insgesamt 10,94% mehr Zeit und acht Antworten mehr. Der ältere Grid-only-Test ohne denselben Vorfilter dient nicht als fairer Entscheidungsvergleich.

Belege: [20261013_grid5_vs_grid10_ratios_workers8.json](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/benchmarks/20261013_grid5_vs_grid10_ratios_workers8.json); [20261013_grid_prefilter_parity_workers8.json](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/benchmarks/20261013_grid_prefilter_parity_workers8.json).

### Tiefe 2 gegenüber Tiefe 3: dieselben 17 erfolgreich gelösten D3-Fälle

| Variante | Fälle | Gesamt s | Mittel s/Problem | Median s | Queries gesamt | Queries/Problem |
| --- | --- | --- | --- | --- | --- | --- |
| V03 D2/G5 | 17 | 126,277 | 7,428 | 5,145 | 139 | 8,18 |
| V05 D3/G5 | 17 | 22.403,099 | 1.317,829 | 975,104 | 171 | 10,06 |

Auf diesem Teilkollektiv braucht D3 **177,41-mal so viel Zeit und 23,02% mehr Antworten**. Beide ausgewählten Gewinner sind je Fall korrekt. D3 stellt in vier Fällen weniger, in drei gleich viele und in zehn mehr Queries.

Die vollständige D3-Serie hatte 20 geplante Probleme (feste Fälle 2–21, Pilot 1 ausgeschlossen): **17 korrekt gelöst, drei numerische Fehler, keine falsche fertige Auswahl**. Fehlerfälle 7/12/16: 3194,171/4611,340/3458,236 s Prozesszeit und 44/75/71 Teilantworten. Alle scheitern an einem ungültigen Innenpunkt oder Radius des normierten Center-LP. Diese 190 Teilantworten zählen nicht zu den 171 Queries erfolgreicher Lösungen. Die Fehlerprozesszeiten sind separat von den Solverzeiten gespeichert.

D2 wurde nur auf den 17 erfolgreichen D3-Fällen nachgemessen; für die drei D3-Fehlerfälle gibt es hier keinen passenden D2-Vergleich. Der erfolgreiche reparierte Pilot 1 (D2: 4,519 s/5Q; D3: 734,258 s/2Q) gehört nicht zur 20er-Serie.

Belege: [20261013_depth2_vs_depth3_matched_success17.json](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/benchmarks/20261013_depth2_vs_depth3_matched_success17.json); [series-summary.json](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-depth3-series20-grid5-ratios-workers8/series-summary.json).

## 4. Reine Codeoptimierungen: gleiche Strategie, gleiche Workerzahl

### Ursprung gegenüber Abendoptimierungen; derselbe Fünfersatz

Fünf Ziele/fünf Alternativen, Grid10 + Ratios, Tiefe 2, exact_volume und 99%-Stop. Ein Lauf je Bedingung/Fall; alle fünf korrekt und aufgezeichnete Semantik identisch.

| Quellstand | Worker | Gesamt s /5 Fälle | Mittel s/Problem | Queries gesamt |
| --- | --- | --- | --- | --- |
| V01 Original (6c48de1) | 4 | 220,454 | 44,091 | 27 |
| V02 Abend (074fbf9) | 4 | 80,899 | 16,180 | 27 |
| V01 Original (6c48de1) | 8 | 138,853 | 27,771 | 27 |
| V02 Abend (074fbf9) | 8 | 50,224 | 10,045 | 27 |

Codeoptimierung allein: Faktor 2,73 bei vier bzw. 2,76 bei acht Workern. Workerwechsel allein: Faktor 1,59 bzw. 1,61. Die ursprüngliche Vier-Worker-Zeit darf nicht als Vergleich für eine spätere Acht-Worker-Codeoptimierung verwendet werden.

Beleg: [20261012_g5_a5_workers4_vs8.json](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/benchmarks/20261012_g5_a5_workers4_vs8.json).

### Welche Optimierungen behalten wurden

| Stand | Änderung | Beleg für Nutzen |
| --- | --- | --- |
| Abend (074fbf9) | Support aus vorhandenen Eckpunkten; Cache affiner Zerlegungen; Wiederverwendung LP-Setup bei kaltem Solverzustand | Frische Zehnerpaare: 569,653→405,218 s; 404,548→378,952→206,832 s; jeweils 75 Queries |
| Nacht S01 | Weniger Constraint-/Schlüsselverwaltung, Byte-Deduplizierung, kompakte LP-Ergebnisse, direkte CSC-Matrizen | Wiederholte gepaarte Zehnersätze |
| Nacht S02 | Bereits bekannte einzelne Supportgrenze wiederverwenden | Wiederholte gepaarte Zehnersätze |
| Nacht S03 | Ratiointervalle gebündelt/vorbereitet berechnen | Wiederholte gepaarte Zehnersätze |
| Nacht S06 | Kontinuierliche LP-Ergebnisse gezielt prüfen | Wiederholte gepaarte Zehnersätze |
| Tageszusatz E12 | Direkte float64-Infinity-Masken | Kleiner gemessener Zusatznutzen; behalten |
| 03ac83f | Fairer Ratio-map-Vorfilter im Grid-only-Modus | Vergleichsprotokoll korrigiert; Standard bleibt Grid + Ratios |
| 252cab5 | Normierung und Validierung des geometrischen Innenpunkts | Reparierter Pilot; drei weitere D3-Fehler bleiben ausdrücklich bestehen |

Direkter frischer Vergleich Abend (074fbf9)→Endstand S06+E12, acht Worker:

| Zehnersatz | Median Zehner-Gesamtzeit Basis s | Median Endstand s | Zeitreduktion | Queries unverändert |
| --- | --- | --- | --- | --- |
| Seed20260902 | 132,937 | 104,872 | 21,11% | 75 |
| Seed20261012 | 121,630 | 97,920 | 19,49% | 73 |

Vier Kontrollläufe und drei Kandidatenläufe je Satz; Median vollständiger Zehner-Gesamtzeiten.75 und 73 Queries beziehen sich auf **unterschiedliche Seeds**, nicht auf eine Änderung durch die Optimierung. Geprüfte rekursive Bewertungen und Gleitkommabits identisch. Stufengewinne und Gewinne verschiedener Messrunden nicht addieren.

Beleg: [20261012_overnight_daytime_summary.json](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/benchmarks/20261012_overnight_daytime_summary.json); [performance_optimizations.md](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/performance_optimizations.md).

### Ursprung gegenüber Endstand nach Dimension

Eigene frische gepaarte Runde: fünf Alternativen, Grid10 + Ratios, D2, acht Worker. Andere Fälle als die 100er-Strategiestudie.

| Ziele | Gepaarte Fälle | Original Mittel s | Optimiert Mittel s | Queries gesamt je Variante | Queries/Fall |
| --- | --- | --- | --- | --- | --- |
| 3 | 3 | 3,110 | 1,178 | 14 | 4,67 |
| 5 | 3 | 26,476 | 8,142 | 16 | 5,33 |
| 7 | 2 | 269,817 | 102,801 | 24 | 12,00 |

Alle acht Paare korrekt und aufgezeichnete Semantik identisch. Der dritte schwere 7D-Fall wurde auf Nutzerwunsch übersprungen. Die separate vollständige 3/3/3-Dimensionsmessung darf nicht als gleiches Vergleichskollektiv verwendet werden:

| Ziele | Fälle | Optimierter Mittelwert s | Queries gesamt | Queries/Fall |
| --- | --- | --- | --- | --- |
| 3 | 3 | 0,920 | 14 | 4,67 |
| 5 | 3 | 7,382 | 16 | 5,33 |
| 7 | 3 | 296,079 | 73 | 24,33 |

Alle neun korrekt; die 7D-Zeile enthält einen zusätzlichen schweren Fall mit 687,245 s und 49 Queries. Timer-/Counter-Replay mit unquantifiziertem Instrumentierungsaufwand; kein cProfile.

Belege: [20261012_original_vs_current_workers8.json](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/benchmarks/20261012_original_vs_current_workers8.json); [20261012_current_dimension_timings.json](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/benchmarks/20261012_current_dimension_timings.json).

### Nicht übernommene Performanceversuche

| Versuch | Ergebnis | Status |
| --- | --- | --- |
| Zusätzlicher gemeinsamer Geometriecache | 206,091→217,941 s im Zehnersatz/vier Worker; 0 Treffer | Entfernt |
| Zusätzlich vollständiger Wahrscheinlichkeitscache auf Geometriecache | 226,802 s; +10,05% zur Referenz; Probability-only nicht separat gemessen | Entfernt |
| E10 lokale Normwiederverwendung | Kleine unsichere Unterschiede −0,15%/−0,41% | Verworfen |
| E09 Chunkgröße 2/4 | Kein robuster Gewinn, eher langsamer | Verworfen |
| E07/E11/E13/E16 | Lokale Prüfung ohne belastbaren Remote-Laufzeitgewinn | Nicht übernommen |
| E08 vorbereitete Volumenarrays | Zurückgestellt; kein gemessener Remote-Misserfolg | Offen/historisch |
| E15 größere Cachekapazität; E18 Hotspot-Audit | Beobachtung ohne übernommene Änderung | Keine neue Variante |

Alle gemessenen Zusatzcache-Varianten: 75 tatsächliche Queries / zehn korrekte Fälle. Aktuell vorgeschlagene Wiederverwendung von Nebenbedingungen und Machbarkeitsnachweisen ist **noch nicht implementiert oder als Beschleunigung gemessen**. Die vier jüngsten Profilingläufe sind Diagnose, keine zusätzliche Performancevariante.

Belege: [performance_optimizations.md](/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm/multistep/docs/performance_optimizations.md); [report.md](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261013-query-efficiency-performance-audit/report.md).

## 5. Branches und wiederherstellbare Stände

Branchspitzen sind lokale Referenzen zum Zeitpunkt dieses Inventars. Tracking-Refs wurden nicht neu vom Server abgerufen. Ein Branchname allein bezeichnet keine vollständige Benchmarkkonfiguration; Modus, Grid, Tiefe und eingefrorener Quellfingerprint gehören dazu.

| Branch | Lokaler Commit | Charakteristik / Sicherungsstand |
| --- | --- | --- |
| `codex/depth3-e2-safety-band` | `e25c78c` | Historischer E2-Guard, separater 3D/10A-Querysatz |
| `codex/expected-reduction-per-query-depth1-depth2` | `1e24d32` | V06 plus feste Referenzmodi; Commit 1e24d32 und Checkpoint-Tag lokal, Git-Bundle Mac+Bench |
| `codex/grid10-ratio-depth2-vs-depth3` | `6c48de1` | Ursprüngliche exakte Grid-Version, V01 |
| `codex/multistep-performance-optimizations` | `e69ed15` | Älterer Optimierungs-/Kapazitätszweig; Vorfahr der neueren Linie |
| `codex/query-efficiency-stop-aware99` | `1e24d32` | V07 zusätzlich als uncommittete Änderungen; vollständiger frozen Source/Patch/Archive Mac+Bench |
| `codex/stopping-time-rollout-experiment` | `8991e56` | Historischer approximierter Stoppzeit-Rollout |
| `codex/vertex-support-cache` | `252cab5` | V02–V05, Grid-/Tiefe-Modi; lokale Spitze 252cab5, Tracking 03ac83f: 1 Commit voraus |
| `main` | `c82e87a` | Hauptzweig/Queryanalyse; kein heutiger Vergleichs-Benchmarkstand |
| `multistep-intervals` | `286c0f3` | Historische Sampling-Version |

Reihenfolge der Hauptlinie: `6c48de1 → 074fbf9 → 8baeac8 → 03ac83f → 252cab5 → 1e24d32`. Aktiver Arbeitsbaum: `/Users/janoberwahrenbrock/.codex/worktrees/vertex-support-cache/optimal-trade-off-query-algorithm`, Branch `codex/query-efficiency-stop-aware99`. Seine Algorithmusänderungen sind weiterhin uncommittet. Checkpoint-Tag `checkpoint/query-efficiency-depth12-2026-10-13` zeigt auf `1e24d32bed2c2266d417fbc3c887c3e04b245b32`.

Wichtige Quellidentitäten: Grid-Vergleiche `09ef2b5…` (03ac83f), D3 mit Innenpunktkorrektur `ad6bccef…` (03ac83f+Patch, später 252cab5), erster Adaptive-Versuch `7d52e67…`, Stop-aware `1e534ba5…`. Volle Commit-IDs, Fingerprints, Konfigurationen, Input-Hashes und Quellen stehen in `inventory.json`.

## 6. Ältere Forschungsvarianten: separate Konfiguration und Hardware

Diese Zahlen sind **nicht direkt mit Bench i7, fünf Alternativen, Grid5/10 und 99%-Stop vergleichbar**. Historische Exact-/Guard-Fälle verwenden 3 Ziele/10 Alternativen, eine Schwerpunkt-Ratio je kanonischem Zielpaar, Singleton-Stop und lexikographische Kandidatenbewertung; Hardware und Workerzahl sind in den untersuchten Metadaten nicht eindeutig ausgewiesen.

| Historische Variante | Fälle | Gesamtzeit s | Mittel s/Problem | Queries gesamt | Queries/Problem |
| --- | --- | --- | --- | --- | --- |
| Sampling, Tiefe 2,3D/10A | 10 | 108,051 | 10,805 | 49 | 4,90 |
| Sampling, Tiefe 2,5D/10A | 10 | 2.173,905 | 217,391 | 119 | 11,90 |
| Historisch exact D2,3D/10A | 100 | 54,071 | 0,541 | 498 | 4,98 |
| Historisch exact D3,3D/10A | 100 | 268,065 | 2,681 | 545 | 5,45 |
| E2-Sicherheitsband, receding, 3D/10A | 100 | 276,052 | 2,761 | 493 | 4,93 |
| E2-Sicherheitsband, countdown, 3D/10A | 100 | 129,732 | 1,297 | 493 | 4,93 |
| Stoppzeit-Rollout-Pilot, 3D/10A | 1 | 44,116 | 44,116 | 3 | 3,00 |

Sampling: Mac-mini-Log, 400 Samples / Burn-in 200 / Thinning 5, feste D2/G10+Ratios, Einzelkandidat-Stop; Workerzahl im gespeicherten Configblock fehlt. Die beiden Zehnersätze haben 3 bzw. 5 Ziele und jeweils 10 Alternativen. Kein aktueller exakter Gegenlauf auf derselben Hardware/Konfiguration.

Historische exakte D2/D3 und Guard: dieselben 100 Probleme mit 3 Zielen und 10 Alternativen, alle gelöst. Guard receding erlaubt E2 höchstens 0,05 über dem E2-Minimum und wählt darunter minE 3; countdown plant Guard-D3→D2→D1 in Dreierblöcken. Ein belastbarer allgemeiner Querygewinn der fünf eingesparten Antworten ist damit nicht nachgewiesen.

Rollout: Blattwert erwartete Restfragen unter D2-Policy. Der praktische Pilot schneidet Pfade unter 0,01 Wahrscheinlichkeit ab und ist daher eine **Approximation**. Ein Fall/h1: 3 Queries / 44,116 s; passende Pilotreferenzen D2: 3 Queries / 0,384 s, D3: 4 Queries / 2,074 s. Ein H3-Pilot wurde nach 90 s vor erster tatsächlicher Frage abgebrochen, kein fertiges Ergebnis.

Ein älterer 5D/10A-Pilot mit zehn Problemen ergab 152 Queries bei D2 und 138 bei D3. Vier parallele Problemprozesse verhindern die Verwendung der summierten Zeiten als saubere absolute Problem-Laufzeitreferenz.

Belege im Git-Archiv: `286c0f3:multistep/data/termination_runs/{3,5}_goals_runs.txt`; `5c83262:multistep/docs/depth2_vs_depth3_g3_replication_100.md`; `e25c78c:multistep/docs/depth3_e2_safety_band_g3_100.md`; `8991e56:multistep/docs/stopping_time_rollout_pilot.md`.

### Ältere große Original-Grid10-Studien

Im Original-Snapshot 6c48de1 liegen außerdem 450 beendete Fälle vor: feste Tiefe 2, Grid10 + Ratios, letzte Ebene Ratios, exact_volume, 99%-Stop, vier persistente Root-Worker, Seed 20260902. Hardware fehlt in den gespeicherten Metadaten. Die 5D-Sätze kombinieren ältere Fälle 1–25 mit späteren Fällen 26–50.

| Ziele | Alternativen | Fälle | Gesamt s | Mittel s/Problem | Queries gesamt | Queries/Problem | Korrekt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 3 | 100 | 205,116 | 2,051 | 260 | 2,60 | 100/100 |
| 3 | 5 | 100 | 308,768 | 3,088 | 367 | 3,67 | 99/100 |
| 3 | 7 | 100 | 371,826 | 3,718 | 393 | 3,93 | 100/100 |
| 5 | 3 | 50 | 1.580,260 | 31,605 | 397 | 7,94 | 50/50 |
| 5 | 5 | 50 | 2.555,083 | 51,102 | 435 | 8,70 | 50/50 |
| 5 | 7 | 50 | 3.264,792 | 65,296 | 523 | 10,46 | 49/50 |

Zwei fertige Auswahlen waren bei einem Volumen-Konfidenzstopp falsch; alle anderen korrekt. Keine numerischen Ausführungsfehler laut gespeicherter Dokumentation. Die 100/100 korrekten Ergebnisse der jüngsten Serie sind deshalb keine allgemeine Fehlerfreiheitsgarantie für die 99%-Regel. Diese historischen Laufzeiten nicht mit der heutigen Acht-Worker-Reihe verrechnen.

Belege: `6c48de1:multistep/docs/dimension3_grid10_ratio_confidence099_300.md` und `6c48de1:multistep/docs/dimension5_grid10_ratio_confidence099_150.md`; zugehörige Rohdaten unter `multistep/data/d{3,5}_grid10_ratio_depth2_confidence099_a{3,5,7}_p{100,50}_seed20260902.json`.

## Einordnung für die weitere Arbeit

Für die aktuell geprüfte 5×5-Konfiguration ist V06 die günstigste der vollständig erfolgreichen 100er-Strategien hinsichtlich Rechenzeit. V07 reduziert die Antwortzahl weiter, bezahlt dafür erheblich mehr Planung. V03 bleibt die einfache festeD 2-Referenz. Grid5 war im separaten Zehnervergleich besser als Grid10, Grid-only insgesamt schlechter. D3 überzeugt auf dem gemessenen 17er-Teilkollektiv weder bei Laufzeit noch bei Queryzahl und hat zusätzlich drei ungelöste numerische Fehler in der 20er-Serie.

Der nächste reine Performanceversuch soll eine exakt benannte eingefrorene Variante als Referenz verwenden und deren Query-/Stopsemantik beibehalten. Die Übersicht ersetzt keine Messung der noch unimplementierten Optimierungsideen.

## Reproduktion und Umfang

`build_overview.py` liest ausschließlich gespeicherte JSON-Belege und Git-Objekte; es startet keinen Solver und ändert keinen Branch. `inventory.json` bewahrt präzise Werte, Fallkollektive, Quellidentitäten, Grenzen und SHA 256 der verwendeten Belege. Bei erneutem Erzeugen kann der lokale Branchstatus abweichen; die Benchmarkquellen bleiben benannt.

## Nachtrag: feste Tiefe 2 ohne Konfidenzabbruch

Auf denselben 100 Problemen (5 Ziele/5 Alternativen/Grid5+Ratios/8 Worker) lösten beide Modi 100/100 korrekt, ohne Fehler oder Fragenlimit. Bis genau ein Kandidat verbleibt: 1256 Queries / 872,379 s (12,56 Queries / 8,724 s je Problem). Mit 99%: 1010 Queries / 736,571 s (10,10 Queries / 7,366 s). 99% spart 19,586% Queries und 15,568% Zeit. In 41 Fällen gleiche Queryzahl, in 59 zusätzliche Queries ohne 99%. Sämtliche gemeinsamen Verläufe exakt identisch. Details: [Tiefe2-Abbruchvergleich](depth2_singleton_vs_confidence99_experiment.md). Die älteren Zeitreihen oben bleiben eigenständige Messrunden.

## Nachtrag: feste Tiefe 2 mit 95% gegenüber 99%

Frischer Vergleich auf denselben 100 Problemen (5 Ziele/5 Alternativen/Grid5+Ratios/8 Worker): 99% benötigt 1010 Queries / 736,193 s (10,10 Queries / 7,362 s pro Problem), alle 100 Auswahlen korrekt. 95% benötigt 831 Queries / 635,741 s (8,31 Queries / 6,357 s), 99 korrekte und eine falsche Auswahl. Kein ungelöster Fall und kein Ausführungsfehler. 95% spart 17,723% Queries und 13,645% Zeit, einschließlich des falschen Ergebnisses im Kostenvergleich. In 55 Fällen weniger und 45 gleich viele Queries; kein Fall mit mehr Queries. Problem 81 endet bei 95% nach 10 Queries falsch, bei 99% nach 17 korrekt. Alle 100 gemeinsamen Verläufe stimmen exakt bis zum 95%-Abbruch überein. Ausschließlich der tatsächliche Abbruch wurde verändert, keine adaptive oder konfidenzabhängige Querybewertung. Details: [95%-gegenüber-99%-Vergleich](depth2_confidence95_vs99_experiment.md). Die älteren Messrunden bleiben getrennt.
