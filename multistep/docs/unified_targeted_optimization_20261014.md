# Gezielte Optimierung des vereinheitlichten Algorithmus am 14.10.2026

**Abgeschlossene Performanceuntersuchung.** Behalten sind E01 und E13. Der direkte Gesamtvergleich bestätigt rund 9 % weniger Planungszeit mit Konfidenzbewertung; ein zusätzlicher gekreuzter Vergleich ohne Planungskonfidenz zeigt lediglich 1,35 % im Median und erfüllt die 2%-Nachweisgrenze nicht. Alle aufgezeichneten mathematischen Vergleichsergebnisse stimmen bitgenau überein. E16 und die übrigen erfolglosen Prototypen sind verworfen. Den abschließenden Sicherungs- und Automatiknachweis führt `archive-verification.json` separat.

Alle Artefaktpfade in diesem Dokument sind relativ zum Benchmarkordner
`outputs/benchmarks/bench-i7/20261014-unified-targeted-optimization-until-noon`
(im Folgenden **O4**). Ist bereits ein Unterordner genannt, beziehen sich weitere Dateinamen auf diesen Ordner. Dort liegen Hypothesen, eingefrorene Quellen, Testnachweise, Rohresultate, verworfene Patches und Fortsetzungsanweisungen. Die früheren Validierungsserien mit 275, 97 und 72 Aufrufen bleiben unverändert; sie werden durch diese Performancekampagne nicht nachträglich umgeschrieben.

## Ausgangsstand und Wiederherstellung

Der geprüfte vereinheitlichte Algorithmus wurde vor der Optimierung mit Commit
`c6eea313bc24454453b5a32eb19e7e8bfab3c307` auf
`codex/unified-query-algorithm` und dem Tag
`checkpoint/unified-reviewed-2026-10-14` auf origin gesichert. Das annotierte Tag zeigt nach Auflösung auf diesen Commit; seine Tagobjekt-ID ist kein anderer Algorithmusstand. Die Ausgangsquelle enthält 104 Pythondateien mit Fingerprint
`7669ad9a421f8eecd651c7b66f766f86c6c4eff504f79e82bb3217a40f637139`.
Nachweis: `baseline-checkpoint.json`.

Die anschließende Arbeit wurde zunächst lokal und uncommittet auf
`codex/unified-targeted-performance`. Im Arbeitscode wurden ausschließlich E01 und E13 übernommen. Dieser Stand umfasst 107 Pythondateien mit Fingerprint
`7dcdd8251c8a80fbc2585781ea34a750042d09fe59c3bd21cab352ffcdd44fd7`.
Auf genau dieser eingefrorenen Quelle bestanden auf Mac und Bench jeweils **384 Tests: 378 Multistep- und sechs Onestep-Tests**. Die bytegleiche Übernahme wurde anschließend anhand des Fingerprints geprüft. Zum Abschluss der Untersuchung waren diese Optimierungen noch nicht committed oder gepusht. Der anschließende Sicherungscommit auf `codex/unified-targeted-performance` nimmt E01 und E13 samt Tests und Dokumentation in den regulären Git-Stand dieses Branches auf.

Der vollständige Wiederherstellungsstand liegt unter
`current-local-checkpoint-v3-final/receipt.json`: Er enthält den Patch für getrackte Änderungen und alle acht geänderten oder neuen Dateien, einschließlich ungetrackter Tests, Fixture und dieses Abschlussdokuments. Die Wiederherstellung aus dem tatsächlichen gesicherten Commit mit Patch und vollständigen Dateien wird separat geprüft. Zum Wiederherstellen auf dem gesicherten Commit sind sowohl `tracked-diff.patch` als auch sämtliche im Receipt aufgeführten `files/`-Pfade erforderlich. Ein Patch allein würde die neuen Dateien auslassen. Danach sind die Einzelhashes und der Gesamtfingerprint zu prüfen. Der ältere reine E01-Snapshot unter `current-local-checkpoint/` und der sieben Dateien umfassende E13-Snapshot unter `current-local-checkpoint-v2-e13/` bleiben erhalten. Der vollständige Hashabgleich von Quellen, Rohdaten, Dokumentation und Snapshots auf Mac und Bench ist durch `final-archive-manifest.json` und `archive-verification.json` belegt, sobald letzterer `status=passed` ausweist.

Weitere Nachweise: `experiments/e13-e01-validated-lp-cleaning/candidate-source-manifest.json`, die beiden `bench-*-tests.log` im selben Ordner sowie `templates/e01-validated-lp-cleaning/full-multistep-tests.log` und `full-onestep-tests.log`.

## Unveränderter Vergleichsrahmen

Die Performancevergleiche verwenden fünf Ziele, fünf Handlungsalternativen, Grid 5 plus Ratio-Queries bei verbleibender Tiefe größer eins und ausschließlich Ratios auf der letzten Ebene. Beide Seiten jedes Paares verwenden acht Worker, geometrische `exact_volume`-Wahrscheinlichkeiten, dieselbe Tiefe, Strategie und Konfidenzregel. In den Konfidenzmodi beträgt die Grenze 99 %. Querymengen, positive Antwortzweige, Auswertungsziel und Abbruchregeln werden nicht reduziert. E01 und E13 führen keine neue Approximation ein.

Für fünfdimensionale Tiefe-3-Läufe werden vollständige **Planungsschritte an echten gespeicherten Zuständen** gemessen. Beispielsweise bezeichnet `five:p001:q08` Problem 1 nach acht bereits tatsächlich gestellten Antworten. Der Test rekonstruiert diesen Zustand und bewertet die vollständige nächste Entscheidung; dabei werden null neue Nutzerfragen gestellt. `planning_seconds` misst die Planung, `one_step_session_seconds` zusätzlich die frische Session, Zustandsrekonstruktion und Poolbeendigung. Diese Zeiten sind keine Zeit bis zur vollständigen Lösung des 5D-Problems. Kleine vollständige 3D-Läufe sind gesonderte Kontrollen.

Zeitvergleiche erfolgen erst nach Gleichheit der vollständig aufgezeichneten mathematischen Felder mittels `float.hex`, ohne Toleranz. Dazu gehören die Query-Identitäten und Reihenfolgen, Ties, Antwortzweige, Kandidaten, gespeicherte Volumina und Shares, E[C], erwarteter Loss, E[Q], Raten und Fortsetzungsbäume. Der Planungsharness erfasst am tatsächlichen Root Kandidatenzahl und maximalen Share; eine vollständige Root-Volumenvektorerfassung wird daraus nicht behauptet. Laufzeitfelder werden getrennt gehalten. Fehler und Zeitüberschreitungen bleiben sichtbar und werden nicht zu fertigen Ergebnissen umetikettiert.

Die Untersuchung belegt Ergebnisgleichheit für die konkret geprüften Eingaben und Pfade. Sie ist keine pauschale Bitgarantie für jede zukünftige Eingabe oder Bibliotheksversion. Die überprüften Sonderfall- und Override-Fallbacks gehören zur Implementierung und zu den Regressionstests.

## E01: unnötige terminale Intervallmodelle vermeiden

Am Ende einer stop-aware Planung werden verbleibende Kandidaten, deren Volumina und normalisierte Anteile benötigt. Vollständige Ratio-Intervallmodelle dienen dort keinem weiteren Query-Schritt. E01 verwendet deshalb eine private terminale Kandidatenzusammenfassung und spart den Aufbau der unbenutzten Intervalldaten und Pydantic-Ergebnismodelle.

Die relevanten Kandidatenprüfungen, native Geometrie, LPs, Volumina, Reihenfolgen, Summierung und Sonderfallbehandlung bleiben erhalten. Zustände, die noch für weitere Queries gebraucht werden, erhalten weiterhin die vollständige Analyse. Die unabhängige Prüfung berücksichtigt auch überschreibbare Geometrie-, LP- und Membership-Hooks: Bei solchen Abweichungen wird der ursprüngliche Analysepfad verwendet. E01 wurde nach strikter Parität und vollständigen ungeprofilten Vergleichsläufen übernommen.

| Modus und gespeicherter Zustand | Ausgangsstand → E01 | Verringerung | Messumfang |
| --- | ---: | ---: | --- |
| Adaptive Tiefe 3, Konfidenz in Planung, p001:q08 | 275,486 → 254,661 s | 7,56 % | Zwei gekreuzte Paare; beide mindestens 5 % schneller |
| Gleicher Modus, p081:q17 | 325,666 → 303,377 s | 6,84 % | Ein weiteres Paar |
| Gleicher Modus, p028:q37 | 189,147 → 176,367 s | 6,76 % | Ein weiteres Paar |

Die zusätzlichen Zustände stützen den Befund, ihre einzelnen H3-Paare belegen aber keine eigene Wiederholungsstabilität. Wiederholungen desselben Zustands sind keine zusätzlichen unabhängigen Probleme. Alle aufgezeichneten Pläne der Vergleichspaare stimmen bitweise überein.

Nachweise: `experiments/e01-terminal-summary/decision.json`, `GENERALIZATION.md` im selben Ordner, `e01-plan-01.json`, `e01-plan-02.json`, `e01-generalization-plan-01.json` und deren Rohdaten/Audits. Der unabhängige Implementierungsreview liegt unter `templates/terminal-candidate-summary/independent-review-v2.json`.

## E13: LP-Eingaben effizienter vorbereiten

E13 ergänzt E01 um eine cachefreie private Vorbereitung gewöhnlicher interner kontinuierlicher LP-Eingaben. Sie vermeidet generische Formatierungsschritte, erhält jedoch Form-, Typ- und Endlichkeitsprüfungen sowie unabhängige frische Kopien. Ungewöhnliche, ungültige oder überschriebene Eingaben verwenden die ursprüngliche SciPy-Bereinigung.

Die nachfolgenden native Solveraufrufe, Koeffizienten, CSC-Reihenfolge, Optionen, kaltes Zurücksetzen und Resultat-/Fehlerprüfungen bleiben erhalten. Es werden keine LPs ausgelassen, kein Warmstart und kein Eingabe-Cache ergänzt.

| Modus und gespeicherter Zustand | E01 → E01+E13 | Zusätzliche Verringerung | Messumfang |
| --- | ---: | ---: | --- |
| Adaptive Tiefe 3, Konfidenz in Planung, p001:q08 | 254,730 → 247,565 s | 2,81 % | Zwei gekreuzte Paare |
| Gleicher Modus, p028:q37 | 177,772 → 172,166 s | 3,15 % | Zwei gekreuzte Paare |
| Adaptive Tiefe 2, Konfidenz in Planung, p001:q00 | 5,516 → 5,295 s | 4,01 % | Zwei gekreuzte Kontrollpaare |
| Adaptive Tiefe 3 ohne Planungskonfidenz, p001:q08 | 92,782 → 89,709 s | 3,31 % | Ein Kontrollpaar |
| Adaptive Tiefe 3, vollständiges 3D-Problem p002 | 29,773 → 28,744 s | 3,46 % | Ein vollständiges Kontrollpaar |

Alle 16 Aufrufe und acht Paare sind vollständig geprüft; sämtliche erfassten mathematischen Felder stimmen bitweise überein. Alle vier Zielpaare und beide Zielmediane bestehen die **vor Beginn festgelegte 2%-Grenze**. Die übrigen Kontrollen bestehen die 10%-Regressionsgrenze. Das vollständige 3D-Kontrollproblem stellt in beiden Varianten zwei Queries und endet beim korrekten Einzelkandidaten 3 mit `exact_single_candidate`. Daraus folgt keine Laufzeitverteilung für viele 3D- oder vollständige 5D-Probleme.

Die gleiche cachefreie Technik wurde zuvor isoliert als E02B auf dem Ausgangsstand ohne Planungskonfidenz geprüft und verworfen: Dort war der Gewinn nicht stabil oberhalb der damaligen vorab festgelegten 5%-Grenze. Dieser negative Befund bleibt unverändert. E13 ist ein eigener, vorab dokumentierter Versuch auf E01 mit zuvor nicht vollständig gemessener stop-aware 5D-H3-Last. Er enthält den zuvor geprüften Patch unverändert; die alten Messungen und damaligen Kriterien werden nicht nachträglich angepasst.

Nachweise: `experiments/e13-e01-validated-lp-cleaning/decision.json`, `BENCHRESULTS.md` und `root-final-audit.json` im selben Ordner, `e13-plan-01.json` sowie `templates/e01-validated-lp-cleaning/independent-review.json`.

## Direkter Gesamtvergleich: vollständig geprüft

E01- und E13-Gewinne aus verschiedenen Messreihen werden nicht multipliziert. `combined-verification-plan-01.json` vergleicht den gesicherten Ausgangsstand 104 direkt mit dem behaltenen E01+E13-Stand 107: adaptive und feste Tiefe 3 mit Konfidenz in der Planung, jeweils zwei gekreuzte Paare; dazu ein adaptives H3-Paar ohne Planungskonfidenz als Kontrolle. Alle zehn Aufrufe verwenden denselben gespeicherten Zustand p001:q08 und sind vollständig beendet. Es gab keinen Fehler und keine Zeitüberschreitung.

| Modus | Ausgangsstand → E01+E13, Median | Weniger Planungszeit | Messumfang |
| --- | ---: | ---: | --- |
| Adaptive maximale Tiefe 3 mit Konfidenz in Planung | 272,930 → 247,172 s | 9,437 % | Zwei gekreuzte Paare |
| Feste Tiefe 3 mit Konfidenz in Planung | 271,742 → 247,013 s | 9,100 % | Zwei gekreuzte Paare |
| Adaptive maximale Tiefe 3 ohne Planungskonfidenz | 93,534 → 90,158 s | 3,609 % | Ein Kontrollpaar |

Alle fünf Alt/Neu-Paare liefern vollständig gleiche aufgezeichnete mathematische Felder und exakte Float-Bits. Bei den Zielmodi bestehen jedes der vier Paare und beide Mediane die vorab festgelegte Grenze Neu/Alt≤0,98. Die Summe der beiden Zielmodemediane fällt von 544,672 auf 494,185s, entsprechend 9,269% weniger Zeit. Dieses Aggregat umfasst zwei Modi desselben gespeicherten Zustands, keine Verteilung unabhängiger Probleme. Das gewöhnliche adaptive Kontrollpaar besteht Neu/Alt≤1,10; sein einzelner 3,609%-Gewinn ist kein wiederholter Nachweis.

Die vollständigen Root-Auswertungszahlen sind unverändert: Bei adaptiver maximaler Tiefe 3 werden 12/100/100 Queries für Horizonte1/2/3 bewertet, bei fester Tiefe 3 insgesamt 100. Beide adaptiven Quellen wählen hier dieselbe Horizon1-Query, nachdem alle Horizonte geprüft wurden. Der Test stellt keine neue tatsächliche Query und ist weiterhin ein Vergleich vollständiger 5D-Planungsschritte, keine vollständige 5D-Problemlösung.

Der lokale `--require-complete`-Audit bestätigt alle zehn fertigen Aufrufe, Eingabe-/Quell-/Rohhashes und vollständige Parität. Die unabhängige zusätzliche Prüfung bestätigt zudem die globale Reihenfolge ohne Überlappung sowie ausdrücklich die 2%-Einzelpaar-, Median- und Aggregat-Gates und die 10%-Kontrollgrenze. Die generischen 5%-Flags des gemeinsamen Auditors ersetzen diese Plan-Gates nicht. Der Root hat Controller 348631/start83782795 und sämtliche zugehörigen Prozessgruppen nach Ende als beendet geprüft.

Nachweise: `independent-reviews/combined-plan-and-profiler-01/final-audit.json`, `combined-benchmark-review.json` und `BENCHRESULTS.md` im selben Reviewordner. Letzteres enthält alle zehn einzelnen ungeprofilten Planungszeiten. Dieser abgeschlossene Vergleich ist ein frischer gemeinsamer Nachweis für E01+E13; die ergänzende Profilierung und der gesamte Sicherungsabschluss bleiben davon getrennt.

## Ergänzender gekreuzter Vergleich ohne Planungskonfidenz

Nach dem einzelnen Kontrollpaar im Gesamtvergleich wurde vor weiteren Messungen `ordinary-adaptive3-verification-plan-01.json` mit vier ungeprofilten Aufrufen deklariert: Alt/Neu, anschließend Neu/Alt; derselbe p001:q08-Zustand, dieselben vollständigen adaptiven Horizonte 1/2/3 und acht Worker. Es wurde kein weiterer Code geändert.

| Wiederholung | Ausgangsstand | E01+E13 | Verringerung |
| --- | ---: | ---: | ---: |
| 1, Alt→Neu | 89,621 s | 89,537 s | 0,094 % |
| 2, Neu→Alt | 92,727 s | 90,343 s | 2,572 % |
| Median | 91,174 s | 89,940 s | 1,354 % |

Alle vier Aufrufe sind vollständig beendet, beide Pläne bitgleich und alle eingefrorenen Eingaben und Quellen unverändert. Die vorab festgelegte mindestens 2%-Verbesserung in jedem Paar und im Median wird nicht erreicht. Der frühere einzelne 3,609%-Kontrollgewinn wird dadurch nicht zu einem robusten allgemeinen H3-Gewinn erklärt. Für ordinary adaptive H3 ist der belegte Nutzen klein und schwankend; die Beibehaltung von E13 beruht auf seinen separat erfüllten stop-aware Zielmessungen. Auch hier werden null neue Fragen gestellt und keine vollständigen 5D-Probleme gelöst.

Nachweise: `ordinary-adaptive3-verification-final-audit.json` und `independent-reviews/ordinary-adaptive3-final-01/`. Die Root-Prozessprüfung bestätigt den beendeten Controller 351383/start84348125 und sämtliche beendeten Solvergruppen.

## Abschließende Diagnose des behaltenen E01+E13-Stands

Profile04 ist vollständig geprüft: zwei Diagnosen mit je acht vollständigen originalen H3-Teilbäumen, unverändertem Grid 5 plus Ratios, acht Workern und exakter 99%-Konfidenzplanung. Die aktuelle Quelle umfasst 107 Dateien (`7dcdd825…`); die ältere Profile03-Referenz umfasst ausschließlich E01 mit 106 Dateien. **Alle 16 Teilbäume stimmen mathematisch bitgenau überein.** Diese Teilbaumprüfung erzeugt keinen vollständigen Root-Entscheid und keine neue tatsächliche Query.

Die folgende Tabelle zerlegt ausschließlich die aufgezeichnete **instrumentierte Worker-Profiler-Self-Zeit**. Die jeweiligen Summen von 155,932 und 843,187 Sekunden stimmen mit `Stats.total_tt` überein; jeweils 537 Methoden gehören genau einer Kategorie an. Das sind keine normalen Algorithmuslaufzeiten und keine Wallclock-Prozentsätze. Rundung kann die dargestellte Summe geringfügig verändern.

| Kategorie | p001:q08 | p001:q00 |
| --- | ---: | ---: |
| Allgemeine NumPy-Operationen | 17,84 % | 20,22 % |
| Native LP-Ausführung | 14,09 % | 11,26 % |
| Private LP-Hülle, Modell- und Matrixvorbereitung | 13,98 % | 13,69 % |
| Geometrievorbereitung und Validierung | 8,85 % | 10,56 % |
| Volumen- und Halfspace-Hüllen einschließlich nativer Qhull-Arbeit | 14,35 % | 16,89 % |
| Übrige disjunkte Kategorien zusammen | 30,88 % | 27,40 % |

Die private E13-Eingabebereinigung ist im tatsächlichen Profilerlauf aktiv: 120.315 beziehungsweise 626.042 Worker-LP-Aufrufe liefen durch `_clean_private_inputs`; die ursprüngliche SciPy-Eingabebereinigung hatte in diesen Worker-Teilbäumen null Aufrufe. Die LP-Aufrufzahlen stimmen mit Profile03 überein. E01 bleibt mit 30.843 beziehungsweise 101.466 terminalen Kandidatenzusammenfassungen aktiv. Der ursprüngliche Infinity-Helper ist unverändert aktiv: 438.116 beziehungsweise 2.471.348 Aufrufe, entsprechend 109.529 beziehungsweise 617.837 Modell-Misses nach dem unveränderten Vier-Aufruf-Vertrag. E16 ist nicht Teil dieser profilierten Quelle. Die Null-Fallbackzahl ist keine Garantie für andere Eingaben oder komplette Läufe.

Allgemeine NumPy-Arbeit lässt sich aus diesen aggregierten Profilen nicht exakt LPs oder Geometrie zuordnen. Native Qhull-Arbeit bleibt in ihren Hüllen enthalten; verschachtelte inklusive Zeiten werden nicht addiert. Die Diagnose begründet keinen weiteren Speedup, keine vollständige 5D-Lösung und keine Laufzeitverteilung unabhängiger Probleme. Maßgeblich für Performanceentscheidungen bleiben die ungeprofilten vollständigen Alt/Neu-Messungen.

Nachweise: `profile-final/independent-final-review-e13.json` (SHA `6229d35e…`), `exclusive-profile-breakdown-e13.json`, `PROFILE-DETAILS-e13.md` und `FINAL-PROFILE-REVIEW-e13.md` im selben Ordner sowie `profile-plan-04-e13.independent-subtree-audit.json`. Die älteren Profile03-Dokumente bleiben unverändert erhalten.

## Historische Diagnose vor E13

Die ausführliche Diagnose betrifft **E01, Quelle 106 vor E13**. Acht vollständige ausgewählte H3-Teilbäume werden pro Zustand profiliert; es wird kein vollständiger Root-Entscheid und keine vollständige Problemlösung berechnet. `PROFILE-DETAILS.md` und `profile-design/exclusive-profile-breakdown.json` zerlegen die aufgezeichnete cProfile-Selfzeit ohne Überschneidungen: jede Funktion gehört genau einer Kategorie an, und die Summe stimmt mit `Stats.total_tt` überein.

In den beiden ausgewerteten Zuständen entfallen ungefähr 29–33 % dieser instrumentierten Selfzeit auf die ausdrücklich zugeordnete LP-Pipeline, 22–27 % auf Geometrievorbereitung und die Volumen-/Halfspace-Hüllen sowie 19–21 % auf allgemeine NumPy-Operationen. Diese letzten Operationen können ohne zusätzlichen Stackkontext nicht exakt LPs oder Geometrie zugeordnet werden. Native Qhull-Arbeit ist in den jeweiligen Hüllen enthalten. Schon die acht ausgewählten Teilbäume des Anfangszustands enthalten rund 626.000 native LP-Aufrufe und 509.000 Volumenberechnungen. Die Kombination der kleinen Pareto-Frontiers ist dagegen kaum relevant.

Die Prozentsätze sind **keine Anteile der normalen Laufzeit**. Verschachtelte inklusive Funktions-/Operationszeiten dürfen nicht addiert werden; kumulierte Workerzeiten sind keine Root-Wanduhrzeit. Nur ungeprofilte vollständige Alt/Neu-Läufe begründen Performanceentscheidungen. Die Profile sind außerdem kein neuer Kostenbefund für den späteren107-Datei-Stand.

Nachweise: `profile-design/exclusive-profile-breakdown-independent-review.json`, `profile-design/kept-e01-subtree-hotspot-review.json` sowie `independent-reviews/combined-plan-and-profiler-01/review.json`. Die Einzelmethoden, Aufrufzahlen und Profilhashes stehen im Breakdown-JSON.

## Warum weitere Versuche nicht übernommen wurden

**Lokale Komponentengewinne trugen nicht bis zur vollständigen Planung.** E03 wiederverwendete dieselbe Konditionierung innerhalb eines Geometrieaufrufs; E04 deduplizierte kleine Vertexmengen über exakte Bytes; E06 stellte terminale Kandidatenmitgliedschaft früher fest. Alle hatten sinnvolle lokale Hinweise, ihre vollständigen H3-Vergleiche blieben jedoch wirkungslos oder schwankten: E03 im Median +0,77 %, E04 +0,23 %, E06 +0,032 % Laufzeit. E05 erzeugte bei identischen nativen Qhull-Aufrufen weniger unbenutzte Metadaten, machte die H3-Planung aber reproduzierbar 6,04 % langsamer. Die gespeicherten Pläne waren bitgleich; die Laufzeitbefunde rechtfertigten die zusätzliche Komplexität nicht. Keine dieser Änderungen wurde übernommen. Entscheidungsbelege: `experiments/e03-call-local-conditioning/decision.json`, `experiments/e04-exact-small-row-dedup/decision.json`, `experiments/e05-minimal-qhull-metadata/decision.json`, `experiments/e06-terminal-membership-simplex/decision.json`.

**Mehr Caches lösten die gemessene Last nicht.** E02A speicherte vorbereitete LP-Eingaben und war bereits lokal langsamer; E02B blieb in seiner ursprünglichen vollständigen Messung unter dem stabilen 5%-Ziel. E11 simulierte einen größeren bestehenden HiGHS-Modellcache auf zwei vollständigen tatsächlichen LP-Traces. Fast alle Misses waren neue Koeffizientensysteme: 128 statt 32 Einträge ergaben nur 0,606 % zusätzliche Hitgelegenheiten, und die native LP-Berechnung würde auch bei diesen Treffern weiterlaufen. Der ursprüngliche Cache bleibt erhalten. Belege: `templates/validated-lp-cleaning/candidate-a-rejection-and-b-hypothesis.json`, `experiments/e02-validated-lp-cleaning/decision.json`, `templates/highs-model-cache-capacity/decision.json`.

**Einige vermeintlich redundante Berechnungen existieren im Zielpfad bereits nicht.** E08 fand keine angeblich wiederholten Ratio-Feasibility-LPs. E09 fand in 235 beobachteten Kandidatengeometrien keinen identischen Query-Geometrie-Key; die Kandidatenbereiche enthalten zusätzliche Optimalitätsrestriktionen. E10s normale terminale Kandidatenanalyse war im vereinheitlichten Lauf bereits durch direkte Intervallzählung ersetzt. Deshalb entstand kein neuer Cache oder Algorithmuspatch. Belege: `templates/ratio-normalized-fallback-reuse/rejection-receipt.json`, `templates/query-local-geometry-reuse/rejection-receipt.json`, `templates/ordinary-leaf-terminal-summary/rejection-receipt.json`.

**Mathematische Wiederverwendung allein beweist keine unveränderten nativen Fehlerpfade.** E07 konnte vorhandene Eckpunkte in exakte Machbarkeitszeugen überführen. Die Prüfkosten waren jedoch je Zustand schneller, gleich schnell oder langsamer; außerdem beweist ein rational gültiger Zeuge nicht, dass ein übersprungener nativer LP dieselbe Status-/Fehlerantwort gegeben hätte. E12s spezialisierte Zeilennorm bewahrte die Arithmetik, wurde durch ihre Guards aber rund 28–29 % langsamer. E15 wollte die Größenbegrenzung des bereits verworfenen Membership-Shortcuts erhöhen, fand jedoch keinen neuen tatsächlichen Workload oberhalb der bisherigen Grenze. Belege: `templates/cached-feasibility-certificate/decision.json`, `templates/exact-row-norm-dispatch/rejection-receipt.json`, `templates/terminal-membership-large-array-cap/rejection-receipt.json`.

**E18 zeigte echte Musterwiederholung, aber keinen nutzbaren Gesamtpfad.** Anders als die kompletten Modellkeys aus E11 wiederholten sich die exakten CSC-Null-/Nichtnullmuster in knapp 98 % von 24.782 tatsächlichen Modell-Miss-Aufrufen. Der Versuch speicherte ausschließlich kleine ganzzahlige Koordinaten-/Indexmetadaten in einem auf 32 Einträge und 512 KiB begrenzten thread-/PID-getrennten Cache. Jeder aktuelle Koeffizient wurde neu in Originalreihenfolge eingesammelt, alle zurückgegebenen Arrays blieben frische Kopien. Sämtliche CSC-Ausgaben waren byte-, Layout- und Schreibbarkeitsgleich.

Der vollständige erste Guardpfad war trotzdem im Median 6,81 % beziehungsweise 0,37 % langsamer als die ursprüngliche Vorbereitung. Ein separat vorab deklarierter engerer zweiter Versuch bewahrte einen strikten nativen Getter-/Globals-Vertrag, sparte wiederholte Dtype-Metadaten und blieb trotzdem 3,00 % beziehungsweise 2,55 % langsamer. Beide unterschritten die geforderte 15%-Komponentenverbesserung deutlich und wurden komplett verworfen. Zehn isolierte Matrix-/Hook-/Cachetests bestanden; es gab keine neue breite Suite, keinen Bench-Plan und keine Übernahme. Der schnelle ungeschützte Vorversuch ist ausdrücklich kein Produktions- oder Achtworker-Gewinn. Belege: `templates/csc-pattern-reuse/decision.json`, `final-rejection-manifest.json` im selben Ordner, `versions/v1-full-guard-rejected/` und `versions/v2-getter-guard-probe/`.

Diese Entscheidungen gelten für die geprüften Lasten und Verträge, nicht als allgemeine Unmöglichkeitsbeweise. Ein neuer Versuch benötigt eine belegbar andere Last oder neue konkrete Kostenhypothese und eigene vorher festgelegte Grenzen; die alten negativen Resultate bleiben erhalten.

## Weitere Befunde und Kampagnenabschluss

E14 war eine Diagnose der nativen Threadaktivität, keine Solveränderung. Acht Worker erzeugen zusätzlich HiGHS-Scheduler-Hilfsthreads; in drei realen Messfenstern rechneten die acht Worker-Hauptthreads, während die Helfer null CPU-Ticks zeigten. Das ist kein Beleg für versteckte zusätzliche Rechenparallelität in diesen Fenstern, aber auch keine allgemeine Aussage für alle Läufe. Solveroptionen blieben unverändert. Beleg: `templates/native-thread-audit/decision.json`.

**E16 ist nach vollständiger Makroprüfung verworfen.** Die separat freigegebene E13-Kombination sollte vier nach tatsächlicher LP-Eingabevalidierung bitweise wirkungslose Infinity-Zuweisungen sparen. Nach den notwendigen Provenanz-/Hook-Guards liefert sie in zwölf vollständigen Aufrufen und sechs Paaren dieselben mathematischen Ergebnisse; der Laufzeitgewinn fehlt jedoch:

| E16-Vergleich | E01+E13 → E16 | Laufzeitwirkung | Umfang |
| --- | ---: | ---: | --- |
| Adaptive H3 mit Planungskonfidenz, p001:q08 | 247,396 → 247,384 s | −0,004745 %, praktisch gleich | Zwei gekreuzte Paare |
| Adaptive H3 ohne Planungskonfidenz, p001:q08 | 89,927 → 90,603 s | +0,752 % | Zwei gekreuzte Paare |
| Adaptive H2 mit Planungskonfidenz, p001:q00 | 5,500 → 5,863 s | +6,605 % | Ein Kontrollpaar |
| Vollständiges 3D-Problem p002 | 28,930 → 28,643 s | −0,991 % | Ein Kontrollpaar |

Alle vier Zielpaare sowie beide Zielmediane und Zielaggregate scheitern an der vorher festgelegten Grenze Neu/Alt ≤ 0,98. Die Kontrollen bestehen die 10%-Regressionsgrenze, begründen aber keine Übernahme. Beide 3D-Quellen stellen zwei Queries, wählen korrekt Alternative 3 und enden beim Einzelkandidaten. Zehn 5D-Aufrufe bleiben vollständige Planungsschritte mit null neuen tatsächlichen Fragen, keine vollständigen 5D-Lösungen.

Der Kandidat source108 `6c2390bb…` hatte 22 neue Tests, bestand 406 Tests je Host und 238 gebundene tatsächliche native LP-Ergebnisvergleiche. Die zusätzlichen netto 175 Backendzeilen (180 eingefügt, fünf ersetzt) und 347 Testzeilen sichern den engen getesteten Vertrag ab, beweisen aber keinen Nutzen für die Laufzeit. E16 wurde nicht übernommen; der Arbeitsstand bleibt E01+E13 source107 `7dcdd825…` mit 384 Tests je Host. Nachweise: `experiments/e16-native-infinity-noop-e13/independent-benchmark-review.json`, `BENCHRESULTS.md`, `independent-full-audit.json` und `root-final-audit.json` im selben Ordner. Vorlagen/Patches/Tests und alle zwölf Makrorohresultate bleiben erhalten.

Die frühen isolierten Standalone-Prototypen v1/v2/v3 wurden wegen konkreter Hook-, Import-, Helper-Globals- oder Auswertungsreihenfolgefragen wieder geöffnet und nicht übernommen. Die zuletzt getestete E13-Kombination ist ein separat freigegebener Quellstand. Ein Versionsfehler beim Zwischenreview ist ausdrücklich dokumentiert: Der ursprüngliche vorläufige v2-Reviewbeleg wurde beim Wiederöffnen überschrieben; seine ursprünglichen Bytes sind nicht mehr verfügbar. Der bekannte alte SHA ersetzt diese fehlenden Bytes nicht, und es wurde kein synthetischer Beleg rekonstruiert. Die Quellen-, Patch-, Test- und Mikroexperimentbelege dieser Version sowie die endgültige Kombination und eingefrorenen Benchrohresultate blieben erhalten. Beleg: `templates/native-infinity-noop/review-versioning-incident-20261014.json`. Neue Befunde und Reviews erhalten seither getrennte versionierte Dateien.

Die zusätzliche aktuelle Profilierung des behaltenen E01+E13-Stands ist inzwischen vollständig geprüft. Der getrennte Diagnosebefund steht im abschließenden Profile04-Abschnitt; er ist keine neue ungeprofilte Laufzeitmessung.

E17 bleibt eine kleine read-only ROI-Hypothese: Eine private gebündelte Kopie unveränderter Constraint-Zeilen könnte wiederholte Dimensionsprüfungen reduzieren, müsste aber Zeilenordnung, Wertebits, unabhängige Kopien und Fehler-/Hook-Verträge exakt bewahren. Der gesamte inklusive Zielbereich beträgt im Profil nur etwa 2,1–2,6 %; notwendige Kopien und neue Guards begrenzen die erreichbare Ersparnis weiter. Kein Prototyp oder Cache wurde angelegt. Beleg: `independent-reviews/combined-plan-and-profiler-01/bulk-constraint-copy-roi.json`.

Bei Unterbrechung sind zuerst `optimization-state.json`, `experiment-log.jsonl`, der aktive Controller-State und Spiegelstatus zu lesen. Controller und Mac-Spiegel laufen unabhängig vom AI-Turn. Die automatische Wiedervorlage `gezielte-algorithmusoptimierung-bis-12-uhr` war während der Kampagne aktiv und wurde zum Abschluss deaktiviert, wenn der Automatikbeleg in `archive-verification.json` dies bestätigt. Für einen erneuten autorisierten Lauf sind der laufende Host und verfügbare Ausführungsmöglichkeiten erforderlich.

Vor einer Fortsetzung sind reale PID, Start-Ticks und alle Controller-/Solvergruppen zu prüfen. Solange eine Gruppe lebt, keinen zweiten Solver oder Bench-Test starten. Erst bei nachweislich beendeten Gruppen den dokumentierten identischen eingefrorenen Plan fortsetzen; fertige Messungen und Versuche erhalten. Quellen, Eingaben und Prüfhilfen laufender Pläne bleiben unverändert. Die Deadline dieser Kampagne ist 14.10.2026, 12:00 Europe/Berlin beziehungsweise 10:00 UTC.

Alle normalen Performancepläne sind vollständig abgeschlossen; es gibt keine offenen Solverfälle dieser Optimierungskampagne. Die frühe Diagnose `profile-plan-01` enthält zwei erhaltene 180-Sekunden-Zeitüberschreitungen sowie den dokumentierten Wechsel zu kurzen vollständigen Teilbaumdiagnosen. Diese Teilresultate begründen keinen Performancegewinn. Die früheren fehlgeschlagenen vollständigen Algorithmusvalidierungen bleiben mit ihren nachfolgenden numerischen Reparatur- und Korrektheitsserien unverändert dokumentiert.

Der unabhängige abschließende Quellenreview bestätigt alle 107 Pythonhashes des Arbeitsstandes, passende 384 Tests je Host und keine übernommenen E16-, E18- oder anderen verworfenen Quellreste: `independent-reviews/final-retained-source-01/final-source-review.json`. Das Abschlussdokument verändert keine Pythonquelle. Wiederherstellungsprüfung, reale Prozessbeendigung, vollständiger Hashabgleich und beendete Statusautomatik werden in `final-report.json`, `current-local-checkpoint-v3-final/receipt.json` und `archive-verification.json` getrennt nachgewiesen. Die neuen Änderungen bleiben uncommittet und ungepusht.

Noch nicht belegt ist die Laufzeitwirkung über eine größere Verteilung frischer vollständiger fünfdimensionaler Probleme. Die Daten rechtfertigen insbesondere keine neue Queryzahlverbesserung und keinen Tiefe-2/Tiefe-3-Qualitätsvergleich: Es wurde ausschließlich dieselbe Berechnung effizienter implementiert und auf den bezeichneten Pfaden verglichen.
