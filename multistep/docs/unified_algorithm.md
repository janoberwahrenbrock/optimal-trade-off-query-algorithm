# Gemeinsamer exakter Queryalgorithmus

Die öffentliche Schnittstelle ist `multistep.optimized.AlgorithmConfig` und
`multistep.optimized.solve`. Sie trennt Problem, Methodenkonfiguration und
Antwortgeber. Der Algorithmuskern erhält keine tatsächlichen Zielgewichte.
Die bisherige Benchmarkfunktion ist ein Adapter desselben Ausführungsloops.

```python
from multistep.optimized import AlgorithmConfig, solve
from multistep.src.query_probability import classify_query_answer

config = AlgorithmConfig(
    query_strategy="reduction_per_query",
    depth=3,
    higher_depth_query_source="both",
    grid_size=5,
    stop_rule="volume_confidence",
    confidence_threshold=0.99,
    consider_stop_in_planning=False,
    workers=8,
    max_queries=100,
)

# Nur der simulierte Antwortgeber kennt target_weights.
def answer_query(query):
    return classify_query_answer(weights=target_weights, query=query, equality_tol=0.0)

result = solve(alternatives=matrix, config=config, answer_query=answer_query)
```

Eine Zeile der Matrix enthält die Zielwerte einer Alternative; die Spaltenzahl
bestimmt die Zielzahl. Der Antwortgeber kann ein Simulator, eine Oberfläche
für eine Befragung oder ein gespeicherter Antwortverlauf sein. Er liefert
`<`, `>` oder `=` für die gewählte Query.

In einem ausführbaren Pythonprogramm gehört der Aufruf bei `workers > 1`
in einen `if __name__ == "__main__":`-Block, damit die Worker das Programm
nicht erneut starten. Das mitgelieferte CLI erledigt dies bereits.

## Die neun Parameter

| Parameter | Werte / Bedeutung |
|---|---|
| query_strategy | `fixed` oder `reduction_per_query` |
| depth | 1, 2 oder 3; feste Tiefe oder maximale adaptive Tiefe |
| higher_depth_query_source | `grid` oder `both` für Grid plus Ratios |
| grid_size | Positive Anzahl Gridwerte; unsere Vergleiche verwenden 5 oder 10 |
| stop_rule | `single_candidate` oder `volume_confidence` |
| confidence_threshold | Schwelle in (0, 1], beispielsweise 0.95 oder 0.99; im Einzelkandidat-Modus inaktiv |
| consider_stop_in_planning | Tatsächliche Abbruchregel bereits in hypothetischen Zuständen berücksichtigen |
| workers | Positive maximale Workerzahl; 1 bedeutet serielle Auswertung |
| max_queries | Positive maximale Gesamtzahl beantworteter Queries oder `None` |

Exakte Volumenberechnung, Ratio-Queries bei verbleibender Tiefe 1,
geometrische Kandidatenanalyse, vorhandene Performanceoptimierungen,
Query-Wertebereich und numerische Toleranzen sind gemeinsam festgelegt.
`config.to_optimized_config()` zeigt die vollständig aufgelösten internen
Einstellungen. Die Samplingfelder dieser bestehenden internen Dataclass sind
bei `exact_volume` inaktiv und gehören nicht zur öffentlichen Schnittstelle.

Bei `single_candidate` beendet der tatsächliche Lauf erst bei genau einem
Kandidaten. Eine Konfidenzschwelle von 1.0 ist kein Ersatz dafür: mehrere
numerisch mögliche Kandidaten können verschiedene oder verschwindende
Volumenanteile haben. Konfidenzanteile beziehen sich auf den verbleibenden
Gewichtsraum und garantieren nicht die richtige Auswahl im einzelnen Fall.

## Bewertung und Tiefe

Die feste Strategie verwendet die ausgewählte Tiefe und den bisherigen
Planner mit seiner lexikographischen Verfeinerung. Bei ausgeschaltetem
Planungsabbruch ist sein Verhalten unverändert.

Die adaptive Strategie betrachtet alle Horizonte 1 bis `depth` und maximiert
`(C_vorher - E[terminaler Loss]) / E[ausgeführte Queries]`. Ohne Konfidenz in der
Planung ist der Loss die tatsächliche Kandidatenanzahl. Mit Planungsabbruch
haben absorbierende Einzelkandidat- oder Konfidenzzustände Loss 1; die echte
erwartete Kandidatenanzahl wird trotzdem separat gespeichert.

Adaptive Tiefe 3 kombiniert die vollständigen Verlust/Kosten-Frontiers der
hypothetischen Tiefe-2-Folgepläne. Die Folgeentscheidungen optimieren den
Bewertungsquotienten des gesamten Plans; eine unabhängige gierige Auswahl
nach dem Quotienten jedes Teilpfads wäre mathematisch anders. Keine
Wahrscheinlichkeitsschwelle, Samplingapproximation oder Query-Shortlist wird
zur Begrenzung eingeführt. Nur Root-Queries werden parallelisiert.

Für feste Tiefe mit aktiviertem Konfidenzabbruch in der Planung gibt es einen
expliziten neuen Planner: Er minimiert den erwarteten terminalen Loss und
entscheidet bei Gleichstand anhand der stabilen Queryidentität. Dies ist ein
neuer Modus ohne historischen Laufzeitgegenlauf. Beim Einzelkandidat-Modus
ist der Planungsabbruch schon im bisherigen festen Planner enthalten; die
Implementierung delegiert deshalb unverändert an ihn.

## Ergebnisse, Beobachtung und vorhandene Antworten

`result` enthält Queryfolge, echte Kandidaten- und Volumenverläufe,
Abbruchgrund, Auswahl und Laufzeiten. Adaptive Records enthalten die gewählte
Tiefe, E[C], E[Q], Gain, Rate und beste Pläne jedes ausgewerteten Horizonts.
Aktivierter Planungsabbruch ergänzt einen separat bezeichneten Loss.

Optional sind `initial_answered_queries`, `on_state` und `on_query` möglich.
Vorhandene Antworten werden als Gewichtsraumbedingungen übernommen.
`max_queries` begrenzt die Gesamtzahl einschließlich dieser Antworten;
`question_count`, `initial_question_count` und `new_question_count` machen
den Unterschied sichtbar. Beobachter sind synchron und ihre Ausführungszeit
gehört zur Solverzeit. Während einer laufenden Rekursion ist kein Checkpoint-
Neustart vorgesehen.

Wenn keine Query mehr verfügbar ist, liefert die öffentliche Schnittstelle
`solved=False`, `termination_reason="no_query"` und keine erfundene Auswahl.
Das Erreichen der Querygrenze liefert `max_questions`. Ungültige Antworten,
unzulässige Einstellungen und numerische Fehler werden ausdrücklich gemeldet.
Der ältere Benchmarkadapter behält seine bisherige No-query-Exception für
bestehende Skripte; seine Berechnungen und Antwortverläufe bleiben dieselben.

## Kommandozeile

Eine JSON-Eingabe enthält `alternatives`, für eine Simulation zusätzlich
`target_weights` und optional bereits vorhandene `answered_queries`.

```sh
python multistep/scripts/run_unified_algorithm.py \
  --input-json problem.json --output-json result.json --simulate \
  --query-strategy reduction_per_query --depth 3 \
  --grid-size 5 --higher-depth-query-source both --workers 8 \
  --stop-rule volume_confidence --confidence-threshold 0.99 \
  --consider-stop-in-planning --max-queries 100
```

`--config-json` kann alle neun Einstellungen als Objekt enthalten. Explizite
CLI-Flags überschreiben die entsprechenden Felder. `--max-queries none`
hebt die Querygrenze auf. Ohne `--simulate` werden die Antworten interaktiv
abgefragt. Seed, Problemgenerierung, Fallzeitlimit und Ergebnisqualität gehören
zum Benchmark beziehungsweise zur umgebenden Anwendung.

## Sicherung und Validierung

Branch `codex/unified-query-algorithm`; Ausgangspunkt `4207421` und Tag
`checkpoint/before-unified-query-algorithm-2026-10-14`. Der Vorgänger enthält
alle bisherigen Stop-aware-Änderungen und Benchmarknachweise.

Die Korrektheits- und Laufzeitvalidierung liegt unter
`outputs/benchmarks/bench-i7/20261014-unified-query-algorithm-validation` im
primären Checkout. Dort liegen unveränderte alte und neue Quellen, Eingaben,
275 geplante Aufrufe in 20 Konfigurationen, alle Versuche und unabhängige
Prüfungen. Drei neue 3D/5A-Fälle sind ausdrücklich von den bisherigen
5D/5A-Fällen und dem vollständigen historischen 5D-Tiefe-3-Pilot getrennt.
Wiederholungen zählen nicht als neue unabhängige Probleme. Konfigurationen
ohne alten Gegenlauf erhalten eigene Messungen; ihre Laufzeit darf nicht als
bewiesene Regression gegenüber einer anders konfigurierten Variante gelten.

<!-- BEGIN UNIFIED_VALIDATION_20261014 -->
## Abschluss der Validierung vom 14. Oktober 2026

Die abgeschlossene Bench-i7-Validierung umfasst **20 Konfigurationen, 14 unterschiedliche Probleme und 275 Aufrufe**. **125 frische Alt/Neu-Paare mit regulärer Rückgabe** stimmen in Queryfolge, Kandidaten-/Volumenverläufen, Bewertungen, Abbruch und Auswahl exakt überein. **3 weitere Paare** enden numerisch mit gleicher Fehlerart/-meldung und identischem erhaltenem tatsächlichem Teilverlauf. Für diese ungelösten Fehlerfälle wird keine vollständige Plan- oder Lösungsparität behauptet. Wiederholungen sind Zeitmessungen derselben Probleme; dies sind keine 100 neuen Probleme je Konfiguration.

Zusätzlich stimmen 29/29 ausgewählte Fall-/Moduspaare mit gespeicherten historischen Rohdaten überein. Dieser Nachweis betrifft den ausgewählten Testausschnitt. Die übrigen vorhandenen Konfigurationen werden gegen den frischen eingefrorenen Alt-Arm geprüft.

### Vergleich vorhandener Konfigurationen

Acht Worker, native Threads jeweils 1, frischer Prozess und persistenter Pool pro Aufruf. Laufzeit: Mittel der Fallmediane des vollständigen Solveraufrufs inklusive synchroner Grenzprotokollierung und Poolstart/-ende. Ungelöste und inkorrekte Ergebnisse bleiben in den Kosten. Numerische Fehler behalten ihre vollständigen Aufrufkosten bis zum Fehler; diese sind keine Lösungszeiten. Vorabgrenze Neu/Alt 1,10 für Gesamtzeit und Medianquotient der Fälle mit Altzeit über 0,1 s. Bei numerischen Fehlern ist das Gesamt-Gate eines Modus nicht verfügbar; ein regulär zurückgekehrter Teilvergleich wird ausdrücklich separat ausgewiesen. Querymittel betreffen Fälle mit regulärer Rückgabe.

| Modus | Fälle × Wiederholungen | Queries/Fall Neu | K/F/U/E Neu | Ø Alt s | Ø Neu s | Neu/Alt gesamt | Medianquotient | Gate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `fixed1_99` | 5 × 1 | 21,00 | 2/0/3/0 | 0,766 | 0,776 | 1,013 | 1,007 | bestanden |
| `fixed2_99` | 8 × 3 | 16,00 | 8/0/0/0 | 10,601 | 10,618 | 1,002 | 1,003 | bestanden |
| `fixed2_95` | 4 × 2 | 7,50 | 3/1/0/0 | 4,504 | 4,479 | 0,995 | 0,997 | bestanden |
| `fixed2_single` | 4 × 2 | 17,75 | 4/0/0/0 | 10,760 | 10,770 | 1,001 | 0,999 | bestanden |
| `fixed2_grid10` | 4 × 2 | 13,00 | 4/0/0/0 | 13,407 | 13,368 | 0,997 | 0,997 | bestanden |
| `fixed2_grid_only` | 4 × 2 | 11,75 | 4/0/0/0 | 7,798 | 7,802 | 1,001 | 0,997 | bestanden |
| `adaptive1_99` | 4 × 1 | 1,25 | 2/0/2/0 | 0,246 | 0,251 | 1,022 | 1,025 | bestanden |
| `adaptive2_99` | 8 × 3 | 16,88 | 8/0/0/0 | 10,181 | 10,168 | 0,999 | 0,996 | bestanden |
| `adaptive2_95` | 4 × 1 | 9,00 | 4/0/0/0 | 5,192 | 5,215 | 1,004 | 1,003 | bestanden |
| `adaptive2_single` | 4 × 1 | 11,75 | 4/0/0/0 | 7,225 | 7,287 | 1,009 | 1,006 | bestanden |
| `adaptive1_stop99` | 4 × 1 | 1,50 | 1/0/3/0 | 0,246 | 0,245 | 0,996 | 0,996 | bestanden |
| `adaptive2_stop99` | 5 × 3 | 6,00 | 5/0/0/0 | 13,847 | 13,928 | 1,006 | 1,006 | bestanden |
| `adaptive2_stop95` | 4 × 1 | 4,25 | 4/0/0/0 | 9,772 | 9,749 | 0,998 | 1,001 | bestanden |
| `adaptive2_stop_single` | 4 × 1 | 10,00 | 4/0/0/0 | 22,824 | 22,704 | 0,995 | 0,994 | bestanden |
| `fixed3_99` | 4 × 1 | 2,00 | 1/0/0/3 | 236,985 | 238,225 | 1,005 | 1,003 | nicht verfügbar: Fehlerfälle |

`fixed3_99`: regulär zurückgekehrter Teilvergleich 1/4 Fälle, Neu/Alt 1,006, Medianquotient 1,006. Die Fehlerfälle werden nicht als erfolgreich gelöst gezählt.

- `three:p001`: vollständiger Aufruf bis zum Fehler Alt 75,596 s / Neu 75,205 s; Teilantworten Alt/Neu [24]/[24].
- `three:p002`: vollständiger Aufruf bis zum Fehler Alt 60,987 s / Neu 60,992 s; Teilantworten Alt/Neu [26]/[26].
- `three:p003`: vollständiger Aufruf bis zum Fehler Alt 70,877 s / Neu 71,507 s; Teilantworten Alt/Neu [24]/[24].

### Neue Konfigurationen

Neue adaptive Tiefe 3, feste Konfidenzplanung und adaptive Grid-only-Planung haben keinen identischen historischen Gegenlauf. Ihre Kosten sind explorativ und nach Zielzahl getrennt; 3D-Ergebnisse werden nicht als Nachweis allgemeiner 5D-Laufzeit verwendet. Queryzahlen betreffen regulär zurückgekehrte Fälle; die Zeitkosten enthalten numerische Fehler, deren Teilantworten im Fehlerabschnitt stehen.

| Modus | Ziele | Fälle | Ø Aufruf s | Queries je Fall mit Rückgabe | Korrekt/falsch/ungelöst/Fehler |
| --- | ---: | ---: | ---: | --- | --- |
| `fixed2_stop99` | 3 | 3 | 2,072 | 4, 10, 6 | 3/0/0/0 |
| `fixed2_stop99` | 5 | 1 | 3,059 | 1 | 1/0/0/0 |
| `fixed3_stop99` | 3 | 3 | 141,544 | 14, 17, 21 | 3/0/0/0 |
| `fixed3_stop99` | 5 | 1 | 833,094 | — | 0/0/0/1 |
| `adaptive3_99` | 3 | 3 | 16,965 | 3, 2, 3 | 3/0/0/0 |
| `adaptive3_99` | 5 | 1 | 166,198 | 1 | 1/0/0/0 |
| `adaptive3_stop99` | 3 | 3 | 62,636 | 4, 2, 3 | 3/0/0/0 |
| `adaptive3_stop99` | 5 | 1 | 486,377 | 1 | 1/0/0/0 |
| `adaptive2_grid_only` | 5 | 3 | 2,264 | 6, 4, 1 | 3/0/0/0 |

### Qualität, Grenzen und Nachweise

Ausführungsfehler: 7; ungelöste Aufrufe: 16; falsche fertige Auswahlen: 4. Diese Zahlen zählen Aufrufe einschließlich Alt/Neu und Wiederholungen. Die eindeutigen Fälle, Gründe, Teilverläufe, Queryzahlen und Regret stehen im JSON und im vollständigen Bericht. Bekannte No-query-/Fragenlimit-Fälle sind ausdrücklich ungelöst; falsche Konfidenzauswahlen sind keine verschwiegenen Ausführungsfehler.

Tests: **305 auf Bench i7, 304 lokale vollständige Tests und 70 lokale fokussierte Tests**, jeweils erfolgreich. Die zwei zusätzlichen realen Geometrieprüfungen für `=` als Antwort und vorhandene Gleichheitsantworten sind ebenfalls erfolgreich; sie sind separate Funktionstests und keine Performancegegenläufe. Controller- und Solveridentitäten sind beendet, Quellen und Harness unverändert, Abschlussaudit bestanden.

Antworten, Gewinner und Regret wurden unabhängig mit rationalen Zahlen geprüft. Volumina bleiben geometrisch berechnete Gleitkommawerte; geprüft wurden ihre Konsistenz und exakte Alt/Neu-Parität, keine symbolisch unabhängige Neuberechnung sämtlicher Polytope.


### Numerische Fehler und tatsächliche Teilantworten

| Modus | Fall | Arm | Wiederholung | Teilantworten | Aufruf bis zum Fehler s | Fehler |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `fixed3_99` | `three:p001` | old | 1 | 24 | 75,596 | RuntimeError: exact query branches have zero total probability |
| `fixed3_99` | `three:p001` | new | 1 | 24 | 75,205 | RuntimeError: exact query branches have zero total probability |
| `fixed3_99` | `three:p002` | new | 1 | 26 | 60,992 | RuntimeError: exact query branches have zero total probability |
| `fixed3_99` | `three:p002` | old | 1 | 26 | 60,987 | RuntimeError: exact query branches have zero total probability |
| `fixed3_99` | `three:p003` | old | 1 | 24 | 70,877 | RuntimeError: exact query branches have zero total probability |
| `fixed3_99` | `three:p003` | new | 1 | 24 | 71,507 | RuntimeError: exact query branches have zero total probability |
| `fixed3_stop99` | `five:p098` | new | 1 | 1 | 833,094 | RuntimeError: linprog failed: (HiGHS Status 0: Not Set) |

[Kompakter Benchmarknachweis](benchmarks/20261014_unified_algorithm_validation_workers8.json); [vollständiger Bericht](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261014-unified-query-algorithm-validation/comparison.md); [unabhängige Prüfung](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261014-unified-query-algorithm-validation/independent-audit.json).
<!-- END UNIFIED_VALIDATION_20261014 -->

<!-- BEGIN UNIFIED_NUMERICAL_RECOVERY_20261014 -->
## Getrennte Validierung der numerischen Reparaturen

Die zweite Bench-i7-Phase umfasst **97 Aufrufe auf sieben zugrunde liegenden Problemen**: 84 gepaarte Kontrollaufrufe, zwölf frische 3D/5A-Lösungsversuche und eine Fortsetzung des gespeicherten 5D/5A-Falls98. Die ursprünglichen 275 Läufe und ihre **sieben Ausführungsfehler** bleiben im vorherigen Abschnitt und im ursprünglichen JSON unverändert dokumentiert.

**Dieser103-Dateien-Stand ist numerisch noch nicht für den breiten Einsatz freigegeben.** Datenintegrität und Kontroll-Laufzeitgates können bestehen, obwohl die numerische Ergebnisprüfung zwei zusätzliche Blocker nachweist. Eine getrennte dritte, gezielte Volumenvalidierung ist erforderlich.

Beide Kontrollarme verwenden die öffentliche API. Fünf Alternativen, Grid5 plus Ratios oberhalb der letzten Ebene, Ratio-only auf der letzten Ebene, exakte Volumina, 99%-Abbruch, acht Worker, native Threads jeweils 1 und höchstens 100 tatsächliche Antworten einschließlich History sind gleich. Wiederholungen sind Zeitmessungen derselben Probleme.

Die geometrische Wiederherstellung greift ausschließlich an zwei vorhandenen Wahrscheinlichkeitsfehlern und den zugehörigen leeren positiven Zweigen. Sie verwendet ursprüngliche binäre Float-Koeffizienten als rationale Zahlen und vollständiges Intervall-/Polygon-Clipping im nichtnegativen Simplex bei affiner Dimension höchstens 2. Ein leerer Zweig wird nur nach einem strikten globalen Singleton-Nachweis korrigiert. Der LP-Fallback greift erst nach ausgeschöpften unklassifizierten numerischen Retries, bei expliziter Nichtnegativität, höchstens 5 Variablen und höchstens 20.000 vollständig zu untersuchenden aktiven Mengen einschließlich Richtungen. Unbewiesene Fälle bleiben ausdrückliche Fehler. Gewöhnliche erfolgreiche Berechnungspfade behalten ihre Implementierung; keine Sampling-, Querymenge- oder Wahrscheinlichkeitsschwellenänderung wurde eingeführt.

**42/42 regulär zurückgekehrte Kontrollpaare** stimmen in allen gespeicherten öffentlichen Ergebnis-, Zustands-, Query- und Planungsfeldern bitgenau überein; zusätzlich 0 identische Fehlerpräfix-Paare. Fehlerpräfixe zählen nicht als vollständige Lösungsparität.

| Kontrolle | Fälle × Wiederholungen | Alt: Summe Fallmediane s | Repariert: Summe Fallmediane s | Neu/Alt | Median Fallquotienten | Gate |
|---|---:|---:|---:|---:|---:|---|
| `fixed2_99` | 7 × 2 |42,517|42,692|1,004|0,997|passed|
| `adaptive2_99` | 7 × 2 |39,111|38,998|0,997|0,998|passed|
| `adaptive2_stop99` | 7 × 2 |159,758|159,631|0,999|0,999|passed|

Die vorab gesetzten Grenzen sind 1,10 für die Gesamtsumme und den Median der Fallquotienten. Altzeiten bis 0,1 s bleiben in der Summe, werden aber beim Quotientenmedian ausgeschlossen. Einzelfälle mit beiden Wiederholungsquotienten über 1,20 werden gesondert untersucht. Fehler oder geänderte Kontrollsemantik erlauben keine allgemeine Nichtregressionsaussage.

| Frischer 3D-Aufruf | Fall | Aufruf s | Queries | Rohstatus / geprüfter Befund | Ende |
|---|---|---:|---|---|---|
|`fixed3_99`|`three:p001`|100,572|[32]|Rohmeldung completed; ungültiger99%-Abbruch|volume_confidence|
|`fixed3_99`|`three:p002`|74,869|[32]|korrekt|volume_confidence|
|`fixed3_99`|`three:p003`|75,759|['Teil: 26']|Fehler|Fehler|
|`fixed3_stop99`|`three:p001`|159,367|[14]|korrekt|exact_single_candidate|
|`fixed3_stop99`|`three:p002`|123,428|[17]|korrekt|volume_confidence|
|`fixed3_stop99`|`three:p003`|141,706|[21]|korrekt|exact_single_candidate|
|`adaptive3_99`|`three:p001`|20,067|[3]|korrekt|exact_single_candidate|
|`adaptive3_99`|`three:p002`|11,618|[2]|korrekt|exact_single_candidate|
|`adaptive3_99`|`three:p003`|19,918|[3]|korrekt|exact_single_candidate|
|`adaptive3_stop99`|`three:p001`|98,312|[4]|korrekt|exact_single_candidate|
|`adaptive3_stop99`|`three:p002`|30,573|[2]|korrekt|exact_single_candidate|
|`adaptive3_stop99`|`three:p003`|59,502|[3]|korrekt|exact_single_candidate|

| 5D-Fortsetzung | Aufruf s (nur Fortsetzung) | Initiale Queries | Neue Queries | Gesamtqueries | Ergebnis |
|---|---:|---:|---|---|---|
|`continuation1:five:p098`|533,423|1|[1]|[2]|korrekt|

Die Fortsetzung beginnt nach der exakt gespeicherten ersten Antwort und misst die verbleibende Berechnung einschließlich erneuter Zustandsanalyse. Die vorherige Querygenerierung ist ausgeschlossen. Es wird weder ein frischer vollständiger 5D-Lauf behauptet noch die frühere Fehlversuchszeit addiert. Auch die reparierten vollständigen 3D-Läufe erhalten keinen Speedup-Faktor gegenüber einer früher abgebrochenen Ausführung.

### Nachgewiesene numerische Blocker

`fixed3_99 / three:p001` meldete nach32 Queries `solved=True` und100% Volumen für Alternative1. Der exakte rationale Nachweis ergibt **77,311457% für Alternative1 und22,688543% für den tatsächlichen Gewinner Alternative3**. Dessen positive Region wurde numerisch als Volumen0 behandelt. Damit war die99%-Voraussetzung nicht erfüllt: Dies ist ein ungültiger Konfidenznachweis, keine reguläre99%-Konfidenzauswahl, die zufällig falsch lag. Sämtliche Antworten bleiben korrekt und der wahre Gewinner bleibt im tatsächlichen Kandidatensatz enthalten.

`fixed3_99 / three:p003` brach nach26 tatsächlichen Antworten mit einem bislang ungeschützten nativen Innenpunkt-/Volumenfehler ab (`row-normalized center LP failed`). Dieser Fall ist ungelöst. Das ursprüngliche Fehlerprotokoll und die separate Diagnose bleiben erhalten; die vollständige Behebung wird erst in der dritten gezielten Phase geprüft.

Die exakten Diagnose- und Rohdatenhashes stehen unter `numerical_readiness.known_issues` im JSON. Diese Befunde werden nicht durch erfolgreiche Kontrollgates oder Unit-Tests aufgehoben.

Diese Phase enthält 1 Ausführungsfehler, 0 regulär ungelöste Aufrufe und 1 im Rohprotokoll fertige falsche Auswahlen. Der bestätigte ungültige99%-Abbruch wird zusätzlich als numerischer Blocker klassifiziert. Die Zahlen zählen Aufrufe einschließlich Wiederholungen. Alle Kosten, Gründe, Teilantworten und Nutzenverluste bleiben im JSON und vollständigen Bericht sichtbar.

- Fehler: `fixed3_99:three:p003:r1:new`, 26 Gesamtqueries, 75,759s; RuntimeError: row-normalized center LP failed: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is Infeasible).
- Falsche Auswahl: `fixed3_99:three:p001:r1:new`, 32 Gesamtqueries, 100,572s; Ende=volume_confidence, Regret=0.2924539917309624.

Mac und Bench i7 bestanden jeweils **337 Multistep- und 6 Onestep-Tests** auf derselben 103-Dateien-Quelle. Abschlussaudit, Quellen-/Harnessprüfung und tatsächliche Controller-/Solver-Endprüfung sind erfolgreich. Diese technischen Nachweise belegen die Phase, nicht numerische Einsatzreife. Antworten, Gewinner und Regret wurden unabhängig mit rationalen binären Eingaben geprüft. Volumina sind intern konsistent geprüft, nicht vollständig mit einem zweiten Solver nachgerechnet. Die drei 3D-Fälle sind kein allgemeiner 5D-Laufzeitnachweis.

[Kompakter Reparaturnachweis](benchmarks/20261014_unified_numerical_recovery_workers8.json); [vollständiger Bericht](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261014-unified-numerical-recovery-validation/comparison.md); [unabhängige Prüfung](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261014-unified-numerical-recovery-validation/independent-audit.json).
<!-- END UNIFIED_NUMERICAL_RECOVERY_20261014 -->

<!-- BEGIN UNIFIED_VOLUME_RECOVERY_20261014 -->

## Zusätzliche Volumenkorrektur:72 geprüfte Aufrufe

Fünf Matrizen:60 Tiefe-2-Kontrollaufrufe und zwölf frische3D/Tiefe-3-Aufrufe. Source104 `7669ad9a421f8eecd651c7b66f766f86c6c4eff504f79e82bb3217a40f637139`. Alle Läufe verwenden Grid5 plus Ratios, letzte Ebene Ratio-only,99%,acht Worker,native Threads1,forkserver,maximal100 Queries und kein Zeitlimit.

Unterstützte native Null-/Degenerations-/Fehlerfälle werden bei explizitem nichtnegativem Simplex und affiner Dimension höchstens2 exakt geprüft. Gewöhnliche positive native Werte bleiben unverändert. Korrigierte Volumina erzwingen eine gemeinsame exakte Parent-/Branch-Partition. Echte Nullfälle und nicht unterstützte Dimensionen behalten ihre bisherigen Pfade; unbewiesene Ausnahmen bleiben Fehler. Der guarded exact-LP-Pfad bleibt bytegleich. Keine Querymengen-, Sampling- oder Wahrscheinlichkeitsschwellenänderung.

Kontrollparität: 30/30 vollständig zurückgekehrte Paare bitgleich; zusätzlich 0 identische Fehlerpräfixe, die nicht als gelöste Probleme zählen. Alle float.hex-Vergleiche und Planungsbeziehungen bleiben im unabhängigen Audit erhalten.

| Kontrolle | Fälle × Wiederholungen | Alt: Fallmedian-Summe s | Neu: Fallmedian-Summe s | Neu/Alt | Median Quotienten | Gate |
|---|---:|---:|---:|---:|---:|---|
| `fixed2_99` |5 ×2|12,793|12,753|0,997|1,007|passed|
| `adaptive2_99` |5 ×2|7,930|7,976|1,006|1,000|passed|
| `adaptive2_stop99` |5 ×2|28,412|28,854|1,016|1,001|passed|

Gates: höchstens1,10 für die aggregierte Median-Summe und den Quotientenmedian; bei zweimal mehr als1,20 wird der Einzelfall untersucht. Fehler, andere Kontrollsemantik oder fehlende Nachweise erlauben keine Nichtregressionsaussage.

| Frische3D-Konfiguration | Fall | Zeit s | Queries | Rohstatus | Zielauswahl | Ende |
|---|---|---:|---:|---|---|---|
|`fixed3_99`|`three:p001`|116,383|32|completed|True|volume_confidence|
|`fixed3_99`|`three:p002`|82,136|32|completed|True|volume_confidence|
|`fixed3_99`|`three:p003`|103,867|33|completed|True|exact_single_candidate|
|`fixed3_stop99`|`three:p001`|159,612|14|completed|True|exact_single_candidate|
|`fixed3_stop99`|`three:p002`|123,658|17|completed|True|volume_confidence|
|`fixed3_stop99`|`three:p003`|142,481|21|completed|True|exact_single_candidate|
|`adaptive3_99`|`three:p001`|19,964|3|completed|True|exact_single_candidate|
|`adaptive3_99`|`three:p002`|11,824|2|completed|True|exact_single_candidate|
|`adaptive3_99`|`three:p003`|19,777|3|completed|True|exact_single_candidate|
|`adaptive3_stop99`|`three:p001`|97,855|4|completed|True|exact_single_candidate|
|`adaptive3_stop99`|`three:p002`|30,493|2|completed|True|exact_single_candidate|
|`adaptive3_stop99`|`three:p003`|58,720|3|completed|True|exact_single_candidate|

Alle 48 abgeschlossenen3D-Auswahlen haben unabhängige Fraction-Nachweise aller ursprünglichen Kandidatengebiete. Binäre native Koeffizienten und die exakte Subtraktion ursprünglicher Matrixwerte werden getrennt geprüft; positive flächige Gleichstände und nullflächige Randkandidaten bleiben sichtbar. Neue ungültige Zertifikate: 0; neue exakte Geometriebefunde: 0.

Gemessene Ausführungsfehler: 0; regulär ungelöst: 0; falsche Zielauswahlen: 0. Die Zahlen zählen Aufrufe einschließlich Wiederholungen. Korrekte Zielauswahl allein genügt nicht als99%-Konfidenznachweis. Kosten und Teilantworten ungelöster/fehlerhafter Läufe bleiben erhalten.

- Erhaltene5D-Fortsetzung `fixed3_stop99_continuation:continuation1:five:p098:r1:new`: 1 initiale + 1 neue = 2 gesamte Queries, 533,423s reine Fortsetzungszeit, korrekt=True. Kein neuer Lauf und kein Nenner für einen Speedup-Faktor.

Geprüfter Umfang: Datenintegrität=True, exakte neue3D-Zertifikate=True (30/30 geprüft), Kontroll-Laufzeitgates=True. Alle neuen Auswahlen korrekt=True. Dies bezieht sich auf die gemessenen fünf Matrizen;5D-Volumina wurden intern, nicht vollständig rational nachgerechnet.

Mac und Bench bestanden jeweils355 Multistep- und sechs Onestep-Tests auf derselben104-Dateien-Quelle. Historische275/97-Rohdaten, sieben primäre Fehler und die zwei97er-Blocker bleiben vollständig erhalten. Die zusätzliche Reparatur ersetzt diese Befunde nicht und liefert keinen künstlichen Speedup gegenüber einem Fehler oder ungültigen frühen Stopp.

[Kompakter Nachweis](benchmarks/20261014_unified_volume_recovery_workers8.json); [vollständiger Bericht](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261014-unified-volume-recovery-validation/comparison.md); [unabhängige Prüfung](/Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261014-unified-volume-recovery-validation/independent-audit.json).
<!-- END UNIFIED_VOLUME_RECOVERY_20261014 -->
