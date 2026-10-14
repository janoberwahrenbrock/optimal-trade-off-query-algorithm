# E19: Gemeinsame Vorbereitung adaptiver Horizonte 2 und 3

Ergebnis am 14. Oktober 2026: **Prototyp verworfen; kein verlässlicher Laufzeitgewinn.** Der bisher beibehaltene E01+E13-Algorithmus bleibt unverändert. Während dieses Experiments wurde nichts committed oder gepusht. Der anschließende Sicherungscommit enthält den beibehaltenen E01+E13-Stand und diese negative Dokumentation; der verworfene E19-Prototyp wird nicht Teil des Algorithmus.

## Umsetzung

Die Auswertung für Tiefe 2 sammelt ihre bereits berechneten Antwortwahrscheinlichkeiten und unmittelbaren Kindzustände mit Gewichtsregion, Kandidatenfilter und vollständiger Kandidatenanalyse. Nach Abschluss sämtlicher Tiefe-2-Auswertungen kann Tiefe 3 diese Vorbereitung für dieselbe Query übernehmen. Alle Fortsetzungspläne, tiefere Teilbäume, Frontiers und Tie-Breaks werden weiterhin separat berechnet.

Exakte Geometrie, LPs, Volumina, Querymengen, Antwortzweige, Erwartungswerte und Summationsreihenfolge bleiben erhalten. Ungeeignete oder außergewöhnliche Kontexte verwenden den bisherigen Pfad. Es gibt keinen globalen Cache und keine neue öffentliche Konfiguration.

Der Prototyp änderte vier Python-Dateien und blieb in einem separaten eingefrorenen Verzeichnis. Native Geometrie-, Volumen-, Wahrscheinlichkeits- und LP-Backends sowie die öffentliche Solver-API blieben bytegleich.

## Prüfungen und Konfiguration

- 398 bestandene Tests auf jedem Host: 392 Multistep und 6 Onestep; darin 14 gezielte neue Tests.
- Separate Diagnose mit acht ausgewählten Root-Queries je Modus: 32 vollständige H2/H3-Teilplan-Paare über beide Modi exakt gleich. Je Modus wurden acht Wahrscheinlichkeitsberechnungen und 14 unmittelbare Kindanalysen wiederverwendet.
- Hauptvergleich: zehn erfolgreiche, seriell ausgeführte Calls, fünf mathematisch exakt gleiche Alt/Neu-Paare; keine Fehler oder Timeouts. Alle gespeicherten mathematischen Floatwerte wurden über float.hex verglichen.
- Gleiche acht Worker, exakte Volumenberechnung, Grid 5 plus Ratios oberhalb der letzten Ebene und nur Ratios auf der letzten Ebene; maximal adaptive Tiefe 3, tatsächlicher Konfidenzabbruch bei 99 %.

Für die 5D-Messung wurden fünf Ziele und fünf Alternativen nach denselben acht bereits beantworteten Queries verwendet. Jeder Call berechnet die vollständigen Horizonte mit 12/100/100 Root-Queries. Es werden dabei null neue tatsächliche Queries gestellt. Gemessen wird planning_seconds, die Zeit eines vollständigen Planungsschritts. Das ist keine Laufzeit bis zur Lösung eines gesamten 5D-Problems.

## Unprofilierte Messungen auf Bench i7

Pro Modus zwei gekreuzte Paare in der Reihenfolge alt, neu, neu, alt. Positive Änderungen bedeuten längere Laufzeit.

| Modus | Alte Zeiten (s) | Neue Zeiten (s) | Alter Median (s) | Neuer Median (s) | Änderung Median |
|---|---:|---:|---:|---:|---:|
| Adaptive Tiefe 3, Konfidenz nur beim Abbruch | 86.383757, 90.667087 | 89.838438, 89.350687 | 88.525422 | 89.594563 | +1.2077 % |
| Adaptive Tiefe 3, Konfidenz auch in der Planung | 247.377016, 245.627482 | 248.787652, 247.253786 | 246.502249 | 248.020719 | +0.6160 % |

Änderungen der jeweiligen Paare:
- Adaptive Tiefe 3, Konfidenz nur beim Abbruch: +3.9992 %, -1.4519 %.
- Adaptive Tiefe 3, Konfidenz auch in der Planung: +0.5702 %, +0.6621 %.

Zusätzlich wurde ein frisches vollständiges 3D-Problem mit fünf Alternativen in beiden Varianten gelöst:

| Variante | Vollständige Laufzeit (s) | Tatsächliche Queries | Auswahl | Stopp |
|---|---:|---:|---|---|
| Bisheriger Stand | 28.878028 | 2 | Kandidat 3 (Index ab 0), korrekt | Genau ein Kandidat, Volumenanteil 100 % |
| Prototyp | 28.599621 | 2 | Kandidat 3 (Index ab 0), korrekt | Genau ein Kandidat, Volumenanteil 100 % |

Auch vollständige Query-/Antwortfolgen, Kandidatenzustände, Volumina und Auswahl dieses Kontrollfalls sind exakt gleich. Der kleine Unterschied von −0,9641 % in diesem einmaligen Kontrollpaar ist kein verlässlich belegter Performancegewinn.

## Entscheidung und Erklärung

Vorab wurde verlangt: mindestens 2 % Gewinn in beiden gekreuzten Paaren und im Median mindestens eines Zielmodus; höchstens 2 % Medianregression in beiden Zielmodi und höchstens 10 % Regression im Kontrollfall. Vollständige Mathematikparität und erfolgreiche abgeschlossene Calls waren zusätzliche Voraussetzungen. Die Integritäts- und Regressionsprüfungen bestehen, aber kein Zielmodus erreicht den Gewinn. Daher keine Übernahme.

Die Diagnose zeigt nur einen kleinen Anteil unmittelbarer Root-Vorbereitung gegenüber den tieferen Berechnungen. Das Wiederverwenden spart reale Aufrufe, doch der zusätzliche Verwaltungs- und Datentransferaufwand kann den kleinen Nutzen aufzehren. Der Vergleich belegt keinen Zeitgewinn; er isoliert diese Overheadkosten nicht einzeln.

Grenzen: ein gespeicherter 5D-Zustand, zwei Wiederholungen je Modus und ein vollständiger 3D-Kontrollfall. Wiederholungen sind keine unabhängigen Probleme. Keine Aussage über eine 100-Probleme-Verteilung oder eine universelle Ergebnis-/Laufzeitgarantie. Profilzeiten und Controllerzeiten wurden nicht als Laufzeitgewinn verwendet.

## Quellen, Fortsetzung und Wiederherstellung

Experimentverzeichnis auf dem Mac: /Users/janoberwahrenbrock/Code/optimal-trade-off-query-algorithm/outputs/benchmarks/bench-i7/20261014-shared-adaptive-root-preparation

Bench: /home/cruncher/otqa-benchmarks/20261014-shared-adaptive-root-preparation

Maßgeblich sind comparison-plan-01.json, validation/acceptance.json, comparison-final-audit.json, performance-decision.json und die unabhängigen Abschlussprüfungen in validation/. Die einzelnen Resultate liegen unter runs/comparison-plan-01/. Diagnose, Quellen, Patches und alle Versuche bleiben erhalten.

- Beibehaltener Algorithmus: 107 Python-Dateien, Fingerprint 7dcdd8251c8a80fbc2585781ea34a750042d09fe59c3bd21cab352ffcdd44fd7.
- Verworfener Prototyp: 109 Python-Dateien, Fingerprint b33875742bc1b475343ae180a7f211f1fcd3a6fad295ef2f3ecd8f207bf45fe5.
- Ausgangscommit: c6eea313bc24454453b5a32eb19e7e8bfab3c307; lokale Branch codex/unified-targeted-performance.
- Prototyp zur Untersuchung: candidate-repo/, candidate-source-manifest.json und candidate-diff.patch. Nicht in den verwendeten Algorithmus integriert.
- Der abschließende current-local-checkpoint-e19-final sichert sämtliche zum Experimentabschluss noch uncommitteten Dateien einschließlich dieser Dokumentation und prüft ihre tatsächliche Wiederherstellung aus dem Ausgangscommit.
