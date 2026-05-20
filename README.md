# Optimal Trade-Off Query Algorithm

Dieses Repository enthaelt den Algorithmus und eine Streamlit-App zur Analyse von Trade-off-Queries.

## Voraussetzungen

- Python 3.11 oder neuer
- `pip`

## Projekt einrichten

### macOS

Virtuelle Umgebung anlegen:

```bash
python3 -m venv .venv
```

Virtuelle Umgebung aktivieren:

```bash
source .venv/bin/activate
```

Abhaengigkeiten installieren:

```bash
pip install -r requirements.txt
```

### Windows (PowerShell)

Virtuelle Umgebung anlegen:

```powershell
py -m venv .venv
```

Virtuelle Umgebung aktivieren:

```powershell
.venv\Scripts\Activate.ps1
```

Abhaengigkeiten installieren:

```powershell
pip install -r requirements.txt
```

Falls PowerShell das Aktivieren blockiert, kann die Ausfuehrungsrichtlinie fuer die aktuelle Sitzung gelockert werden:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows (cmd)

Virtuelle Umgebung anlegen:

```bat
py -m venv .venv
```

Virtuelle Umgebung aktivieren:

```bat
.venv\Scripts\activate.bat
```

Abhaengigkeiten installieren:

```bat
pip install -r requirements.txt
```

## Apps starten

Nach dem Aktivieren der virtuellen Umgebung kann die 3-Ziele-App so gestartet werden:

```bash
streamlit run onestep/scripts/3_goals.py
```

Die 4-Ziele-App mit Plotly-Tetraeder kann so gestartet werden:

```bash
streamlit run onestep/scripts/4_goals.py
```

Danach oeffnet sich die App im Browser. Falls sie nicht automatisch erscheint, zeigt Streamlit im Terminal eine lokale URL an.

## Beispiel mit JSON-Datei

Die App kann mit den vorhandenen Beispieldaten gestartet werden:

```bash
streamlit run onestep/scripts/3_goals.py -- --data-file data/a5_a10_case.json --load-tradeoffs
```

## Performance-Analyse starten

```powershell
python onestep/scripts/analyse_performance_cli.py --goals 3,5,7 --alternatives 3,6,9 -x 10 --seed 1 --max-calls 50
```

## Projektstruktur

- `onestep/src/`: Kernlogik des bisherigen One-Step-Algorithmus
- `onestep/scripts/`: Apps und Analyse-Skripte fuer den One-Step-Ansatz
- `onestep/data/`: Beispieldaten fuer den One-Step-Ansatz
- `onestep/docs/`: Dokumentation des One-Step-Algorithmus
- `onestep/tests/`: Tests fuer den One-Step-Ansatz
- `multistep/src/`: Neue Codebasis fuer den Multi-Step-Ansatz
- `multistep/scripts/`: Skripte fuer den Multi-Step-Ansatz
- `multistep/data/`: Daten fuer den Multi-Step-Ansatz
- `multistep/docs/`: Dokumentation des Multi-Step-Ansatzes
- `multistep/tests/`: Tests fuer den Multi-Step-Ansatz

## Multi-Step Analyse-Skripte

### Terminierungsanalyse

Das Skript `multistep/scripts/analyze_termination_question_counts.py` erzeugt
zufaellige Probleme, simuliert einen Nutzer ueber einen Zielgewichtsvektor und
wendet das optimierte Multi-Step-Verfahren so lange an, bis nur noch ein
Kandidat uebrig ist oder `--max-questions` erreicht wird.

Beispiel:

```bash
python multistep/scripts/analyze_termination_question_counts.py \
  --goals 5 \
  --alternatives 10 \
  --problems 10 \
  --samples 400 \
  --grid-size 10 \
  --max-s 100 \
  --print-problems \
  --export-json multistep/data/termination_runs/goals5_alts10_seed1.json
```

Wichtige Optionen:

- `--goals`: Anzahl der Ziele.
- `--alternatives`: Anzahl der Handlungsalternativen.
- `--problems`: Anzahl der zufaellig erzeugten Probleme.
- `--start-problem`: Erstes Problem, das ausgefuehrt werden soll. Nuetzlich, um
  gezielt z.B. nur Problem 10 zu reproduzieren.
- `--depth`: Lookahead-Tiefe, standardmaessig `2`.
- `--samples`: Anzahl der Samples fuer Antwortwahrscheinlichkeiten.
- `--grid-size`: Anzahl der Grid-Werte pro Zielpaar.
- `--max-s`: Maximales Query-Verhaeltnis im Grid.
- `--root-query-source`: Query-Quellen fuer Ebenen groesser als 1. Standard ist
  `both`, also Grid-Queries plus Ratio-Queries.
- `--print-problems`: Gibt pro Problem eine Ergebniszeile aus.
- `--export-json`: Speichert Alternativenmatrix, simulierten
  Zielgewichtsvektor und gestellte Queries als JSON.
- `--validate-terminal-counts`: Debug-Modus. Validiert terminale
  Ratio-Shortcut-Counts gegen eine exakte Child-Kandidatenberechnung und ist
  deshalb langsamer.

Aktuelles Standardverhalten:

- Auf der untersten Ebene (`depth = 1`) werden Ratio-generierte Queries
  verwendet.
- Auf Ebenen groesser als 1 werden standardmaessig Grid-Queries und
  Ratio-Queries gemeinsam betrachtet.

### Exportierte Query-Laeufe pruefen

Das Skript `multistep/scripts/analyze_exported_query_run.py` laedt eine durch
die Terminierungsanalyse erzeugte JSON-Datei, beantwortet die gespeicherten
Queries mit dem gespeicherten Zielgewichtsvektor und berechnet danach mit
`multistep/src` die eingeschraenkten Zielgewichtsraeume und Kandidatenmengen
neu.

Beispiel:

```bash
python multistep/scripts/analyze_exported_query_run.py \
  --input multistep/data/termination_runs/goals5_alts10_seed1.json \
  --print-steps
```

Optional kann das Ergebnis wieder als JSON geschrieben werden:

```bash
python multistep/scripts/analyze_exported_query_run.py \
  --input multistep/data/termination_runs/goals5_alts10_seed1.json \
  --output-json multistep/data/termination_runs/goals5_alts10_seed1_analysis.json \
  --print-steps
```

Wichtige Optionen:

- `--input`: Export-Datei aus `analyze_termination_question_counts.py`.
- `--print-steps`: Gibt nach jeder beantworteten Query die verbleibenden
  Kandidaten aus.
- `--output-json`: Speichert die Replay-Analyse als JSON.
