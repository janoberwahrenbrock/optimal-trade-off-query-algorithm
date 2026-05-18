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
