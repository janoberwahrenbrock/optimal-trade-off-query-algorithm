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

## App starten

Nach dem Aktivieren der virtuellen Umgebung kann die Streamlit-App so gestartet werden:

```bash
streamlit run scripts/3_goals.py
```

Danach oeffnet sich die App im Browser. Falls sie nicht automatisch erscheint, zeigt Streamlit im Terminal eine lokale URL an.

## Beispiel mit JSON-Datei

Die App kann mit den vorhandenen Beispieldaten gestartet werden:

```bash
streamlit run scripts/3_goals.py -- --data-file data/a5_a10_case.json --load-tradeoffs
```

## Projektstruktur

- `src/`: Kernlogik des Algorithmus
- `scripts/3_goals.py`: Streamlit-App
- `data/`: Beispieldaten
