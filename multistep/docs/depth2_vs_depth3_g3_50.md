# Tiefe 2 gegen Tiefe 3: 50 gepaarte 3D-Probleme

## Versuchsaufbau

- 3 Ziele, 10 Alternativen, 50 unabhängig erzeugte Probleme
- identische Alternativen und Zielgewichte für beide Tiefen
- Seed `20260902`
- exakte Volumenwahrscheinlichkeiten
- geometrische Ratio-Intervall-Engine
- eine Schwerpunkt-Query je kanonischem Zielpaar
- lexikographische Auswahl nach `(E_d, ..., E_1)`

## Ergebnis

| Kennzahl | Tiefe 2 | Tiefe 3 |
|---|---:|---:|
| Gelöst | 50/50 | 50/50 |
| Queries gesamt | 224 | 254 |
| Queries, Mittelwert | 4,48 | 5,08 |
| Queries, Median | 4 | 4 |
| Queries, Standardabweichung | 2,62 | 2,88 |
| Queries, Minimum / Maximum | 0 / 13 | 0 / 13 |
| Laufzeit gesamt | 28,714 s | 143,822 s |
| Laufzeit, Mittelwert | 0,574 s | 2,876 s |
| Laufzeit, Median | 0,520 s | 2,477 s |

Die gepaarte Differenz `Tiefe 3 - Tiefe 2` beträgt im Mittel `+0,60`
Queries. Das 95-%-Konfidenzintervall des gepaarten Mittelwerts beträgt
`[+0,077; +1,123]`. Tiefe 3 war in 10 Problemen besser, in 18 gleich und in
22 schlechter. Der Wilcoxon-Vorzeichen-Rang-Test ergibt `p = 0,0144`; der
reine Vorzeichentest über die 32 ungleichen Paare ergibt `p = 0,0501`.

Die Tiefe-3-Läufe benötigten insgesamt rund das `5,01`-Fache der Laufzeit.

## Interpretation

Der lexikographische Schlüssel behebt die Abweichung zwischen Ausarbeitung
und Implementierung, erzwingt aber keine monotone Abnahme der real benötigten
Queries. Er verwendet `E_{d-1}` nur, wenn zwei Queries in `E_d` exakt gleich
sind. Hat eine Query einen echt kleineren `E_3`-Wert, wird sie auch dann
gewählt, wenn die von Tiefe 2 gewählte Query für das konkrete Zielgewicht
schneller terminieren würde.

Die aktuelle Wertfunktion minimiert die erwartete Zahl verbleibender
Kandidaten nach einem festen Horizont. Sie minimiert nicht die erwartete Zahl
der Queries bis zur eindeutigen Identifikation. Für dieses Ziel wäre eine
Stoppzeit-Wertfunktion beziehungsweise ein explizites Terminierungsziel
erforderlich.

## Rohdaten

- `multistep/data/exact_depth2_g3_50_lexicographic.json`
- `multistep/data/exact_depth3_g3_50_lexicographic.json`
