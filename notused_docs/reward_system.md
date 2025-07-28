# Reward-System für Wochen-basierte Schichtplanung

Dieses Dokument beschreibt das Reward-System für das Reinforcement Learning-Modell zur Schichtplanung über eine ganze Woche.

## Übersicht

Das Reward-System bewertet eine komplette Woche (7 Tage) und verwendet exponentielle Strafen/Belohnungen für optimale Zielerfüllung.

## Harte Constraints (Strafen)

### 1. Bedarfsdeckung pro Stunde (-500 pro fehlender MA-Stunde)
- **Ziel:** Sicherstellen, dass der Personalbedarf pro Stunde gedeckt ist
- **Strafe:** -500 Punkte pro fehlender MA-Stunde
- **Belohnung:** +200 Punkte für exakte Bedarfsdeckung pro Stunde
- **Überbesetzung:** -100 Punkte pro überzähligem MA (bei >30% über Bedarf)

### 2. Öffnungszeiten (-300 pro MA außerhalb 7-22 Uhr)
- **Ziel:** Personal nur während der Geschäftszeiten einteilen
- **Strafe:** -300 Punkte pro MA-Stunde außerhalb 7-22 Uhr

### 3. Schichtlängen (-200 für unzulässige Längen)
- **Ziel:** Schichten zwischen 3-8 Stunden
- **Strafe:** -200 Punkte für Schichten <3 oder >8 Stunden
- **Belohnung:** +30 Punkte für optimale Schichtlänge (6 Stunden)

## Exponentielle Rewards (Neue Struktur)

### 4. Vertragsstunden: Exponentielle Strafe/Belohnung
- **Ziel:** Exakte Einhaltung der Wochenstunden pro Mitarbeiter
- **Strafe:** -100 × (Abweichung)² (quadratische Strafe)
- **Belohnung:** +500 × exp(-0.2 × Abweichung) (exponentielle Belohnung)
- **Beispiel:** MA mit 40h Vertrag arbeitet 35h → Strafe: -100 × 5² = -2500, Belohnung: +500 × exp(-1) ≈ +184

### 5. Forecast-Abweichung pro Tag: Exponentielle Strafe/Belohnung
- **Ziel:** Exakte Bedarfsdeckung pro Tag
- **Strafe:** -200 × (Abweichung)² (quadratische Strafe)
- **Belohnung:** +300 × exp(-0.2 × Abweichung) (exponentielle Belohnung)
- **Beispiel:** Tag mit 44h Bedarf, 50h Besetzung → Strafe: -200 × 6² = -7200, Belohnung: +300 × exp(-1.2) ≈ +90

### 6. Wochenbedarf: Stufenweise Belohnung
- **Ziel:** Gesamtstunden nahe am Wochenbedarf
- **Belohnung:** +2000 Punkte wenn ≥90% des Wochenbedarfs
- **Belohnung:** +1000 Punkte wenn ≥80% des Wochenbedarfs
- **Belohnung:** +500 Punkte wenn ≥60% des Wochenbedarfs
- **Strafe:** -5000 Punkte wenn <30% des Wochenbedarfs

## Weiche Constraints

### 7. Arbeitsanreiz (+5 pro gearbeiteter Stunde)
- **Ziel:** Verhindern, dass das Modell "faul" plant
- **Belohnung:** +5 Punkte pro gearbeiteter Stunde

### 8. Zeitliche Verteilung über die Woche
- **Ziel:** Gleichmäßige Arbeitslast über alle Tage
- **Belohnung:** +200 Punkte wenn Varianz >2 (bessere Verteilung)
- **Strafe:** -400 Punkte wenn Varianz ≤2 (zu gleichmäßig)

### 9. Perfekte Bedarfsdeckung
- **Ziel:** Alle Bedarfe der Woche gedeckt
- **Belohnung:** +1000 Punkte wenn alle Bedarfe gedeckt sind

## Beispiel-Berechnung

Für eine Woche mit 260 Stunden Gesamtbedarf und perfekter Einhaltung der Vertragsstunden:

**Vertragsstunden (10 MA):**
- MA1-2 (40h): 0 Abweichung → +500 Belohnung pro MA
- MA3-6 (30h): 0 Abweichung → +500 Belohnung pro MA  
- MA7-8 (20h): 0 Abweichung → +500 Belohnung pro MA
- MA9-10 (10h): 0 Abweichung → +500 Belohnung pro MA
- **Summe:** +5000 Punkte

**Forecast-Abweichung (7 Tage):**
- Perfekte Deckung → +300 × exp(0) = +300 pro Tag
- **Summe:** +2100 Punkte

**Wochenbedarf:** +2000 Punkte (≥90%)
**Arbeitsanreiz:** +1300 Punkte (260h × 5)
**Verteilung:** +200 Punkte
**Perfekte Deckung:** +1000 Punkte

**Gesamt:** +11.600 Punkte

## Aktuelle Ergebnisse

### Erfolge:
- **Bedarfsdeckung:** 92-140% pro Tag (sehr gut)
- **Gesamtstunden:** 276 (nahe am Ziel von 260)
- **Vertragsstunden:** Durchschnitt 27,6h (Verbesserung gegenüber vorherigen 40h+)

### Verbesserungspotential:
- **Vertragsstunden:** Noch nicht perfekt verteilt (Ziel: 10-40h je nach Vertrag)
- **Training:** Längeres Training für bessere Konvergenz

## Ziele für das Training

1. **Bedarfsdeckung:** 95-105% pro Tag
2. **Vertragsstunden:** Exakte Einhaltung (±5% Toleranz)
3. **Schichtlängen:** Alle Schichten zwischen 3-8 Stunden
4. **Öffnungszeiten:** Personal nur 7-22 Uhr
5. **Verteilung:** Gleichmäßige Arbeitslast über die Woche
6. **Gesamtstunden:** 260 Stunden (entsprechend Forecast) 