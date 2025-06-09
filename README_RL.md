# 🤖 Reinforcement Learning für optimale Dienstplanung

Dieses Projekt implementiert ein **Single-Agent Reinforcement Learning** System zur optimalen Dienstplanung im Call-Center-Umfeld. Der RL-Agent lernt, Mitarbeiter optimal zu Zeitslots zuzuweisen und dabei alle relevanten Constraints zu berücksichtigen.

## 🎯 Projektziele

- **Service Level erreichen**: 80/20 Service Level (80% der Anrufe innerhalb 20 Sekunden)
- **Constraints einhalten**: Arbeitszeiten, Pausen, Ruhezeiten
- **Kosten optimieren**: Minimale Überbesetzung bei maximaler Effizienz
- **Flexibilität**: Automatische Anpassung an Forecast-Änderungen

## 🏗️ Projektstruktur

```
├── data_csv/                    # CSV-Daten für Training
│   ├── agents.csv              # Mitarbeiter-Daten
│   ├── lines.csv               # Line-Informationen (AHT)
│   ├── forecast.csv            # Kontakt-Forecast
│   └── constraints.csv         # Constraint-Definitionen
├── src/
│   ├── environment/
│   │   └── shift_scheduling_env.py    # Gymnasium Environment
│   ├── models/
│   │   └── train_agent.py             # Training mit Stable-Baselines3
│   ├── utils/
│   │   └── data_loader.py             # CSV-Datenverarbeitung
│   └── evaluation/                    # Evaluation und Metriken
├── requirements.txt            # Python-Dependencies
├── test_environment.py        # Environment-Tests
└── train_shift_scheduler.py   # Hauptskript für Training
```

## 🚀 Quick Start

### 1. Installation

```bash
# Python Virtual Environment erstellen
python -m venv rl_scheduling_env
source rl_scheduling_env/bin/activate  # Windows: rl_scheduling_env\Scripts\activate

# Dependencies installieren
pip install -r requirements.txt
```

### 2. Environment testen

```bash
# Basis-Tests ohne RL-Training
python train_shift_scheduler.py --mode test
```

### 3. RL-Training starten

```bash
# Schnelles Training (50k steps)
python train_shift_scheduler.py --mode train --timesteps 50000

# Vollständiges Training (200k steps)
python train_shift_scheduler.py --mode train --timesteps 200000 --max_steps 500
```

### 4. Trainiertes Model evaluieren

```bash
python train_shift_scheduler.py --mode eval --model_path results/models/best_model
```

## 📊 Datenstruktur

### Agents (agents.csv)
```csv
csvId,DisplayName,Wochenstunden,Quali_Line1,Quali_Line2,Quali_Line3
A001,Mueller_Max,40.0,True,False,False
```

### Forecast (forecast.csv)
```csv
csvDatum,Interval,Erwartete_Kontakte
2025-06-09,07:00-07:30,12
```

### Constraints (constraints.csv)
```csv
csvID,Titel,Schweregrad,IsPositiv,Gewicht
MIN_REST_TIME,Mindest-Ruhezeit zwischen zwei Schichten,Muss,-1,100
```

## 🤖 RL-Environment Details

### State Space
- **Schedule Matrix**: Aktuelle Zuweisungen (10 Agenten × 210 Zeitslots)
- **Forecast**: Erwartete Kontakte pro Zeitslot
- **Agent Verfügbarkeit**: Wochenstunden und Qualifikationen
- **Constraint Status**: Aktuelle Violations und Compliance

### Action Space
- **Discrete Actions**: `Agent_Index * Time_Slots + Time_Slot`
- **Toggle-Mechanismus**: Zuweisen (0→1) oder Entfernen (1→0)
- **Action Space Size**: 2,100 (10 Agenten × 210 Zeitslots)

### Reward Function
```python
reward = service_level_reward + efficiency_bonus - constraint_penalties + hours_compliance
```

**Komponenten:**
- **Service Level**: +2.0 wenn ≥80%, -1.0 × (0.8 - level) wenn <80%
- **Constraint Penalties**: -0.5 pro Violation
- **Efficiency**: +0.5 bei optimaler Agenten-Zuordnung
- **Hours Compliance**: +0.3 bei korrekter Wochenstunden-Einhaltung

## 📈 Training-Algorithmus

**PPO (Proximal Policy Optimization)**
- **Policy**: Multi-Layer Perceptron (MLP)
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Gamma**: 0.99
- **Training Steps**: 50k - 200k

### Training-Parameter

```python
training_params = {
    'total_timesteps': 100000,
    'max_steps': 300,
    'learning_rate': 3e-4,
    'batch_size': 64,
    'gamma': 0.99
}
```

## 🎯 Evaluation-Metriken

### Primäre Metriken
- **Service Level**: % Zeitslots mit ausreichend Agenten
- **Constraint Violations**: Anzahl verletzter Constraints
- **Hours Compliance**: Einhaltung der Wochenstunden
- **Schedule Completeness**: Vollständigkeit der Planung

### Constraint-Typen
1. **Muss-Constraints** (Gewicht: 75-100)
   - Mindest-/Maximal-Schichtlänge
   - Ruhezeiten zwischen Schichten
   - Obligatorische Pausen

2. **Soll-Constraints** (Gewicht: 50-70)
   - Service Level 80/20
   - Wochenstunden-Einhaltung
   - Wochenend-Abdeckung

3. **Kann-Constraints** (Gewicht: 30-40)
   - Kostenoptimierung
   - Gleichmäßige Verteilung

## 📊 Ergebnisse

Das System wird in `results/` gespeichert:

```
results/
├── models/
│   ├── best_model.zip          # Bestes Model während Training
│   └── final_model.zip         # Finales trainiertes Model
├── logs/
│   ├── training_log.csv        # Training-Verlauf
│   └── tensorboard/            # TensorBoard-Logs
├── plots/
│   └── training_results.png    # Visualisierungen
├── evaluation_results.csv      # Evaluation-Statistiken
└── final_schedule.csv          # Optimaler Schedule
```

## 🛠️ Erweiterte Nutzung

### Custom Training

```python
from src.models.train_agent import ShiftSchedulingTrainer

trainer = ShiftSchedulingTrainer(data_path="data_csv")
trainer.train_model(
    total_timesteps=100000,
    learning_rate=1e-3,
    batch_size=128
)
```

### Environment Anpassungen

```python
from src.environment.shift_scheduling_env import ShiftSchedulingEnv

env = ShiftSchedulingEnv(
    data_path="custom_data/",
    max_steps=500
)
```

## 🔧 Troubleshooting

### Häufige Probleme

1. **Import-Fehler**
   ```bash
   pip install -r requirements.txt
   ```

2. **CSV-Dateien fehlen**
   - Prüfen Sie den `data_csv/` Ordner
   - Stellen Sie sicher, dass alle 4 CSV-Dateien vorhanden sind

3. **GPU/CPU Performance**
   ```python
   # Für CPU-only Training
   import torch
   torch.device('cpu')
   ```

4. **Memory Issues**
   - Reduzieren Sie `max_steps` Parameter
   - Verwenden Sie kleinere `batch_size`

### Debug-Modus

```bash
# Ausführliche Logs
python train_shift_scheduler.py --mode train --timesteps 10000 --verbose 2

# Environment-Details
python test_environment.py
```

## 📚 Technische Details

### Dependencies
- **stable-baselines3**: RL-Algorithmen
- **gymnasium**: Environment-Framework
- **pandas/numpy**: Datenverarbeitung
- **matplotlib/seaborn**: Visualisierung
- **torch**: Deep Learning Backend

### Performance
- **Training Zeit**: 30-60 Min (50k steps)
- **Memory Usage**: ~1-2 GB RAM
- **GPU**: Optional, aber empfohlen für >100k steps

## 🔮 Roadmap

### Geplante Features
- [ ] **Multi-Agent RL**: Jeder Agent als eigenständiger RL-Agent
- [ ] **Hierarchical RL**: Schicht-Level und Agent-Level Entscheidungen
- [ ] **Transfer Learning**: Pre-trained Models für neue Szenarien
- [ ] **Real-time Adaptation**: Online-Learning bei Forecast-Änderungen
- [ ] **Advanced Constraints**: Qualifikations-Matching, Teamwork
- [ ] **Web Interface**: Dashboard für Schedule-Visualisierung

### Algorithmus-Verbesserungen
- [ ] **SAC**: Soft Actor-Critic für bessere Exploration
- [ ] **Rainbow DQN**: Erweiterte DQN-Varianten
- [ ] **Meta-Learning**: Schnelle Anpassung an neue Szenarien

## 🤝 Contributing

1. Fork das Repository
2. Erstellen Sie einen Feature-Branch
3. Implementieren Sie Ihre Änderungen
4. Fügen Sie Tests hinzu
5. Erstellen Sie einen Pull Request

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe `LICENSE` für Details.

## 📞 Support

Bei Fragen oder Problemen:
- Erstellen Sie ein Issue im Repository
- Überprüfen Sie die Troubleshooting-Sektion
- Kontaktieren Sie das Entwicklerteam

---

**Hinweis**: Dieses Projekt dient als Proof-of-Concept für RL-basierte Dienstplanung. Für Produktionseinsatz sind zusätzliche Validierungen und Tests erforderlich.