import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """
    Einfaches neuronales Netz für DQN.
    Besteht aus drei vollverbundenen Schichten.
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        # Definiere die Schichten des Netzes
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),   # Erste Schicht: input_dim -> 128
            nn.ReLU(),                   # Aktivierungsfunktion
            nn.Linear(128, 128),         # Zweite Schicht: 128 -> 128
            nn.ReLU(),                   # Aktivierungsfunktion
            nn.Linear(128, output_dim)   # Ausgabeschicht: 128 -> output_dim
        )
        
    def forward(self, x):
        """Führt Vorwärtspropagation durch."""
        return self.layers(x)

class DQNAgent:
    """
    Ein vereinfachter DQN-Agent für Schichtplanung.
    
    Verwendet:
    - Experience Replay (Speichern und Wiederholen von Erfahrungen)
    - Epsilon-Greedy Exploration (Balance zwischen Erkunden und Ausnutzen)
    - Target Network (Stabilität beim Lernen)
    """
    
    def __init__(self, state_dim, num_employees, learning_rate=0.0005):  # Reduzierte Lernrate
        # Dimensionen für Zustände und Aktionen
        self.state_dim = state_dim
        self.num_employees = num_employees
        self.action_dim = 24 * 6  # Mögliche Aktionen pro MA
        
        # Hyperparameter
        self.gamma = 0.95          # Discount-Faktor für zukünftige Belohnungen
        self.epsilon = 1.0         # Startwahrscheinlichkeit für zufällige Aktionen
        self.epsilon_min = 0.05    # Minimale Epsilon-Wahrscheinlichkeit (erhöht)
        self.epsilon_decay = 0.998 # Langsamerer Epsilon-Decay
        self.learning_rate = learning_rate
        
        # Speicher für Erfahrungen
        self.memory = deque(maxlen=5000)  # Größerer Replay-Buffer
        
        # Gerät (CPU/GPU) für PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Erstelle Haupt- und Ziel-Netzwerke
        self.model = DQN(state_dim, self.action_dim).to(self.device)
        self.target_model = DQN(state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizer und Verlustfunktion
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        """Speichert eine Erfahrung im Replay-Speicher."""
        # Konvertiere numpy arrays zu Listen für einfachere Speicherung
        state = state.tolist()
        next_state = next_state.tolist()
        self.memory.append((state, int(action), float(reward), next_state, bool(done)))
    
    def act(self, state):
        """
        Wählt Aktionen für alle Mitarbeiter basierend auf dem aktuellen Zustand.
        Verwendet Epsilon-Greedy für die Exploration.
        """
        actions = np.zeros(self.num_employees, dtype=np.int32)
        
        # Wähle für jeden MA eine Aktion
        for i in range(self.num_employees):
            if random.random() < self.epsilon:
                # Zufällige Aktion
                actions[i] = np.random.randint(self.action_dim)
            else:
                # Beste Aktion basierend auf dem Modell
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                actions[i] = q_values.argmax().item()
        
        return actions
    
    def replay(self, batch_size):
        """
        Trainiert das Modell mit Erfahrungen aus dem Replay-Speicher.
        
        Args:
            batch_size: Anzahl der Erfahrungen, die pro Trainingsschritt verwendet werden
        """
        # Prüfe ob genug Erfahrungen vorhanden sind
        if len(self.memory) < batch_size:
            return
        
        # Wähle zufällige Erfahrungen aus
        batch = random.sample(self.memory, batch_size)
        
        # Bereite Daten für Training vor
        states = torch.FloatTensor([i[0] for i in batch]).to(self.device)
        actions = torch.LongTensor([i[1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in batch]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in batch]).to(self.device)
        
        # Berechne aktuelle Q-Werte für alle Aktionen
        current_q_values = self.model(states)
        
        # Wähle die Q-Werte für die ausgeführten Aktionen
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Berechne nächste Q-Werte mit Target-Netzwerk
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        
        # Berechne Ziel-Q-Werte
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Berechne Verlust und optimiere
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Reduziere Epsilon (weniger zufällige Aktionen über Zeit)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Aktualisiert das Target-Netzwerk mit den Gewichten des Haupt-Netzwerks."""
        self.target_model.load_state_dict(self.model.state_dict()) 