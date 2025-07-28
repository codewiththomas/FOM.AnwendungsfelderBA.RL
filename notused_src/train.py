from environment.shift_env import ShiftEnvironment
from agent.dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_progress(rewards, filename='training_progress.png'):
    """
    Erstellt einen Plot des Trainingsverlaufs.
    
    Args:
        rewards: Liste der Rewards pro Episode
        filename: Name der zu speichernden Datei
    """
    plt.figure(figsize=(12, 6))
    
    # Plot der einzelnen Rewards
    plt.plot(rewards, alpha=0.3, color='blue', label='Einzelne Rewards')
    
    # Plot des gleitenden Durchschnitts
    window_size = 20
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(rewards)), moving_avg, color='red', 
             label=f'Gleitender Durchschnitt (n={window_size})')
    
    plt.title('Trainingsverlauf')
    plt.xlabel('Episode')
    plt.ylabel('Gesamtbelohnung')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def train(episodes=2000, batch_size=64):  # Verlängert auf 2000 Episoden für bessere Konvergenz
    """
    Haupttrainingsfunktion für Wochen-basierte Schichtplanung.
    
    Args:
        episodes: Anzahl der Trainingsepisoden (Wochen)
        batch_size: Größe des Batches für das Training
    """
    print("Starte Training für Wochen-basierte Schichtplanung...")
    print(f"Episoden: {episodes}")
    print(f"Batch-Größe: {batch_size}")
    
    # Initialisiere Umgebung und Agent
    env = ShiftEnvironment('data/forecast.csv', 'data/employees.csv')
    state_dim = env.observation_space.shape[0]
    num_employees = env.num_employees
    agent = DQNAgent(state_dim, num_employees)
    
    # Listen für Tracking des Fortschritts
    rewards_history = []
    best_reward = float('-inf')
    
    # Trainingsschleife
    for episode in range(episodes):
        # Reset Umgebung für neue Woche
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # Episode durchlaufen (7 Tage)
        while not done:
            # Agent wählt Aktionen für alle Mitarbeiter für den aktuellen Tag
            actions = agent.act(state)
            
            # Führe Aktionen aus (plant einen Tag)
            next_state, reward, done, _, _ = env.step(actions)
            
            # Speichere Erfahrung
            agent.remember(state, actions[0], reward, next_state, done)  # Vereinfachung: Nur erste Aktion speichern
            
            # Trainiere Agent
            agent.replay(batch_size)
            
            # Update für nächsten Schritt
            state = next_state
            total_reward += reward
        
        # Aktualisiere Target-Netzwerk am Ende jeder Episode
        agent.update_target_model()
        
        # Speichere Reward
        rewards_history.append(total_reward)
        
        # Speichere besten Reward
        if total_reward > best_reward:
            best_reward = total_reward
            print(f"\nNeuer bester Reward: {best_reward:.2f}")
        
        # Ausgabe alle 50 Episoden (angepasst für längeres Training)
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])  # Durchschnitt der letzten 50 Episoden
            print(f"\nEpisode: {episode}")
            print(f"Aktueller Reward: {total_reward:.2f}")
            print(f"Durchschnitt (letzte 50): {avg_reward:.2f}")
            print(f"Bester Reward bisher: {best_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.2f}")
            print("------------------------")
    
    # Erstelle Plot
    plot_training_progress(rewards_history)
    
    # Exportiere beste Schichtplanung
    print("\nExportiere beste Schichtplanung für die Woche...")
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    env.export_best_shift_plan(os.path.join(results_dir, "beste_schichtplanung_woche.csv"))
    
    print("\nTraining abgeschlossen!")
    print(f"Bester Reward: {best_reward:.2f}")
    print(f"Durchschnitt (letzte 100): {np.mean(rewards_history[-100:]):.2f}")
    print(f"Gesamt-Episoden: {episodes}")

if __name__ == "__main__":
    train() 