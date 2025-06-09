"""
Training Script für Dienstplanung RL Agent
Verwendet Stable-Baselines3 PPO für das Training
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Pfad für Import hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
except ImportError as e:
    print(f"Fehler beim Importieren der Bibliotheken: {e}")
    print("Bitte installieren Sie die requirements.txt mit: pip install -r requirements.txt")
    sys.exit(1)

from src.environment.shift_scheduling_env import ShiftSchedulingEnv

class ShiftSchedulingTrainer:
    """Trainer für das Dienstplanungs RL Model"""

    def __init__(self, data_path: str = "data_csv", results_dir: str = "results"):
        self.data_path = data_path
        self.results_dir = results_dir
        self.model = None
        self.env = None
        self.training_stats = []

        # Ergebnisordner erstellen
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "plots"), exist_ok=True)

    def create_environment(self, max_steps: int = 500) -> ShiftSchedulingEnv:
        """Erstellt das Training Environment"""
        env = ShiftSchedulingEnv(data_path=self.data_path, max_steps=max_steps)
        env = Monitor(env, filename=os.path.join(self.results_dir, "logs", "training_log.csv"))
        return env

    def train_model(self,
                   total_timesteps: int = 100000,
                   max_steps: int = 500,
                   learning_rate: float = 3e-4,
                   batch_size: int = 64,
                   n_epochs: int = 10,
                   gamma: float = 0.99,
                   verbose: int = 1):
        """Trainiert das PPO Model"""

        print("Erstelle Training Environment...")
        self.env = self.create_environment(max_steps=max_steps)

        print("Initialisiere PPO Model...")
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=verbose,
            tensorboard_log=os.path.join(self.results_dir, "logs", "tensorboard")
        )

        # Callbacks für Training
        eval_env = self.create_environment(max_steps=max_steps)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.results_dir, "models", "best_model"),
            log_path=os.path.join(self.results_dir, "logs"),
            eval_freq=5000,
            deterministic=True,
            render=False
        )

        print(f"Starte Training für {total_timesteps} Timesteps...")
        start_time = datetime.now()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )

        end_time = datetime.now()
        training_duration = end_time - start_time

        print(f"Training abgeschlossen in {training_duration}")

        # Model speichern
        model_path = os.path.join(self.results_dir, "models", "final_model")
        self.model.save(model_path)
        print(f"Model gespeichert in: {model_path}")

        return training_duration

    def evaluate_model(self, n_eval_episodes: int = 10) -> dict:
        """Evaluiert das trainierte Model"""
        if self.model is None:
            raise ValueError("Model muss erst trainiert werden!")

        print(f"Evaluiere Model über {n_eval_episodes} Episoden...")

        eval_env = self.create_environment()
        evaluation_results = []

        for episode in range(n_eval_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            evaluation_results.append({
                'episode': episode + 1,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'service_level': info['service_level'],
                'constraint_violations': info['constraint_violations'],
                'hours_compliance': info['hours_compliance'],
                'schedule_completeness': info['schedule_completeness']
            })

            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Service Level={info['service_level']:.2%}, "
                  f"Violations={info['constraint_violations']}")

        # Statistiken berechnen
        results_df = pd.DataFrame(evaluation_results)

        summary_stats = {
            'mean_reward': results_df['episode_reward'].mean(),
            'std_reward': results_df['episode_reward'].std(),
            'mean_service_level': results_df['service_level'].mean(),
            'mean_violations': results_df['constraint_violations'].mean(),
            'mean_hours_compliance': results_df['hours_compliance'].mean(),
            'mean_completeness': results_df['schedule_completeness'].mean(),
            'episodes_evaluated': n_eval_episodes
        }

        # Ergebnisse speichern
        results_df.to_csv(os.path.join(self.results_dir, "evaluation_results.csv"), index=False)

        return summary_stats

    def generate_final_schedule(self) -> pd.DataFrame:
        """Generiert einen finalen optimalen Schedule"""
        if self.model is None:
            raise ValueError("Model muss erst trainiert werden!")

        print("Generiere finalen optimalen Schedule...")

        env = self.create_environment()
        obs, info = env.reset()
        done = False
        step_count = 0

        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            # Alle 50 Schritte Status ausgeben
            if step_count % 50 == 0:
                print(f"Schritt {step_count}: Service Level={info['service_level']:.2%}, "
                      f"Violations={info['constraint_violations']}")

        # Finalen Schedule als DataFrame
        final_schedule = env.get_final_schedule_df()

        # Schedule speichern
        schedule_path = os.path.join(self.results_dir, "final_schedule.csv")
        final_schedule.to_csv(schedule_path, index=False)
        print(f"Finaler Schedule gespeichert in: {schedule_path}")

        # Finale Statistiken
        final_stats = {
            'total_steps': step_count,
            'final_service_level': info['service_level'],
            'final_violations': info['constraint_violations'],
            'final_hours_compliance': info['hours_compliance'],
            'total_assigned_hours': info['total_assigned_hours'],
            'schedule_completeness': info['schedule_completeness']
        }

        print("\n=== FINALE SCHEDULE STATISTIKEN ===")
        for key, value in final_stats.items():
            if 'level' in key or 'compliance' in key or 'completeness' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value}")

        return final_schedule, final_stats

    def create_visualizations(self, evaluation_stats: dict):
        """Erstellt Visualisierungen der Trainingsergebnisse"""
        print("Erstelle Visualisierungen...")

        # Style setzen
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dienstplanung RL - Trainingsergebnisse', fontsize=16)

        # 1. Service Level
        axes[0, 0].bar(['Ergebnis'], [evaluation_stats['mean_service_level']],
                       color='green', alpha=0.7)
        axes[0, 0].axhline(y=0.8, color='red', linestyle='--', label='Ziel: 80%')
        axes[0, 0].set_ylabel('Service Level')
        axes[0, 0].set_title('Durchschnittlicher Service Level')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)

        # 2. Constraint Violations
        axes[0, 1].bar(['Ergebnis'], [evaluation_stats['mean_violations']],
                       color='red', alpha=0.7)
        axes[0, 1].set_ylabel('Anzahl Violations')
        axes[0, 1].set_title('Durchschnittliche Constraint Violations')

        # 3. Hours Compliance
        axes[1, 0].bar(['Ergebnis'], [evaluation_stats['mean_hours_compliance']],
                       color='blue', alpha=0.7)
        axes[1, 0].set_ylabel('Hours Compliance')
        axes[1, 0].set_title('Wochenstunden Compliance')
        axes[1, 0].set_ylim(0, 1)

        # 4. Schedule Completeness
        axes[1, 1].bar(['Ergebnis'], [evaluation_stats['mean_completeness']],
                       color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('Schedule Completeness')
        axes[1, 1].set_title('Schedule Vollständigkeit')
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, "plots", "training_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualisierungen gespeichert in: {plot_path}")

    def load_model(self, model_path: str):
        """Lädt ein bereits trainiertes Model"""
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"Model nicht gefunden: {model_path}")

        self.model = PPO.load(model_path)
        print(f"Model geladen von: {model_path}")

    def get_model_info(self) -> dict:
        """Gibt Informationen über das Model zurück"""
        if self.model is None:
            return {"status": "Kein Model geladen"}

        return {
            "policy": str(type(self.model.policy)),
            "learning_rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs
        }

def main():
    """Hauptfunktion für Training"""
    print("=== DIENSTPLANUNG RL TRAINING ===")

    # Trainer initialisieren
    trainer = ShiftSchedulingTrainer()

    # Training Parameter
    training_params = {
        'total_timesteps': 50000,  # Für ersten Test
        'max_steps': 300,
        'learning_rate': 3e-4,
        'batch_size': 64,
        'verbose': 1
    }

    try:
        # 1. Model trainieren
        print("\n1. Starte Model Training...")
        duration = trainer.train_model(**training_params)

        # 2. Model evaluieren
        print("\n2. Evaluiere trainiertes Model...")
        eval_stats = trainer.evaluate_model(n_eval_episodes=5)

        # 3. Finalen Schedule generieren
        print("\n3. Generiere finalen Schedule...")
        final_schedule, final_stats = trainer.generate_final_schedule()

        # 4. Visualisierungen erstellen
        print("\n4. Erstelle Visualisierungen...")
        trainer.create_visualizations(eval_stats)

        print("\n=== TRAINING ERFOLGREICH ABGESCHLOSSEN ===")
        print(f"Trainingsdauer: {duration}")
        print(f"Durchschnittlicher Service Level: {eval_stats['mean_service_level']:.2%}")
        print(f"Durchschnittliche Violations: {eval_stats['mean_violations']:.1f}")
        print(f"Schedule Einträge: {len(final_schedule)}")

    except Exception as e:
        print(f"Fehler beim Training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()