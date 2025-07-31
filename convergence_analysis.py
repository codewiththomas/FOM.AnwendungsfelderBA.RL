"""
Convergence Analysis for RL Scenarios
Analyzes learning convergence and generates diagrams for scenarios A and B alternative
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy import stats
import sys
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ConvergenceAnalyzer:
    def __init__(self):
        self.base_dir = Path(__file__).parent

    def load_training_data(self, scenario_path):
        """Load training progress data from JSON file"""
        json_file = scenario_path / "training_progress.json"
        if not json_file.exists():
            print(f"❌ Training data not found for {scenario_path.name}")
            return None

        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def calculate_convergence_metrics(self, scores):
        """Calculate convergence metrics for training scores"""
        scores = np.array(scores)

        # Rolling statistics
        window_size = 100
        rolling_mean = pd.Series(scores).rolling(window=window_size, min_periods=1).mean()
        rolling_std = pd.Series(scores).rolling(window=window_size, min_periods=1).std()

        # Convergence metrics
        final_100_mean = np.mean(scores[-100:])
        final_100_std = np.std(scores[-100:])
        max_score = np.max(scores)

        # Stability analysis (last 500 episodes)
        final_500 = scores[-500:] if len(scores) >= 500 else scores
        stability_coefficient = np.std(final_500) / np.mean(final_500) if np.mean(final_500) != 0 else 0

        # Find convergence point (where rolling std drops below threshold)
        convergence_threshold = 0.1 * np.std(scores[:500]) if len(scores) >= 500 else 0.1 * np.std(scores)
        convergence_episode = None
        for i in range(len(rolling_std)):
            if rolling_std.iloc[i] <= convergence_threshold and i > 500:
                convergence_episode = i
                break

        return {
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'final_100_mean': final_100_mean,
            'final_100_std': final_100_std,
            'max_score': max_score,
            'stability_coefficient': stability_coefficient,
            'convergence_episode': convergence_episode,
            'total_episodes': len(scores)
        }

    def create_convergence_plot(self, data, metrics, scenario_name, save_path):
        """Create comprehensive convergence analysis plot"""
        episodes = data['episodes']
        scores = data['scores']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Lernkonvergenz-Analyse - {scenario_name}', fontsize=16, fontweight='bold')

        # Plot 1: Training Progress with Rolling Mean
        ax1.plot(episodes, scores, alpha=0.3, color='lightblue', label='Episode Rewards')
        ax1.plot(episodes, metrics['rolling_mean'], color='darkblue', linewidth=2, label='Rolling Mean (100 episodes)')
        ax1.fill_between(episodes,
                        metrics['rolling_mean'] - metrics['rolling_std'],
                        metrics['rolling_mean'] + metrics['rolling_std'],
                        alpha=0.2, color='blue', label='±1 Std Dev')

        # Mark convergence point if found
        if metrics['convergence_episode']:
            ax1.axvline(x=metrics['convergence_episode'], color='red', linestyle='--',
                       label=f'Konvergenz (Episode {metrics["convergence_episode"]})')

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Trainingsverlauf und Konvergenz')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Standard Deviation Over Time
        ax2.plot(episodes, metrics['rolling_std'], color='orange', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Standard Abweichung')
        ax2.set_title('Stabilität des Lernprozesses')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Distribution of Rewards in Different Phases
        early_phase = scores[:500] if len(scores) >= 500 else scores[:len(scores)//3]
        middle_phase = scores[500:1500] if len(scores) >= 1500 else scores[len(scores)//3:2*len(scores)//3]
        late_phase = scores[-500:] if len(scores) >= 500 else scores[2*len(scores)//3:]

        ax3.hist(early_phase, alpha=0.7, bins=30, label='Frühe Phase (0-500)', color='red')
        ax3.hist(middle_phase, alpha=0.7, bins=30, label='Mittlere Phase (500-1500)', color='orange')
        ax3.hist(late_phase, alpha=0.7, bins=30, label='Späte Phase (letzte 500)', color='green')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Häufigkeit')
        ax3.set_title('Reward-Verteilung nach Trainingsphasen')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Cumulative Maximum Reward
        cumulative_max = np.maximum.accumulate(scores)
        ax4.plot(episodes, cumulative_max, color='green', linewidth=2, label='Kumulatives Maximum')
        ax4.plot(episodes, scores, alpha=0.3, color='lightgreen', label='Episode Rewards')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        ax4.set_title('Progression des besten Rewards')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Konvergenz-Diagramm gespeichert: {save_path}")

        return fig

    def generate_convergence_report(self, data, metrics, scenario_name):
        """Generate detailed convergence report"""
        report = f"""
=== KONVERGENZ-ANALYSE: {scenario_name.upper()} ===

📊 ALLGEMEINE STATISTIKEN:
• Gesamte Episoden: {metrics['total_episodes']:,}
• Bester Reward: {metrics['max_score']:.2f}
• Durchschnitt (letzte 100 Episoden): {metrics['final_100_mean']:.2f}
• Standardabweichung (letzte 100): {metrics['final_100_std']:.2f}
• Stabilitätskoeffizient: {metrics['stability_coefficient']:.4f}

🎯 KONVERGENZ-ANALYSE:
"""
        if metrics['convergence_episode']:
            report += f"• Konvergenz erreicht bei Episode: {metrics['convergence_episode']:,}\n"
            report += f"• Konvergenz-Phase: {(metrics['convergence_episode']/metrics['total_episodes']*100):.1f}% des Trainings\n"
        else:
            report += "• Vollständige Konvergenz noch nicht erreicht\n"

        # Phase analysis
        total_episodes = metrics['total_episodes']
        early_phase = data['scores'][:500] if total_episodes >= 500 else data['scores'][:total_episodes//3]
        late_phase = data['scores'][-500:] if total_episodes >= 500 else data['scores'][2*total_episodes//3:]

        improvement = np.mean(late_phase) - np.mean(early_phase)
        improvement_pct = (improvement / np.mean(early_phase)) * 100 if np.mean(early_phase) != 0 else 0

        report += f"""
📈 LERNFORTSCHRITT:
• Verbesserung früh → spät: {improvement:+.2f} Punkte ({improvement_pct:+.1f}%)
• Variabilität früh: {np.std(early_phase):.2f}
• Variabilität spät: {np.std(late_phase):.2f}
• Reduktion der Variabilität: {((np.std(early_phase) - np.std(late_phase))/np.std(early_phase)*100):.1f}%

🔍 EXPLORATION VS. EXPLOITATION:
• Finale Epsilon: {data.get('final_epsilon', 'N/A')}
• Performance-Stabilität (CV): {metrics['stability_coefficient']:.4f}
"""

        return report

    def run_analysis_for_scenario_b_alt(self):
        """Analyze scenario B alternative with existing data"""
        scenario_path = self.base_dir / "test_celine_rl_szenario_b_alternative"
        data = self.load_training_data(scenario_path)

        if data is None:
            return None

        metrics = self.calculate_convergence_metrics(data['scores'])

        # Create convergence plot
        plot_path = self.base_dir / "convergence_szenario_b_alternative.png"
        self.create_convergence_plot(data, metrics, "Szenario B Alternative", plot_path)

        # Generate report
        report = self.generate_convergence_report(data, metrics, "Szenario B Alternative")

        return {
            'data': data,
            'metrics': metrics,
            'report': report,
            'plot_path': plot_path
        }

    def run_scenario_a_with_logging(self):
        """Run scenario A training with proper logging"""
        print("🚀 Starte Training für Szenario A mit Logging...")

        # Import the RL agent from scenario A
        sys.path.append(str(self.base_dir / "test_celine_rl_szenario_a"))

        try:
            from rl_shift_optimizer import EqualDemandRLAgent

            # Modify the agent to track training progress
            base_dir = self.base_dir
            forecast_file = base_dir / "data" / "forecast_szenario_a.csv"
            employees_file = base_dir / "data" / "employees.csv"

            agent = EqualDemandRLAgent(forecast_file, employees_file)

            # Custom training with logging
            episodes = 2000
            print(f"🎓 Training RL agent for {episodes} episodes...")

            best_schedule = None
            best_score = float('-inf')
            episode_scores = []
            episodes_list = []

            for episode in range(episodes):
                schedule, decisions = agent.generate_schedule_episode()

                # Update Q-table based on decisions
                for state_key, action, reward in decisions:
                    agent.update_q_table(state_key, action, reward)

                # Evaluate schedule quality
                score = agent.evaluate_schedule(schedule)
                episode_scores.append(score)
                episodes_list.append(episode)

                if score > best_score:
                    best_score = score
                    best_schedule = schedule

                # Decay epsilon (reduce exploration over time)
                if episode % 100 == 0:
                    agent.epsilon = max(0.05, agent.epsilon * 0.95)
                    avg_score = np.mean(episode_scores[-100:]) if len(episode_scores) >= 100 else np.mean(episode_scores)
                    print(f"Episode {episode}: Best score = {best_score:.2f}, Avg score = {avg_score:.2f}, Epsilon = {agent.epsilon:.3f}")

            # Save training progress
            training_data = {
                'episodes': episodes_list,
                'scores': episode_scores,
                'best_score': best_score,
                'final_epsilon': agent.epsilon
            }

            # Save to JSON file
            json_path = self.base_dir / "test_celine_rl_szenario_a" / "training_progress.json"
            with open(json_path, 'w') as f:
                json.dump(training_data, f, indent=2)

            print(f"✅ Training completed! Best score: {best_score:.2f}")
            print(f"📊 Training data saved to: {json_path}")

            # Analyze the results
            metrics = self.calculate_convergence_metrics(episode_scores)

            # Create convergence plot
            plot_path = self.base_dir / "convergence_szenario_a.png"
            self.create_convergence_plot(training_data, metrics, "Szenario A", plot_path)

            # Generate report
            report = self.generate_convergence_report(training_data, metrics, "Szenario A")

            return {
                'data': training_data,
                'metrics': metrics,
                'report': report,
                'plot_path': plot_path
            }

        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_combined_comparison_plot(self, results_a, results_b_alt):
        """Create a combined comparison plot for both scenarios"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Vergleichende Konvergenz-Analyse: Szenario A vs. B Alternative', fontsize=16, fontweight='bold')

        # Plot 1: Training Progress Comparison
        if results_a:
            ax1.plot(results_a['data']['episodes'], results_a['metrics']['rolling_mean'],
                    color='blue', linewidth=2, label='Szenario A')

        if results_b_alt:
            ax1.plot(results_b_alt['data']['episodes'], results_b_alt['metrics']['rolling_mean'],
                    color='red', linewidth=2, label='Szenario B Alternative')

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Rolling Mean Reward')
        ax1.set_title('Konvergenz-Vergleich (Rolling Mean)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Stability Comparison
        if results_a:
            ax2.plot(results_a['data']['episodes'], results_a['metrics']['rolling_std'],
                    color='blue', linewidth=2, label='Szenario A')

        if results_b_alt:
            ax2.plot(results_b_alt['data']['episodes'], results_b_alt['metrics']['rolling_std'],
                    color='red', linewidth=2, label='Szenario B Alternative')

        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Rolling Standard Deviation')
        ax2.set_title('Stabilität-Vergleich')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Final Performance Comparison
        scenarios = []
        final_means = []
        final_stds = []

        if results_a:
            scenarios.append('Szenario A')
            final_means.append(results_a['metrics']['final_100_mean'])
            final_stds.append(results_a['metrics']['final_100_std'])

        if results_b_alt:
            scenarios.append('Szenario B\nAlternative')
            final_means.append(results_b_alt['metrics']['final_100_mean'])
            final_stds.append(results_b_alt['metrics']['final_100_std'])

        x_pos = np.arange(len(scenarios))
        bars = ax3.bar(x_pos, final_means, yerr=final_stds, capsize=5,
                      color=['blue', 'red'][:len(scenarios)], alpha=0.7)
        ax3.set_xlabel('Szenario')
        ax3.set_ylabel('Durchschnittlicher Reward (letzte 100 Episoden)')
        ax3.set_title('Finale Performance (mit Standardabweichung)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenarios)
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, final_means, final_stds)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')

        # Plot 4: Convergence Metrics Comparison
        metrics_names = ['Max Score', 'Stability\nCoefficient', 'Final 100\nMean']

        if results_a and results_b_alt:
            a_values = [results_a['metrics']['max_score'],
                       results_a['metrics']['stability_coefficient'] * 1000,  # Scale for visibility
                       results_a['metrics']['final_100_mean']]
            b_values = [results_b_alt['metrics']['max_score'],
                       results_b_alt['metrics']['stability_coefficient'] * 1000,  # Scale for visibility
                       results_b_alt['metrics']['final_100_mean']]

            x_pos = np.arange(len(metrics_names))
            width = 0.35

            ax4.bar(x_pos - width/2, a_values, width, label='Szenario A', color='blue', alpha=0.7)
            ax4.bar(x_pos + width/2, b_values, width, label='Szenario B Alternative', color='red', alpha=0.7)

            ax4.set_xlabel('Metriken')
            ax4.set_ylabel('Werte')
            ax4.set_title('Vergleich der Konvergenz-Metriken')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(metrics_names)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Note about stability coefficient scaling
            ax4.text(0.02, 0.98, 'Hinweis: Stability Coefficient × 1000',
                    transform=ax4.transAxes, fontsize=8, va='top')

        plt.tight_layout()
        comparison_path = self.base_dir / "convergence_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"✅ Vergleichs-Diagramm gespeichert: {comparison_path}")

        return comparison_path

def main():
    print("🔍 KONVERGENZ-ANALYSE FÜR RL-SZENARIEN")
    print("=" * 50)

    analyzer = ConvergenceAnalyzer()

    # Analyze Scenario B Alternative (existing data)
    print("\n📊 Analysiere Szenario B Alternative...")
    results_b_alt = analyzer.run_analysis_for_scenario_b_alt()

    if results_b_alt:
        print(results_b_alt['report'])

    # Run Scenario A with logging
    print("\n🎯 Führe Training für Szenario A durch...")
    results_a = analyzer.run_scenario_a_with_logging()

    if results_a:
        print(results_a['report'])

    # Create combined comparison
    if results_a and results_b_alt:
        print("\n🔄 Erstelle Vergleichs-Analyse...")
        comparison_path = analyzer.create_combined_comparison_plot(results_a, results_b_alt)

        print(f"\n📈 ZUSAMMENFASSUNG:")
        print(f"• Szenario A Diagramm: {results_a['plot_path']}")
        print(f"• Szenario B Alternative Diagramm: {results_b_alt['plot_path']}")
        print(f"• Vergleichs-Diagramm: {comparison_path}")

    print("\n✅ Konvergenz-Analyse abgeschlossen!")
    print("Die generierten Diagramme können für die Dokumentation verwendet werden.")

if __name__ == "__main__":
    main()