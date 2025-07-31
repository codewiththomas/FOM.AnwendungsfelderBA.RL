"""
Simple Convergence Analysis - Easy to understand single plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class SimpleConvergenceAnalyzer:
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

    def create_simple_convergence_plot(self, data, scenario_name, save_path):
        """Create a simple, single convergence plot"""
        episodes = data['episodes']
        scores = data['scores']

        # Calculate rolling mean
        rolling_mean = pd.Series(scores).rolling(window=100, min_periods=1).mean()

        plt.figure(figsize=(12, 6))

        # Plot raw scores with low alpha
        plt.plot(episodes, scores, alpha=0.3, color='lightblue', linewidth=0.5, label='Episode-Rewards')

        # Plot rolling mean prominently
        plt.plot(episodes, rolling_mean, color='darkblue', linewidth=3, label='Gleitender Durchschnitt (100 Episoden)')

        # Add some key statistics as text
        final_mean = np.mean(scores[-100:])
        best_score = np.max(scores)
        plt.axhline(y=final_mean, color='red', linestyle='--', alpha=0.7,
                   label=f'Finale Performance: {final_mean:.1f}')
        plt.axhline(y=best_score, color='green', linestyle='--', alpha=0.7,
                   label=f'Bester Reward: {best_score:.1f}')

        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title(f'Lernkonvergenz - {scenario_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add annotation for convergence phases
        if len(episodes) >= 500:
            plt.axvspan(0, 300, alpha=0.1, color='red', label='Exploration')
            plt.axvspan(300, 600, alpha=0.1, color='orange')
            plt.axvspan(600, len(episodes), alpha=0.1, color='green')

            # Add text annotations
            plt.text(150, plt.ylim()[1]*0.9, 'Exploration', ha='center', fontweight='bold', color='red')
            plt.text(450, plt.ylim()[1]*0.9, 'Stabilisierung', ha='center', fontweight='bold', color='orange')
            plt.text(min(800, len(episodes)-100), plt.ylim()[1]*0.9, 'Konvergenz', ha='center', fontweight='bold', color='green')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Einfaches Konvergenz-Diagramm gespeichert: {save_path}")
        plt.close()

    def create_simple_comparison_plot(self, data_a, data_b_alt):
        """Create a simple comparison plot"""
        plt.figure(figsize=(12, 6))

        # Calculate rolling means
        rolling_mean_a = pd.Series(data_a['scores']).rolling(window=100, min_periods=1).mean()
        rolling_mean_b = pd.Series(data_b_alt['scores']).rolling(window=100, min_periods=1).mean()

        # Plot both scenarios
        plt.plot(data_a['episodes'], rolling_mean_a, color='red', linewidth=3,
                label='Szenario A (problematisch)')
        plt.plot(data_b_alt['episodes'], rolling_mean_b, color='blue', linewidth=3,
                label='Szenario B Alternative (erfolgreich)')

        # Add horizontal lines for final performance
        final_a = np.mean(data_a['scores'][-100:])
        final_b = np.mean(data_b_alt['scores'][-100:])

        plt.axhline(y=final_a, color='red', linestyle='--', alpha=0.7,
                   label=f'Finale Performance A: {final_a:.1f}')
        plt.axhline(y=final_b, color='blue', linestyle='--', alpha=0.7,
                   label=f'Finale Performance B: {final_b:.1f}')

        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward (Gleitender Durchschnitt)', fontsize=12)
        plt.title('Vergleich der Lernkonvergenz: Szenario A vs. B Alternative', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add performance difference annotation
        diff = final_b - final_a
        plt.text(0.02, 0.98, f'Performance-Unterschied: {diff:.1f} Punkte',
                transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top')

        plt.tight_layout()
        comparison_path = self.base_dir / "simple_convergence_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"✅ Einfaches Vergleichs-Diagramm gespeichert: {comparison_path}")
        plt.close()

        return comparison_path

def main():
    print("🔍 EINFACHE KONVERGENZ-ANALYSE")
    print("=" * 40)

    analyzer = SimpleConvergenceAnalyzer()

    # Load data for both scenarios
    scenario_b_alt_path = analyzer.base_dir / "test_celine_rl_szenario_b_alternative"
    scenario_a_path = analyzer.base_dir / "test_celine_rl_szenario_a"

    data_b_alt = analyzer.load_training_data(scenario_b_alt_path)
    data_a = analyzer.load_training_data(scenario_a_path)

    if data_b_alt:
        print("\n📊 Erstelle einfaches Diagramm für Szenario B Alternative...")
        simple_b_path = analyzer.base_dir / "simple_convergence_szenario_b_alternative.png"
        analyzer.create_simple_convergence_plot(data_b_alt, "Szenario B Alternative", simple_b_path)

    if data_a and data_b_alt:
        print("\n🔄 Erstelle einfaches Vergleichs-Diagramm...")
        comparison_path = analyzer.create_simple_comparison_plot(data_a, data_b_alt)

        print(f"\n📈 ERSTELLTE DIAGRAMME:")
        print(f"• Szenario B Alternative: simple_convergence_szenario_b_alternative.png")
        print(f"• Vergleich beider Szenarien: simple_convergence_comparison.png")

    print("\n✅ Einfache Konvergenz-Analyse abgeschlossen!")

if __name__ == "__main__":
    main()