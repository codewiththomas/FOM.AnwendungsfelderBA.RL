"""
Hauptskript für das Training des Dienstplanungs RL Agents
Kombiniert alle Komponenten für ein vollständiges Training
"""

import os
import sys
import argparse
from datetime import datetime

def main():
    """Hauptfunktion für das RL Training"""
    print("🚀 DIENSTPLANUNG RL AGENT")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Dienstplanung RL Agent Training")
    parser.add_argument("--mode", choices=["test", "train", "eval"], default="test",
                       help="Modus: test (Environment testen), train (Model trainieren), eval (Model evaluieren)")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Anzahl Training Timesteps")
    parser.add_argument("--max_steps", type=int, default=300,
                       help="Maximale Schritte pro Episode")
    parser.add_argument("--data_path", default="data_csv",
                       help="Pfad zu CSV-Daten")
    parser.add_argument("--results_dir", default="results",
                       help="Pfad für Ergebnisse")
    parser.add_argument("--model_path", default=None,
                       help="Pfad zu vortrainiertem Model (für eval)")

    args = parser.parse_args()

    if args.mode == "test":
        print("🔧 TESTE ENVIRONMENT...")
        return run_tests()
    elif args.mode == "train":
        print("🏋️ STARTE TRAINING...")
        return run_training(args)
    elif args.mode == "eval":
        print("📊 EVALUIERE MODEL...")
        return run_evaluation(args)

def run_tests():
    """Führt Environment Tests aus"""
    try:
        import test_environment
        return test_environment.main()
    except ImportError:
        print("❌ test_environment.py nicht gefunden!")
        return False

def run_training(args):
    """Führt das RL Training aus"""
    try:
        # Versuche RL Bibliotheken zu importieren
        from src.models.train_agent import ShiftSchedulingTrainer

        print(f"📋 TRAINING PARAMETER:")
        print(f"  Timesteps: {args.timesteps}")
        print(f"  Max Steps: {args.max_steps}")
        print(f"  Data Path: {args.data_path}")
        print(f"  Results Dir: {args.results_dir}")

        # Trainer initialisieren
        trainer = ShiftSchedulingTrainer(
            data_path=args.data_path,
            results_dir=args.results_dir
        )

        # Training starten
        duration = trainer.train_model(
            total_timesteps=args.timesteps,
            max_steps=args.max_steps,
            verbose=1
        )

        # Evaluation
        print("\n📊 EVALUIERE TRAINIERTES MODEL...")
        eval_stats = trainer.evaluate_model(n_eval_episodes=5)

        # Finalen Schedule generieren
        print("\n📅 GENERIERE FINALEN SCHEDULE...")
        final_schedule, final_stats = trainer.generate_final_schedule()

        # Visualisierungen
        print("\n📈 ERSTELLE VISUALISIERUNGEN...")
        trainer.create_visualizations(eval_stats)

        # Zusammenfassung
        print("\n✅ TRAINING ERFOLGREICH ABGESCHLOSSEN!")
        print(f"⏱️  Trainingsdauer: {duration}")
        print(f"🎯 Service Level: {eval_stats['mean_service_level']:.2%}")
        print(f"⚠️  Violations: {eval_stats['mean_violations']:.1f}")
        print(f"📄 Schedule Einträge: {len(final_schedule)}")

        return True

    except ImportError as e:
        print(f"❌ RL Bibliotheken nicht verfügbar: {e}")
        print("💡 Installieren Sie die requirements.txt: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Training fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_evaluation(args):
    """Evaluiert ein vortrainiertes Model"""
    try:
        from src.models.train_agent import ShiftSchedulingTrainer

        if args.model_path is None:
            print("❌ Kein Model-Pfad angegeben! Verwenden Sie --model_path")
            return False

        print(f"📊 EVALUIERE MODEL: {args.model_path}")

        # Trainer initialisieren
        trainer = ShiftSchedulingTrainer(
            data_path=args.data_path,
            results_dir=args.results_dir
        )

        # Model laden
        trainer.load_model(args.model_path)

        # Evaluation
        eval_stats = trainer.evaluate_model(n_eval_episodes=10)

        # Finalen Schedule generieren
        final_schedule, final_stats = trainer.generate_final_schedule()

        # Ergebnisse anzeigen
        print("\n📊 EVALUATIONSERGEBNISSE:")
        print(f"🎯 Service Level: {eval_stats['mean_service_level']:.2%}")
        print(f"⚠️  Violations: {eval_stats['mean_violations']:.1f}")
        print(f"📊 Hours Compliance: {eval_stats['mean_hours_compliance']:.2%}")
        print(f"📄 Schedule Einträge: {len(final_schedule)}")

        return True

    except Exception as e:
        print(f"❌ Evaluation fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_requirements():
    """Überprüft ob alle Anforderungen erfüllt sind"""
    print("🔍 ÜBERPRÜFE ANFORDERUNGEN...")

    # Prüfe CSV-Daten
    required_files = ["agents.csv", "lines.csv", "forecast.csv", "constraints.csv"]
    data_path = "data_csv"

    if not os.path.exists(data_path):
        print(f"❌ Datenordner nicht gefunden: {data_path}")
        return False

    for file in required_files:
        file_path = os.path.join(data_path, file)
        if not os.path.exists(file_path):
            print(f"❌ Datei nicht gefunden: {file_path}")
            return False
        else:
            print(f"✅ {file} gefunden")

    # Prüfe Python-Module
    try:
        import pandas
        import numpy
        print("✅ Basis-Bibliotheken verfügbar")
    except ImportError as e:
        print(f"❌ Basis-Bibliotheken fehlen: {e}")
        return False

    print("✅ Alle Anforderungen erfüllt")
    return True

if __name__ == "__main__":
    # Anforderungen prüfen
    if not check_requirements():
        print("\n⚠️  Bitte beheben Sie die obigen Probleme vor dem Start.")
        sys.exit(1)

    # Hauptprogramm ausführen
    success = main()

    if success:
        print("\n🎉 PROGRAMM ERFOLGREICH ABGESCHLOSSEN!")
    else:
        print("\n❌ PROGRAMM MIT FEHLERN BEENDET!")

    sys.exit(0 if success else 1)