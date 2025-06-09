"""
Einfacher Test des Dienstplanungs-Environments
Testet die Grundfunktionalität ohne vollständige RL-Bibliotheken
"""

import sys
import os
import numpy as np

# Pfad für Import hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_data_loader():
    """Testet den DataLoader"""
    print("=== TESTE DATA LOADER ===")

    try:
        from src.utils.data_loader import DataLoader

        # DataLoader initialisieren
        loader = DataLoader("data_csv")

        # Daten laden
        data = loader.load_all_data()

        # Statistiken ausgeben
        stats = loader.get_summary_stats()
        print("Zusammenfassungsstatistiken:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Qualifikationen testen
        qualifications = loader.get_agent_qualifications()
        print(f"\nAnzahl Agenten mit Qualifikationen: {len(qualifications)}")

        # Forecast testen
        forecast = loader.preprocess_forecast()
        print(f"Forecast Zeiträume: {len(forecast)} Zeitslots")

        # Test erfolgreich
        print("✅ DataLoader Test erfolgreich!")
        return True

    except Exception as e:
        print(f"❌ DataLoader Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_basic():
    """Testet das Environment ohne RL-Bibliotheken"""
    print("\n=== TESTE ENVIRONMENT (Basic) ===")

    try:
        # Versuche, Environment zu importieren
        from src.environment.shift_scheduling_env import ShiftSchedulingEnv

        # Environment erstellen
        env = ShiftSchedulingEnv("data_csv", max_steps=50)

        print(f"Environment erstellt!")
        print(f"Anzahl Agenten: {env.num_agents}")
        print(f"Anzahl Zeitslots: {env.num_time_slots}")
        print(f"Action Space: {env.action_space}")
        print(f"Observation Space Shape: {env.observation_space.shape}")

        # Reset testen
        obs, info = env.reset()
        print(f"Reset erfolgreich. Observation shape: {obs.shape}")
        print(f"Info: {info}")

        # Einige zufällige Aktionen testen
        print("\nTeste zufällige Aktionen:")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Schritt {i+1}: Action={action}, Reward={reward:.3f}, "
                  f"Service Level={info['service_level']:.2%}, "
                  f"Violations={info['constraint_violations']}")

            if terminated or truncated:
                print("Episode beendet!")
                break

        # Finalen Schedule testen
        final_schedule = env.get_final_schedule_df()
        print(f"\nFinaler Schedule: {len(final_schedule)} Einträge")

        print("✅ Environment Basic Test erfolgreich!")
        return True

    except Exception as e:
        print(f"❌ Environment Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_manual():
    """Manueller Test mit strategischen Aktionen"""
    print("\n=== TESTE ENVIRONMENT (Manual Strategy) ===")

    try:
        from src.environment.shift_scheduling_env import ShiftSchedulingEnv

        # Environment erstellen
        env = ShiftSchedulingEnv("data_csv", max_steps=100)
        obs, info = env.reset()

        print("Strategischer Test: Versuche hohen Service Level zu erreichen")

        # Strategische Aktionen: Agenten zu den Hauptzeiten zuweisen
        high_demand_slots = []

        # Finde Zeitslots mit hoher Nachfrage
        for ts in range(min(20, env.num_time_slots)):  # Erste 20 Slots testen
            required = env.data_loader.calculate_required_agents(ts)
            if required > 1:
                high_demand_slots.append((ts, required))

        print(f"Slots mit hoher Nachfrage: {len(high_demand_slots)}")

        # Weise Agenten zu diesen Slots zu
        actions_taken = 0
        for ts, required in high_demand_slots[:10]:  # Nur erste 10
            for agent_idx in range(min(required + 1, env.num_agents)):
                action = agent_idx * env.num_time_slots + ts
                obs, reward, terminated, truncated, info = env.step(action)
                actions_taken += 1

                if actions_taken % 5 == 0:
                    print(f"Nach {actions_taken} Aktionen: "
                          f"Service Level={info['service_level']:.2%}, "
                          f"Violations={info['constraint_violations']}")

                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        # Finale Statistiken
        print(f"\n=== FINALE ERGEBNISSE ===")
        print(f"Aktionen ausgeführt: {actions_taken}")
        print(f"Service Level: {info['service_level']:.2%}")
        print(f"Constraint Violations: {info['constraint_violations']}")
        print(f"Hours Compliance: {info['hours_compliance']:.2%}")
        print(f"Total Assigned Hours: {info['total_assigned_hours']:.1f}")

        # Schedule anzeigen
        final_schedule = env.get_final_schedule_df()
        print(f"Schedule Einträge: {len(final_schedule)}")

        if len(final_schedule) > 0:
            print("\nErste 5 Schedule-Einträge:")
            print(final_schedule.head())

        print("✅ Environment Manual Test erfolgreich!")
        return True

    except Exception as e:
        print(f"❌ Environment Manual Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Haupttestfunktion"""
    print("🚀 STARTE DIENSTPLANUNG RL TESTS")
    print("=" * 50)

    # Tests ausführen
    tests = [
        ("DataLoader", test_data_loader),
        ("Environment Basic", test_environment_basic),
        ("Environment Manual", test_environment_manual)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔄 Führe {test_name} Test aus...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Unerwarteter Fehler in {test_name}: {e}")
            results.append((test_name, False))

    # Ergebnisse zusammenfassen
    print("\n" + "=" * 50)
    print("📊 TEST ERGEBNISSE:")
    successful_tests = 0

    for test_name, success in results:
        status = "✅ ERFOLGREICH" if success else "❌ FEHLGESCHLAGEN"
        print(f"  {test_name}: {status}")
        if success:
            successful_tests += 1

    print(f"\n🎯 {successful_tests}/{len(tests)} Tests erfolgreich")

    if successful_tests == len(tests):
        print("🎉 Alle Tests bestanden! Das Environment ist bereit für RL Training.")
        return True
    else:
        print("⚠️  Einige Tests sind fehlgeschlagen. Bitte Fehler beheben.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)