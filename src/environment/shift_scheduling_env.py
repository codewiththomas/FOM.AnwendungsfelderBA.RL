"""
Custom Gymnasium Environment für Dienstplanung
Single-Agent RL Environment für optimale Schichtplanung
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any
import sys
import os

# Pfad für Import hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.data_loader import DataLoader

class ShiftSchedulingEnv(gym.Env):
    """
    Gymnasium Environment für Dienstplanung

    Der Agent lernt, Mitarbeiter optimal zu Zeitslots zuzuweisen, um:
    - Service Level zu erreichen (80/20)
    - Alle Constraints zu erfüllen
    - Kosten zu minimieren
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, data_path: str = "data_csv", max_steps: int = 500):
        super().__init__()

        # Daten laden
        self.data_loader = DataLoader(data_path)
        self.data = self.data_loader.load_all_data()

        # Environment Parameter
        self.max_steps = max_steps
        self.current_step = 0

        # Datenstrukturen
        self.agents = self.data['agents']
        self.forecast = self.data_loader.preprocess_forecast()
        self.constraints_weights = self.data_loader.get_constraint_weights()
        self.agent_qualifications = self.data_loader.get_agent_qualifications()

        # Environment State
        self.num_agents = len(self.agents)
        self.num_time_slots = len(self.forecast)
        self.schedule = np.zeros((self.num_agents, self.num_time_slots), dtype=np.int8)

        # Action Space: Agent auswählen (0 bis num_agents-1) und Zeitslot (0 bis num_time_slots-1)
        # Vereinfacht: Eine Aktion = ein Agent-Zeitslot Paar
        self.action_space = spaces.Discrete(self.num_agents * self.num_time_slots)

        # Observation Space:
        # - Aktueller Schedule (flattened)
        # - Forecast Information
        # - Agent Verfügbarkeit
        # - Time Information
        obs_size = (
            self.num_agents * self.num_time_slots +  # Schedule
            self.num_time_slots +                    # Forecast (erwartete Kontakte)
            self.num_agents +                        # Agent Wochenstunden verfügbar
            self.num_time_slots +                    # Required agents per time slot
            7                                        # Zusätzliche Features (current_step, etc.)
        )

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )

        # Für Statistiken
        self.total_reward = 0
        self.constraint_violations = 0
        self.service_level_achieved = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset des Environments"""
        super().reset(seed=seed)

        # State zurücksetzen
        self.current_step = 0
        self.schedule = np.zeros((self.num_agents, self.num_time_slots), dtype=np.int8)
        self.total_reward = 0
        self.constraint_violations = 0
        self.service_level_achieved = 0

        # Initiale Observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Führt eine Aktion aus"""
        self.current_step += 1

        # Aktion decodieren: Agent und Zeitslot
        agent_idx = action // self.num_time_slots
        time_slot = action % self.num_time_slots

        # Aktion ausführen (Toggle: 0->1 oder 1->0)
        self.schedule[agent_idx, time_slot] = 1 - self.schedule[agent_idx, time_slot]

        # Reward berechnen
        reward = self._calculate_reward(agent_idx, time_slot)
        self.total_reward += reward

        # Episode beenden?
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        # Observation und Info
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Erstellt die Observation für den Agent"""
        obs = []

        # 1. Aktueller Schedule (flattened und normalisiert)
        obs.extend(self.schedule.flatten())

        # 2. Forecast Information (normalisiert)
        max_contacts = self.forecast['Erwartete_Kontakte'].max()
        forecast_norm = self.forecast['Erwartete_Kontakte'].values / max_contacts
        obs.extend(forecast_norm)

        # 3. Agent Verfügbarkeit (Wochenstunden normalisiert)
        max_hours = self.agents['Wochenstunden'].max()
        agent_hours_norm = self.agents['Wochenstunden'].values / max_hours
        obs.extend(agent_hours_norm)

        # 4. Required agents per time slot (normalisiert)
        required_agents = [self.data_loader.calculate_required_agents(ts) for ts in range(self.num_time_slots)]
        max_required = max(required_agents)
        required_norm = [r / max_required for r in required_agents]
        obs.extend(required_norm)

        # 5. Zusätzliche Features
        obs.extend([
            self.current_step / self.max_steps,  # Progress
            self._get_current_service_level(),   # Aktueller Service Level
            self._get_total_assigned_hours() / (self.agents['Wochenstunden'].sum()),  # Stunden-Auslastung
            self._count_constraint_violations() / 10,  # Constraint Violations (normalisiert)
            len(self._get_unassigned_agents()) / self.num_agents,  # Unassigned agents ratio
            self._get_avg_agents_per_slot() / self.num_agents,     # Avg agents per slot
            self._calculate_schedule_completeness()  # Schedule completeness
        ])

        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, agent_idx: int, time_slot: int) -> float:
        """Berechnet den Reward für eine Aktion"""
        reward = 0.0

        # Basis-Reward für Zuweisung/Entfernung
        if self.schedule[agent_idx, time_slot] == 1:  # Agent wurde zugewiesen
            reward += 0.1
        else:  # Agent wurde entfernt
            reward -= 0.05

        # Service Level Reward
        service_level = self._get_current_service_level()
        if service_level >= 0.8:  # 80% Service Level erreicht
            reward += 2.0 * service_level
        else:
            reward -= 1.0 * (0.8 - service_level)

        # Constraint Penalties
        constraint_violations = self._count_constraint_violations()
        reward -= 0.5 * constraint_violations

        # Effizienz-Bonus: Nicht zu viele Agenten zuweisen
        total_assigned = np.sum(self.schedule)
        optimal_assignments = sum(self.data_loader.calculate_required_agents(ts) for ts in range(self.num_time_slots))
        efficiency = 1.0 - abs(total_assigned - optimal_assignments) / optimal_assignments
        reward += 0.5 * efficiency

        # Wochenstunden-Compliance
        hours_compliance = self._check_hours_compliance()
        reward += 0.3 * hours_compliance

        return reward

    def _get_current_service_level(self) -> float:
        """Berechnet aktuellen Service Level (% der Zeitslots mit ausreichend Agenten)"""
        slots_covered = 0
        for ts in range(self.num_time_slots):
            assigned_agents = np.sum(self.schedule[:, ts])
            required_agents = self.data_loader.calculate_required_agents(ts)
            if assigned_agents >= required_agents:
                slots_covered += 1

        return slots_covered / self.num_time_slots if self.num_time_slots > 0 else 0.0

    def _count_constraint_violations(self) -> int:
        """Zählt Constraint-Verletzungen"""
        violations = 0

        for agent_idx in range(self.num_agents):
            agent_schedule = self.schedule[agent_idx, :]

            # Schichtlängen-Constraints
            violations += self._check_shift_length_violations(agent_schedule)

            # Ruhezeit-Constraints
            violations += self._check_rest_time_violations(agent_schedule)

            # Pausen-Constraints
            violations += self._check_break_violations(agent_schedule)

        return violations

    def _check_shift_length_violations(self, agent_schedule: np.ndarray) -> int:
        """Überprüft Schichtlängen-Constraints"""
        violations = 0

        # Kontinuierliche Schichten finden
        shifts = self._find_continuous_shifts(agent_schedule)

        for shift_start, shift_end in shifts:
            shift_length = (shift_end - shift_start) * 0.5  # 30-Min Slots -> Stunden

            # Minimale Schichtlänge (3h)
            if shift_length < 3:
                violations += 1

            # Maximale Schichtlänge (10h)
            if shift_length > 10:
                violations += 1

        return violations

    def _check_rest_time_violations(self, agent_schedule: np.ndarray) -> int:
        """Überprüft Ruhezeit-Constraints"""
        violations = 0
        shifts = self._find_continuous_shifts(agent_schedule)

        for i in range(len(shifts) - 1):
            current_end = shifts[i][1]
            next_start = shifts[i + 1][0]
            rest_time = (next_start - current_end) * 0.5  # in Stunden

            # Mindest-Ruhezeit 11 Stunden
            if rest_time < 11:
                violations += 1

        return violations

    def _check_break_violations(self, agent_schedule: np.ndarray) -> int:
        """Überprüft Pausen-Constraints"""
        violations = 0
        shifts = self._find_continuous_shifts(agent_schedule)

        for shift_start, shift_end in shifts:
            shift_length = (shift_end - shift_start) * 0.5

            # Pause bei Schichten über 6h erforderlich
            if shift_length > 6:
                violations += 1  # Vereinfacht: Pause nicht explizit modelliert

            # Zusätzliche Pause bei Schichten über 9h
            if shift_length > 9:
                violations += 1

        return violations

    def _find_continuous_shifts(self, agent_schedule: np.ndarray) -> List[Tuple[int, int]]:
        """Findet kontinuierliche Schichten eines Agenten"""
        shifts = []
        in_shift = False
        shift_start = 0

        for i, assigned in enumerate(agent_schedule):
            if assigned == 1 and not in_shift:
                in_shift = True
                shift_start = i
            elif assigned == 0 and in_shift:
                in_shift = False
                shifts.append((shift_start, i))

        # Letzte Schicht, falls sie bis zum Ende geht
        if in_shift:
            shifts.append((shift_start, len(agent_schedule)))

        return shifts

    def _check_hours_compliance(self) -> float:
        """Überprüft Einhaltung der Wochenstunden"""
        total_compliance = 0.0

        for agent_idx in range(self.num_agents):
            contracted_hours = self.agents.iloc[agent_idx]['Wochenstunden']
            assigned_hours = np.sum(self.schedule[agent_idx, :]) * 0.5  # 30-Min Slots

            # Compliance: Je näher an contracted hours, desto besser
            if contracted_hours > 0:
                compliance = 1.0 - abs(assigned_hours - contracted_hours) / contracted_hours
                total_compliance += max(0, compliance)

        return total_compliance / self.num_agents if self.num_agents > 0 else 0.0

    def _get_total_assigned_hours(self) -> float:
        """Gibt total zugewiesene Stunden zurück"""
        return np.sum(self.schedule) * 0.5

    def _get_unassigned_agents(self) -> List[int]:
        """Gibt Liste der nicht zugewiesenen Agenten zurück"""
        unassigned = []
        for agent_idx in range(self.num_agents):
            if np.sum(self.schedule[agent_idx, :]) == 0:
                unassigned.append(agent_idx)
        return unassigned

    def _get_avg_agents_per_slot(self) -> float:
        """Durchschnittliche Anzahl Agenten pro Zeitslot"""
        return np.mean(np.sum(self.schedule, axis=0))

    def _calculate_schedule_completeness(self) -> float:
        """Berechnet wie vollständig der Schedule ist"""
        total_possible = self.num_agents * self.num_time_slots
        total_assigned = np.sum(self.schedule)
        return total_assigned / total_possible if total_possible > 0 else 0.0

    def _is_terminated(self) -> bool:
        """Überprüft ob Episode beendet werden soll"""
        # Episode beenden wenn Service Level erreicht und wenig Violations
        service_level = self._get_current_service_level()
        violations = self._count_constraint_violations()

        return service_level >= 0.85 and violations <= 2

    def _get_info(self) -> Dict[str, Any]:
        """Gibt zusätzliche Informationen zurück"""
        return {
            'service_level': self._get_current_service_level(),
            'constraint_violations': self._count_constraint_violations(),
            'total_assigned_hours': self._get_total_assigned_hours(),
            'hours_compliance': self._check_hours_compliance(),
            'schedule_completeness': self._calculate_schedule_completeness(),
            'step': self.current_step,
            'total_reward': self.total_reward
        }

    def render(self, mode: str = 'human'):
        """Visualisiert den aktuellen Zustand"""
        if mode == 'human':
            print(f"\n=== Schritt {self.current_step} ===")
            print(f"Service Level: {self._get_current_service_level():.2%}")
            print(f"Constraint Violations: {self._count_constraint_violations()}")
            print(f"Assigned Hours: {self._get_total_assigned_hours():.1f}")
            print(f"Schedule Completeness: {self._calculate_schedule_completeness():.2%}")
            print(f"Total Reward: {self.total_reward:.2f}")

            # Schedule Matrix (ersten 5 Agenten und 10 Zeitslots als Beispiel)
            print("\nSchedule (erste 5 Agenten, erste 10 Zeitslots):")
            print(self.schedule[:5, :10])

    def get_final_schedule_df(self) -> pd.DataFrame:
        """Gibt finalen Schedule als DataFrame zurück"""
        schedule_data = []

        for agent_idx in range(self.num_agents):
            agent_id = self.agents.iloc[agent_idx]['csvId']
            agent_name = self.agents.iloc[agent_idx]['DisplayName']

            for time_slot in range(self.num_time_slots):
                if self.schedule[agent_idx, time_slot] == 1:
                    slot_info = self.data_loader.get_time_slot_info(time_slot)
                    schedule_data.append({
                        'Agent_ID': agent_id,
                        'Agent_Name': agent_name,
                        'Date': slot_info['date'],
                        'Time_Interval': slot_info['interval'],
                        'Time_Slot': time_slot,
                        'Expected_Contacts': slot_info['expected_contacts']
                    })

        return pd.DataFrame(schedule_data)