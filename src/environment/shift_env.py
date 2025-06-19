import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Tuple, Dict, Any

class ShiftEnvironment(gym.Env):
    """
    Eine Umgebung für Schichtplanung über eine ganze Woche.
    
    Die Umgebung verarbeitet:
    - Forecast-Daten (Personalbedarf pro Stunde für 7 Tage)
    - Mitarbeiterdaten (verfügbare Wochenstunden)
    
    Der Agent plant für jeden Tag die Schichten aller Mitarbeiter.
    Eine Episode umfasst eine komplette Woche (7 Tage).
    """
    
    def __init__(self, forecast_path, employees_path):
        super().__init__()
        
        # Lade die Daten aus CSV-Dateien
        self.forecast_df = pd.read_csv(forecast_path)
        self.employees_df = pd.read_csv(employees_path)
        
        # Speichere wichtige Dimensionen
        self.num_employees = len(self.employees_df)
        self.num_hours = 24
        self.num_days = 7  # Eine Woche
        
        # Aktionsraum: Für jeden MA eine Aktion pro Tag
        # action = start_hour * 6 + (length - 3) für 3-8 Stunden Schichten
        self.action_space = spaces.Box(
            low=0,
            high=24*6-1,
            shape=(self.num_employees,),  # Eine Aktion pro MA pro Tag
            dtype=np.int32
        )
        
        # Beobachtungsraum: [Forecast für alle 7 Tage, aktuelle Besetzung für alle 7 Tage]
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(self.num_days * self.num_hours * 2,),  # 7 Tage * 24 Stunden * 2 (Forecast + Besetzung)
            dtype=np.float32
        )
        
        # Initialisiere Zustandsvariablen
        self.current_day = 0
        # Schichtmatrix: (Mitarbeiter, Tage, Stunden)
        self.shift_matrix = np.zeros((self.num_employees, self.num_days, self.num_hours), dtype=np.int32)
        self.employee_hours_worked = np.zeros(self.num_employees, dtype=np.float32)
        
        # Speichere beste Schichtplanung
        self.best_shift_plan = None
        self.best_reward = float('-inf')
        
    def reset(self, seed=None):
        """
        Setzt die Umgebung zurück für eine neue Episode (Woche).
        Returns:
            observation: Aktuelle Beobachtung
            info: Zusätzliche Informationen (leer)
        """
        super().reset(seed=seed)
        
        # Setze alle Zustandsvariablen zurück
        self.current_day = 0
        self.shift_matrix = np.zeros((self.num_employees, self.num_days, self.num_hours), dtype=np.int32)
        self.employee_hours_worked = np.zeros(self.num_employees, dtype=np.float32)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """
        Erstellt die aktuelle Beobachtung für alle 7 Tage.
        Returns:
            observation: [Forecast für alle 7 Tage, aktuelle Besetzung für alle 7 Tage]
        """
        # Hole den Forecast für alle 7 Tage (ohne Datum-Spalte)
        forecast_data = self.forecast_df.iloc[:, 1:].values.astype(np.float32)  # Alle 7 Tage
        
        # Aktuelle Besetzung für alle 7 Tage
        current_staffing = np.sum(self.shift_matrix, axis=0)  # Summe über alle MA: (Tage, Stunden)
        
        # Kombiniere Forecast und aktuelle Besetzung
        return np.concatenate([forecast_data.flatten(), current_staffing.flatten()])
    
    def _calculate_reward(self):
        """
        Berechnet die Belohnung für die gesamte Woche.
        Returns:
            float: Gesamtbelohnung für die Woche
        """
        reward = 0
        
        # --- KRITISCHE BEDARFSDECKUNG FÜR ALLE 7 TAGE ---
        
        total_missing = 0
        total_perfect = 0
        total_worked_hours = 0
        
        for day in range(self.num_days):
            current_forecast = self.forecast_df.iloc[day, 1:].values.astype(np.float32)
            current_staffing = np.sum(self.shift_matrix[:, day, :], axis=0)  # Besetzung für diesen Tag
            
            # Bedarfsdeckung für diesen Tag
            for hour in range(self.num_hours):
                if current_forecast[hour] > 0:
                    if current_staffing[hour] < current_forecast[hour]:
                        missing_staff = current_forecast[hour] - current_staffing[hour]
                        reward -= 500 * missing_staff
                        total_missing += missing_staff
                    elif current_staffing[hour] == current_forecast[hour]:
                        reward += 200
                        total_perfect += 1
                    elif current_staffing[hour] > current_forecast[hour] * 1.3:
                        excess_staff = current_staffing[hour] - current_forecast[hour] * 1.3
                        reward -= 100 * excess_staff
            
            total_worked_hours += np.sum(current_staffing)
        
        # --- MASSIVE Belohnung für hohe Gesamtstunden (näher am Wochenbedarf) ---
        forecast_total = np.sum(self.forecast_df.iloc[:, 1:].values)
        
        if total_worked_hours >= forecast_total * 0.9:  # Mindestens 90% des Wochenbedarfs
            reward += 2000  # MASSIVE Belohnung
        elif total_worked_hours >= forecast_total * 0.8:  # Mindestens 80% des Wochenbedarfs
            reward += 1000
        elif total_worked_hours >= forecast_total * 0.6:  # Mindestens 60% des Wochenbedarfs
            reward += 500
        elif total_worked_hours < forecast_total * 0.3:  # Weniger als 30% = MASSIVE Strafe
            reward -= 5000
        
        # --- Öffnungszeiten für alle 7 Tage ---
        for day in range(self.num_days):
            current_staffing = np.sum(self.shift_matrix[:, day, :], axis=0)
            outside_hours = np.sum(current_staffing[:7]) + np.sum(current_staffing[22:])
            if outside_hours > 0:
                reward -= 300 * outside_hours
        
        # --- Schichtlängen für alle 7 Tage ---
        for emp_idx in range(self.num_employees):
            for day in range(self.num_days):
                shift_length = np.sum(self.shift_matrix[emp_idx, day, :])
                if shift_length > 0:  # Wenn MA arbeitet
                    if shift_length < 3 or shift_length > 8:
                        reward -= 200
                    elif shift_length == 6:  # Optimale Länge
                        reward += 30
        
        # --- Vertragsstunden: Strikte 20%-Regel mit verstärkter Strafe ---
        for emp_idx in range(self.num_employees):
            weekly_hours = self.employees_df.iloc[emp_idx]['weekly_hours']
            worked_hours = self.employee_hours_worked[emp_idx]
            diff = abs(worked_hours - weekly_hours)
            max_dev = weekly_hours * 0.2
            
            if diff <= max_dev:
                # Moderate lineare Strafe
                reward -= 500 * diff
                # Belohnung für perfekte Einhaltung
                if diff == 0:
                    reward += 3000
                elif diff <= 2:
                    reward += 1500 * (1 - diff/2)
            else:
                # Sehr hohe Strafe für Überschreitung der 20%-Grenze
                reward -= 10000 * (diff - max_dev)
        
        # --- Forecast-Abweichung pro Tag: exponentielle Strafe/Belohnung ---
        for day in range(self.num_days):
            current_forecast = self.forecast_df.iloc[day, 1:].values.astype(np.float32)
            current_staffing = np.sum(self.shift_matrix[:, day, :], axis=0)
            forecast_total = np.sum(current_forecast)
            staffing_total = np.sum(current_staffing)
            diff = abs(staffing_total - forecast_total)
            # Exponentielle Strafe für Abweichung
            reward -= 200 * (diff ** 2)
            # Exponentielle Belohnung für Zielnähe
            reward += 300 * np.exp(-0.2 * diff)
        
        # --- Arbeitsanreiz: Höherer Reward für gearbeitete Stunden ---
        reward += total_worked_hours * 5  # +5 pro gearbeiteter Stunde
        
        # --- Belohnung für zeitliche Verteilung über die Woche ---
        if total_worked_hours > 0:
            # Berechne Varianz der Besetzung über alle Tage und Stunden
            all_staffing = np.sum(self.shift_matrix, axis=0)  # (Tage, Stunden)
            variance = np.var(all_staffing)
            if variance > 2:  # Höhere Varianz = bessere Verteilung
                reward += 200  # Belohnung für Verteilung
            else:
                reward -= 400  # Strafe für zu gleichmäßige Verteilung
        
        # --- Zusätzliche Belohnung für Bedarfsdeckung ---
        if total_missing == 0 and total_perfect > 0:
            reward += 1000  # Bonus wenn alle Bedarfe gedeckt sind
        
        return float(reward)
    
    def step(self, action):
        """
        Führt einen Schritt in der Umgebung aus (plant einen Tag).
        
        Args:
            action: Array mit Aktionen für jeden MA für den aktuellen Tag
            
        Returns:
            observation: Neue Beobachtung
            reward: Belohnung für diese Aktion
            done: Ob Episode beendet ist (nach 7 Tagen)
            truncated: Immer False
            info: Zusätzliche Informationen (leer)
        """
        # Setze Schichten für den aktuellen Tag zurück
        self.shift_matrix[:, self.current_day, :] = 0
        
        # Verarbeite die Aktionen für jeden MA für den aktuellen Tag
        for emp_idx, act in enumerate(action):
            # Berechne Startzeit und Länge aus der Aktion
            start_hour = int(act) // 6
            shift_length = (int(act) % 6) + 3  # 3-8 Stunden
            
            # Berechne alle Stunden der Schicht
            shift_hours = np.arange(start_hour, start_hour + shift_length) % 24
            # Setze Schicht für diesen MA an diesem Tag
            self.shift_matrix[emp_idx, self.current_day, shift_hours] = 1
            # Aktualisiere gearbeitete Stunden
            self.employee_hours_worked[emp_idx] += shift_length
        
        # Berechne Reward für die gesamte Woche
        reward = self._calculate_reward()
        
        # Prüfe ob Episode beendet (Ende der Woche)
        done = self.current_day >= self.num_days - 1
        
        # Gehe zum nächsten Tag wenn nicht fertig
        if not done:
            self.current_day += 1
        
        # Update beste Schichtplanung
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_shift_plan = self.shift_matrix.copy()
        
        return self._get_observation(), reward, done, False, {}
    
    def export_best_shift_plan(self, filename: str):
        """
        Exportiert die beste Schichtplanung für die ganze Woche als CSV.
        
        Args:
            filename: Pfad zur CSV-Datei
        """
        if self.best_shift_plan is None:
            print("Keine Schichtplanung zum Exportieren vorhanden.")
            return
        
        # Erstelle DataFrame mit Tagen und Stunden als Spalten
        columns = ['Mitarbeiter_ID', 'Name']
        for day in range(self.num_days):
            for hour in range(self.num_hours):
                columns.append(f"Tag_{day+1}_Stunde_{hour:02d}")
        
        shift_data = []
        for emp_idx in range(self.num_employees):
            row = [self.employees_df.iloc[emp_idx]['employee_id'], f"MA{emp_idx+1}"]
            # Füge Schichtdaten für alle 7 Tage hinzu
            for day in range(self.num_days):
                for hour in range(self.num_hours):
                    row.append(self.best_shift_plan[emp_idx, day, hour])
            shift_data.append(row)
        
        df = pd.DataFrame(shift_data, columns=columns)
        
        # Füge Zusammenfassung hinzu
        hour_columns = [col for col in columns if col.startswith('Tag_')]
        df['Gesamtstunden'] = df[hour_columns].sum(axis=1)
        
        # Speichere CSV
        df.to_csv(filename, index=False)
        print(f"Beste Schichtplanung für die Woche gespeichert in: {filename}")
        print(f"Bester Reward: {self.best_reward:.2f}")
        
        # Zeige Zusammenfassung
        print("\nZusammenfassung der besten Schichtplanung:")
        print(f"Gesamtstunden: {df['Gesamtstunden'].sum()}")
        print(f"Durchschnittliche Stunden pro MA: {df['Gesamtstunden'].mean():.1f}")
        print(f"Anzahl MA mit Schichten: {(df['Gesamtstunden'] > 0).sum()}")
        
        # Zeige Bedarfsdeckung für jeden Tag
        print(f"\nBedarfsdeckung pro Tag:")
        for day in range(self.num_days):
            forecast = self.forecast_df.iloc[day, 1:].values
            total_staffing = np.sum(self.best_shift_plan[:, day, :], axis=0)
            forecast_total = np.sum(forecast)
            staffing_total = np.sum(total_staffing)
            print(f"  Tag {day+1}: Bedarf {forecast_total}, Besetzung {staffing_total}, Deckung {staffing_total/forecast_total*100:.1f}%") 