"""
Data Loader für Dienstplanung RL Environment
Lädt und verarbeitet die CSV-Dateien für das Training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import os

class DataLoader:
    """Lädt und verarbeitet alle CSV-Dateien für das Dienstplanungs-Environment"""

    def __init__(self, data_path: str = "data_csv"):
        self.data_path = data_path
        self.agents_df = None
        self.lines_df = None
        self.forecast_df = None
        self.constraints_df = None

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Lädt alle CSV-Dateien und gibt sie als Dictionary zurück"""
        try:
            # Agents laden
            self.agents_df = pd.read_csv(os.path.join(self.data_path, "agents.csv"))
            print(f"Agents geladen: {len(self.agents_df)} Agenten")

            # Lines laden
            self.lines_df = pd.read_csv(os.path.join(self.data_path, "lines.csv"))
            print(f"Lines geladen: {len(self.lines_df)} Lines")

            # Forecast laden
            self.forecast_df = pd.read_csv(os.path.join(self.data_path, "forecast.csv"))
            self.forecast_df['csvDatum'] = pd.to_datetime(self.forecast_df['csvDatum'])
            print(f"Forecast geladen: {len(self.forecast_df)} Zeitslots")

            # Constraints laden
            self.constraints_df = pd.read_csv(os.path.join(self.data_path, "constraints.csv"))
            print(f"Constraints geladen: {len(self.constraints_df)} Constraints")

            return {
                'agents': self.agents_df,
                'lines': self.lines_df,
                'forecast': self.forecast_df,
                'constraints': self.constraints_df
            }

        except Exception as e:
            print(f"Fehler beim Laden der Daten: {e}")
            raise

    def preprocess_forecast(self) -> pd.DataFrame:
        """Verarbeitet Forecast-Daten für einfachere Verwendung"""
        if self.forecast_df is None:
            self.load_all_data()

        # Zeitslot-Index hinzufügen
        forecast_processed = self.forecast_df.copy()
        forecast_processed['time_slot'] = range(len(forecast_processed))

        # Zeitslot in Stunden umrechnen (30-Minuten-Intervalle)
        forecast_processed['hour_of_day'] = forecast_processed['Interval'].apply(
            lambda x: float(x.split('-')[0].split(':')[0]) + float(x.split('-')[0].split(':')[1])/60
        )

        # Wochentag hinzufügen
        forecast_processed['weekday'] = forecast_processed['csvDatum'].dt.dayofweek

        return forecast_processed

    def get_agent_qualifications(self) -> Dict[str, List[str]]:
        """Gibt Qualifikationen der Agenten zurück"""
        if self.agents_df is None:
            self.load_all_data()

        qualifications = {}
        for _, agent in self.agents_df.iterrows():
            agent_quals = []
            if agent['Quali_Line1']:
                agent_quals.append('L001')
            if agent['Quali_Line2']:
                agent_quals.append('L002')  # Falls weitere Lines hinzugefügt werden
            if agent['Quali_Line3']:
                agent_quals.append('L003')  # Falls weitere Lines hinzugefügt werden

            qualifications[agent['csvId']] = agent_quals

        return qualifications

    def get_constraint_weights(self) -> Dict[str, float]:
        """Gibt Constraint-Gewichte zurück"""
        if self.constraints_df is None:
            self.load_all_data()

        weights = {}
        for _, constraint in self.constraints_df.iterrows():
            weights[constraint['csvID']] = constraint['Gewicht'] * constraint['IsPositiv']

        return weights

    def calculate_required_agents(self, time_slot: int, aht: float = 8.5) -> int:
        """Berechnet benötigte Agenten für einen Zeitslot basierend auf Forecast"""
        if self.forecast_df is None:
            self.load_all_data()

        if time_slot < len(self.forecast_df):
            expected_contacts = self.forecast_df.iloc[time_slot]['Erwartete_Kontakte']
            # Berechnung: (Erwartete Kontakte * AHT in Minuten) / 30 Minuten Zeitslot
            required_agents = np.ceil((expected_contacts * aht) / 30)
            return max(1, int(required_agents))  # Mindestens 1 Agent
        return 1

    def get_time_slot_info(self, time_slot: int) -> Dict:
        """Gibt Informationen zu einem spezifischen Zeitslot zurück"""
        if self.forecast_df is None:
            self.load_all_data()

        if time_slot < len(self.forecast_df):
            row = self.forecast_df.iloc[time_slot]
            return {
                'date': row['csvDatum'],
                'interval': row['Interval'],
                'expected_contacts': row['Erwartete_Kontakte'],
                'required_agents': self.calculate_required_agents(time_slot),
                'hour_of_day': float(row['Interval'].split('-')[0].split(':')[0]),
                'weekday': row['csvDatum'].weekday()
            }
        return {}

    def get_summary_stats(self) -> Dict:
        """Gibt Zusammenfassungsstatistiken zurück"""
        if any(df is None for df in [self.agents_df, self.forecast_df, self.constraints_df]):
            self.load_all_data()

        return {
            'total_agents': len(self.agents_df),
            'total_time_slots': len(self.forecast_df),
            'total_constraints': len(self.constraints_df),
            'forecast_period_days': (self.forecast_df['csvDatum'].max() - self.forecast_df['csvDatum'].min()).days + 1,
            'avg_contacts_per_slot': self.forecast_df['Erwartete_Kontakte'].mean(),
            'max_contacts_per_slot': self.forecast_df['Erwartete_Kontakte'].max(),
            'min_contacts_per_slot': self.forecast_df['Erwartete_Kontakte'].min(),
            'total_weekly_hours': self.agents_df['Wochenstunden'].sum()
        }