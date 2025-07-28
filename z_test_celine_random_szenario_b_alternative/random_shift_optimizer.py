"""
Einfacher Random Agent für Employee Shift Scheduling - Untere Benchmark
Sehr einfache zufällige Zuweisung mit minimalen Constraints
Monte-Carlo-Simulation mit 1.000 Stichproben für den besten zufälligen Plan
"""

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path
from copy import deepcopy

class SimpleRandomShiftOptimizer:
    def __init__(self, forecast_file, employees_file):
        """
        Initialize the random shift optimizer
        
        Args:
            forecast_file: Path to forecast.csv
            employees_file: Path to employees.csv
        """
        self.forecast_df = pd.read_csv(forecast_file)
        self.employees_df = pd.read_csv(employees_file)
        
        # Extract basic information
        self.days = self.forecast_df['weekday'].tolist()
        self.shifts = ['shift_morning', 'shift_afternoon', 'shift_evening']
        self.employees = self.employees_df['employee_id'].tolist()
        
        # Create employee shifts dictionary
        self.employee_shifts = dict(zip(
            self.employees_df['employee_id'], 
            self.employees_df['shifts']
        ))
        
        # Create demand dictionary
        self.demand = {}
        for _, row in self.forecast_df.iterrows():
            day = row['weekday']
            for shift in self.shifts:
                self.demand[(day, shift)] = row[shift]
        
        # Set random seed for reproducibility of the best solution
        random.seed(42)
        np.random.seed(42)
    
    def create_empty_schedule(self):
        """Create empty schedule structure"""
        schedule = {}
        for emp in self.employees:
            schedule[emp] = {}
            for day in self.days:
                schedule[emp][day] = {shift: 0 for shift in self.shifts}
        return schedule
    
    def is_valid_assignment(self, schedule, employee, day, shift):
        """
        Nur minimale Constraints prüfen - sehr einfach!
        """
        # 1. Maximal 2 Schichten pro Tag
        daily_shifts = sum(schedule[employee][day][s] for s in self.shifts)
        if daily_shifts >= 2:
            return False
        
        # 2. Nicht mehr als die Vertragsstunden
        total_shifts = sum(
            sum(schedule[employee][d][s] for s in self.shifts) 
            for d in self.days
        )
        if total_shifts >= self.employee_shifts[employee]:
            return False
        
        # Das war's! Keine konsekutiven Constraints - einfach random!
        return True
    
    def calculate_fitness(self, schedule):
        """
        Einfacher Fitness-Score - nur grundlegende Metriken
        """
        score = 0
        
        # 1. Demand coverage (wichtigste Metrik)
        for day in self.days:
            for shift in self.shifts:
                required = self.demand[(day, shift)]
                assigned = sum(schedule[emp][day][shift] for emp in self.employees)
                
                # Einfache Bewertung: +10 für jede erfüllte Schicht
                covered = min(assigned, required)
                score += covered * 10
                
                # -5 für jede fehlende Schicht
                if assigned < required:
                    score -= (required - assigned) * 5
        
        # 2. Employee satisfaction (einfach)
        for emp in self.employees:
            required = self.employee_shifts[emp]
            assigned = sum(
                sum(schedule[emp][d][s] for s in self.shifts) 
                for d in self.days
            )
            
            # +5 für jede zugewiesene Schicht
            score += assigned * 5
            
            # Kleiner Bonus wenn exakt erfüllt
            if assigned == required:
                score += 20
        
        return score
    
    def generate_random_schedule(self):
        """
        Sehr einfache zufällige Schichtgenerierung
        Kein cleveres Assignment - einfach probieren!
        """
        schedule = self.create_empty_schedule()
        
        # Einfacher Ansatz: Für jeden Mitarbeiter zufällige Schichten zuweisen
        for emp in self.employees:
            required_shifts = self.employee_shifts[emp]
            assigned_shifts = 0
            attempts = 0
            max_attempts = 100  # Verhindert Endlosschleifen
            
            while assigned_shifts < required_shifts and attempts < max_attempts:
                attempts += 1
                
                # Zufälligen Tag und Schicht wählen
                random_day = random.choice(self.days)
                random_shift = random.choice(self.shifts)
                
                # Versuchen zuzuweisen (nur mit minimalen Constraints)
                if self.is_valid_assignment(schedule, emp, random_day, random_shift):
                    schedule[emp][random_day][random_shift] = 1
                    assigned_shifts += 1
        
        return schedule
    
    def monte_carlo_optimization(self, num_samples=1000):
        """
        Run Monte Carlo simulation to find best random schedule
        """
        print(f"🎲 Starting Monte Carlo simulation with {num_samples} samples...")
        
        best_schedule = None
        best_fitness = -float('inf')
        fitness_scores = []
        
        valid_solutions = 0
        
        for i in range(num_samples):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{num_samples} samples completed")
            
            # Generate random schedule
            schedule = self.generate_random_schedule()
            fitness = self.calculate_fitness(schedule)
            fitness_scores.append(fitness)
            
            # Check if any assignments were made
            total_assignments = sum(
                sum(sum(schedule[emp][day][s] for s in self.shifts) for day in self.days)
                for emp in self.employees
            )
            
            if total_assignments > 0:
                valid_solutions += 1
            
            # Update best solution
            if fitness > best_fitness:
                best_fitness = fitness
                best_schedule = deepcopy(schedule)
        
        print(f"✅ Monte Carlo completed!")
        print(f"   Valid solutions found: {valid_solutions}/{num_samples}")
        print(f"   Best fitness score: {best_fitness:.2f}")
        print(f"   Average fitness: {np.mean(fitness_scores):.2f}")
        print(f"   Fitness std dev: {np.std(fitness_scores):.2f}")
        
        return best_schedule, best_fitness, fitness_scores
    
    def save_employee_schedules(self, schedule, output_dir):
        """
        Save individual employee schedules to CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for employee in self.employees:
            # Create schedule data for this employee
            schedule_data = []
            for day in self.days:
                row = {'weekday': day}
                for shift in self.shifts:
                    row[shift] = schedule[employee][day][shift]
                schedule_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(schedule_data)
            filename = f"{employee}_schedule.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Created schedule for {employee}: {filepath}")
        
        print(f"✅ Employee schedules saved to: {output_dir}")
    
    def print_solution_summary(self, schedule):
        """
        Print summary of the simple random solution
        """
        print("\n=== EINFACHER RANDOM AGENT - BENCHMARK LÖSUNG ===")
        
        # Employee summary
        print("\nMitarbeiter-Zusammenfassung:")
        total_coverage_rate = 0
        for emp in self.employees:
            required = self.employee_shifts[emp]
            assigned = sum(
                sum(schedule[emp][day][s] for s in self.shifts) 
                for day in self.days
            )
            coverage = (assigned / required * 100) if required > 0 else 0
            total_coverage_rate += coverage
            status = "✅" if assigned == required else "⚠️"
            print(f"{status} {emp}: {assigned}/{required} Schichten ({coverage:.1f}%)")
        
        avg_coverage = total_coverage_rate / len(self.employees)
        print(f"\n📊 Durchschnittliche Mitarbeiter-Abdeckung: {avg_coverage:.1f}%")
        
        # Demand coverage
        print("\nTägliche Schichtabdeckung:")
        total_demand = 0
        total_assigned = 0
        
        for day in self.days:
            print(f"\n{day.upper()}:")
            for shift in self.shifts:
                required = self.demand[(day, shift)]
                assigned_employees = [
                    emp for emp in self.employees 
                    if schedule[emp][day][shift] == 1
                ]
                assigned = len(assigned_employees)
                total_demand += required
                total_assigned += assigned
                
                coverage = (assigned / required * 100) if required > 0 else 100
                status = "✅" if assigned >= required else "❌"
                print(f"  {status} {shift}: {assigned}/{required} ({coverage:.0f}%) - {assigned_employees}")
        
        demand_coverage = (total_assigned / total_demand * 100) if total_demand > 0 else 0
        print(f"\n📈 Gesamte Bedarfsabdeckung: {total_assigned}/{total_demand} ({demand_coverage:.1f}%)")


def main():
    # File paths
    base_dir = Path(__file__).parent.parent
    forecast_file = base_dir / "data" / "forecast_szenario_b_alternative.csv"
    employees_file = base_dir / "data" / "employees.csv"
    output_dir = Path(__file__).parent / "employee_schedules"
    
    # Create optimizer
    optimizer = SimpleRandomShiftOptimizer(forecast_file, employees_file)
    
    print("🎲 EINFACHER Random Agent - Untere Benchmark")
    print("=" * 50)
    print("⚠️ Hinweis: Sehr einfache zufällige Zuweisung ohne clevere Optimierung!")
    
    # Run Monte Carlo optimization
    best_schedule, best_fitness, fitness_scores = optimizer.monte_carlo_optimization(1000)
    
    if best_schedule is None:
        print("❌ Keine gültige Lösung gefunden!")
        return
    
    # Print solution summary
    optimizer.print_solution_summary(best_schedule)
    
    # Save employee schedules
    optimizer.save_employee_schedules(best_schedule, output_dir)
    
    print(f"\n🎯 Einfacher Benchmark Score: {best_fitness:.2f}")
    print("📊 Dies ist die UNTERE Benchmark für Algorithmus-Vergleiche.")
    print("💡 Andere Algorithmen sollten deutlich besser abschneiden!")


if __name__ == "__main__":
    main()
