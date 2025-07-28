"""
Simplified Linear Optimization for Employee Shift Scheduling
"""

import pandas as pd
import pulp
import os
from pathlib import Path

class SimpleShiftOptimizer:
    def __init__(self, forecast_file, employees_file):
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
    
    def solve_scheduling(self):
        # Create the problem
        prob = pulp.LpProblem("SimpleShiftScheduling", pulp.LpMinimize)
        
        # Decision variables: x[employee, day, shift] = 1 if employee works that shift
        x = {}
        for emp in self.employees:
            for day in self.days:
                for shift in self.shifts:
                    x[emp, day, shift] = pulp.LpVariable(
                        f"x_{emp}_{day}_{shift}", 
                        cat='Binary'
                    )
        
        # Simple objective: minimize total assignments (will be satisfied by constraints)
        prob += pulp.lpSum([x[emp, day, shift] for emp in self.employees 
                           for day in self.days for shift in self.shifts])
        
        # Constraint 1: Meet shift demand for each day and shift type
        for day in self.days:
            for shift in self.shifts:
                prob += pulp.lpSum([
                    x[emp, day, shift] for emp in self.employees
                ]) == self.demand[(day, shift)]
        
        # Constraint 2: Each employee works exactly their assigned number of shifts
        for emp in self.employees:
            prob += pulp.lpSum([
                x[emp, day, shift] 
                for day in self.days 
                for shift in self.shifts
            ]) == self.employee_shifts[emp]
        
        # Constraint 3: Each employee can work at most 2 consecutive shifts per day
        for emp in self.employees:
            for day in self.days:
                # Max 2 shifts per day
                prob += pulp.lpSum([
                    x[emp, day, shift] for shift in self.shifts
                ]) <= 2
                
                # Consecutive shifts constraint: cannot work morning and evening without afternoon
                prob += x[emp, day, 'shift_morning'] + x[emp, day, 'shift_evening'] - x[emp, day, 'shift_afternoon'] <= 1
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Check if solution was found
        if prob.status != pulp.LpStatusOptimal:
            print(f"Problem status: {pulp.LpStatus[prob.status]}")
            
            # Try with relaxed constraints (allow slight deviation in employee shifts)
            print("Trying with relaxed constraints...")
            return self.solve_with_relaxed_constraints()
        
        # Extract solution
        self.solution = {}
        for emp in self.employees:
            for day in self.days:
                for shift in self.shifts:
                    if x[emp, day, shift].varValue == 1:
                        if emp not in self.solution:
                            self.solution[emp] = {}
                        if day not in self.solution[emp]:
                            self.solution[emp][day] = []
                        self.solution[emp][day].append(shift)
        
        return self.solution
    
    def solve_with_relaxed_constraints(self):
        # Create a new problem with relaxed constraints
        prob = pulp.LpProblem("RelaxedShiftScheduling", pulp.LpMinimize)
        
        # Decision variables
        x = {}
        for emp in self.employees:
            for day in self.days:
                for shift in self.shifts:
                    x[emp, day, shift] = pulp.LpVariable(
                        f"x_{emp}_{day}_{shift}", 
                        cat='Binary'
                    )
        
        # Deviation variables for employee shifts
        dev_pos = {}
        dev_neg = {}
        for emp in self.employees:
            dev_pos[emp] = pulp.LpVariable(f"dev_pos_{emp}", lowBound=0)
            dev_neg[emp] = pulp.LpVariable(f"dev_neg_{emp}", lowBound=0)
        
        # Objective: minimize deviations from required shifts
        prob += pulp.lpSum([dev_pos[emp] + dev_neg[emp] for emp in self.employees])
        
        # Constraint 1: Meet shift demand (this is non-negotiable)
        for day in self.days:
            for shift in self.shifts:
                prob += pulp.lpSum([
                    x[emp, day, shift] for emp in self.employees
                ]) == self.demand[(day, shift)]
        
        # Constraint 2: Employee shifts with deviation tracking
        for emp in self.employees:
            total_shifts = pulp.lpSum([
                x[emp, day, shift] 
                for day in self.days 
                for shift in self.shifts
            ])
            prob += total_shifts - self.employee_shifts[emp] == dev_pos[emp] - dev_neg[emp]
        
        # Constraint 3: Each employee can work at most 2 consecutive shifts per day
        for emp in self.employees:
            for day in self.days:
                # Max 2 shifts per day
                prob += pulp.lpSum([
                    x[emp, day, shift] for shift in self.shifts
                ]) <= 2
                
                # Consecutive shifts constraint: cannot work morning and evening without afternoon
                prob += x[emp, day, 'shift_morning'] + x[emp, day, 'shift_evening'] - x[emp, day, 'shift_afternoon'] <= 1
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            raise Exception(f"No solution found even with relaxed constraints. Status: {pulp.LpStatus[prob.status]}")
        
        # Extract solution
        self.solution = {}
        for emp in self.employees:
            for day in self.days:
                for shift in self.shifts:
                    if x[emp, day, shift].varValue == 1:
                        if emp not in self.solution:
                            self.solution[emp] = {}
                        if day not in self.solution[emp]:
                            self.solution[emp][day] = []
                        self.solution[emp][day].append(shift)
        
        return self.solution
    
    def create_employee_schedules(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        for emp in self.employees:
            # Create a schedule DataFrame for this employee
            schedule_data = []
            
            for day in self.days:
                row = {'weekday': day}
                
                # Initialize all shifts to 0
                for shift in self.shifts:
                    row[shift] = 0
                
                # Set shifts to 1 if employee is assigned
                if emp in self.solution and day in self.solution[emp]:
                    for shift in self.solution[emp][day]:
                        row[shift] = 1
                
                schedule_data.append(row)
            
            # Create DataFrame and save to CSV
            schedule_df = pd.DataFrame(schedule_data)
            filename = f"{emp}_schedule.csv"
            filepath = os.path.join(output_dir, filename)
            schedule_df.to_csv(filepath, index=False)
            
            print(f"Created schedule for {emp}: {filepath}")
    
    def print_solution_summary(self):
        print("\n=== SHIFT SCHEDULING SOLUTION ===")
        print("\nEmployee Work Summary:")
        
        for emp in self.employees:
            total_shifts = 0
            if emp in self.solution:
                for day_shifts in self.solution[emp].values():
                    total_shifts += len(day_shifts)
            
            required = self.employee_shifts[emp]
            status = "✅" if total_shifts == required else "⚠️"
            print(f"{status} {emp}: {total_shifts} shifts assigned (required: {required})")
        
        print("\nDaily Shift Coverage:")
        for day in self.days:
            print(f"\n{day.upper()}:")
            for shift in self.shifts:
                assigned_employees = []
                for emp in self.employees:
                    if (emp in self.solution and 
                        day in self.solution[emp] and 
                        shift in self.solution[emp][day]):
                        assigned_employees.append(emp)
                
                required = self.demand[(day, shift)]
                assigned = len(assigned_employees)
                status = "✅" if assigned == required else "❌"
                print(f"  {status} {shift}: {assigned}/{required} - {assigned_employees}")


def main():
    # File paths
    base_dir = Path(__file__).parent.parent
    forecast_file = base_dir / "data" / "forecast_szenario_a.csv"
    employees_file = base_dir / "data" / "employees.csv"
    output_dir = Path(__file__).parent / "employee_schedules"
    
    # Create optimizer
    optimizer = SimpleShiftOptimizer(forecast_file, employees_file)
    
    print("Starting simplified shift optimization...")
    
    # Solve the problem
    try:
        solution = optimizer.solve_scheduling()
        print("✅ Optimization completed successfully!")
        
        # Print solution summary
        optimizer.print_solution_summary()
        
        # Create individual employee schedules
        optimizer.create_employee_schedules(output_dir)
        
        print(f"\n✅ Employee schedules saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")


if __name__ == "__main__":
    main()
