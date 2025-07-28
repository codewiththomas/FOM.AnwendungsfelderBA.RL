"""
Linear Optimization for Under-Demand Employee Shift Scheduling
Scenario: More capacity than demand - need to fairly distribute limited work
"""

import pandas as pd
import pulp
import os
from pathlib import Path

class UnderDemandLinearOptimizer:
    def __init__(self, forecast_file, employees_file):
        self.forecast_df = pd.read_csv(forecast_file)
        self.employees_df = pd.read_csv(employees_file)
        
        # Extract basic information
        self.days = self.forecast_df['weekday'].tolist()
        self.shifts = ['shift_morning', 'shift_afternoon', 'shift_evening']
        self.employees = self.employees_df['employee_id'].tolist()
        
        # Create employee capacity dictionary
        self.employee_capacity = dict(zip(
            self.employees_df['employee_id'], 
            self.employees_df['shifts']
        ))
        
        # Create demand dictionary
        self.demand = {}
        for _, row in self.forecast_df.iterrows():
            day = row['weekday']
            for shift in self.shifts:
                self.demand[(day, shift)] = row[shift]
        
        print(f"Total demand: {sum(self.demand.values())} shifts")
        print(f"Total capacity: {sum(self.employee_capacity.values())} shifts")
    
    def solve_scheduling(self):
        # Create the problem - maximize fairness in under-demand scenario
        prob = pulp.LpProblem("UnderDemandScheduling", pulp.LpMaximize)
        
        # Decision variables: x[employee, day, shift] = 1 if employee works that shift
        x = {}
        for emp in self.employees:
            for day in self.days:
                for shift in self.shifts:
                    x[emp, day, shift] = pulp.LpVariable(
                        f"x_{emp}_{day}_{shift}", 
                        cat='Binary'
                    )
        
        # Auxiliary variables for fairness - track how many shifts each employee gets
        employee_total_shifts = {}
        for emp in self.employees:
            employee_total_shifts[emp] = pulp.lpSum([
                x[emp, day, shift] 
                for day in self.days 
                for shift in self.shifts
            ])
        
        # Objective: maximize minimum shifts assigned (fairness)
        min_shifts = pulp.LpVariable("min_shifts", lowBound=0)
        for emp in self.employees:
            prob += employee_total_shifts[emp] >= min_shifts
        
        prob += min_shifts
        
        # Constraint 1: Meet exact shift demand for each day and shift type
        for day in self.days:
            for shift in self.shifts:
                prob += pulp.lpSum([
                    x[emp, day, shift] for emp in self.employees
                ]) == self.demand[(day, shift)]
        
        # Constraint 2: Each employee cannot exceed their capacity
        for emp in self.employees:
            prob += employee_total_shifts[emp] <= self.employee_capacity[emp]
        
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
            raise Exception(f"No optimal solution found. Status: {pulp.LpStatus[prob.status]}")
        
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
    
    def create_metrics_summary(self, output_dir):
        """Create employee metrics CSV for comparison analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        metrics_data = []
        
        for emp in self.employees:
            total_shifts = 0
            if emp in self.solution:
                for day_shifts in self.solution[emp].values():
                    total_shifts += len(day_shifts)
            
            capacity = self.employee_capacity[emp]
            utilization = total_shifts / capacity if capacity > 0 else 0
            unused_capacity = capacity - total_shifts
            
            metrics_data.append({
                'employee_id': emp,
                'total_shifts_assigned': total_shifts,
                'capacity': capacity,
                'utilization_rate': round(utilization, 3),
                'unused_capacity': unused_capacity
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = os.path.join(output_dir, 'employee_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Employee metrics saved to: {metrics_file}")
        
        return metrics_df
    
    def print_solution_summary(self):
        print("\n=== UNDER-DEMAND SCHEDULING SOLUTION ===")
        print("\nEmployee Work Summary:")
        
        total_assigned = 0
        total_capacity = 0
        
        for emp in self.employees:
            shifts_assigned = 0
            if emp in self.solution:
                for day_shifts in self.solution[emp].values():
                    shifts_assigned += len(day_shifts)
            
            capacity = self.employee_capacity[emp]
            utilization = shifts_assigned / capacity if capacity > 0 else 0
            unused = capacity - shifts_assigned
            
            total_assigned += shifts_assigned
            total_capacity += capacity
            
            print(f"{emp}: {shifts_assigned}/{capacity} shifts ({utilization:.1%} utilization, {unused} unused)")
        
        print(f"\nOverall Summary:")
        print(f"Total shifts assigned: {total_assigned}")
        print(f"Total capacity: {total_capacity}")
        print(f"Overall utilization: {total_assigned/total_capacity:.1%}")
        print(f"Unused capacity: {total_capacity - total_assigned} shifts")
        
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
        
        # Fairness analysis
        shift_counts = []
        for emp in self.employees:
            count = 0
            if emp in self.solution:
                for day_shifts in self.solution[emp].values():
                    count += len(day_shifts)
            shift_counts.append(count)
        
        if shift_counts:
            import statistics
            print(f"\nFairness Metrics:")
            print(f"Min shifts assigned: {min(shift_counts)}")
            print(f"Max shifts assigned: {max(shift_counts)}")
            print(f"Average shifts: {statistics.mean(shift_counts):.1f}")
            print(f"Standard deviation: {statistics.stdev(shift_counts) if len(shift_counts) > 1 else 0:.2f}")


def main():
    # File paths
    base_dir = Path(__file__).parent.parent
    forecast_file = base_dir / "data" / "forecast_szenario_c.csv"
    employees_file = base_dir / "data" / "employees.csv"
    output_dir = Path(__file__).parent / "employee_schedules"
    
    # Create optimizer
    optimizer = UnderDemandLinearOptimizer(forecast_file, employees_file)
    
    print("Starting under-demand linear optimization...")
    
    # Solve the problem
    try:
        solution = optimizer.solve_scheduling()
        print("✅ Optimization completed successfully!")
        
        # Print solution summary
        optimizer.print_solution_summary()
        
        # Create individual employee schedules
        optimizer.create_employee_schedules(output_dir)
        
        # Create metrics summary for comparison
        optimizer.create_metrics_summary(output_dir)
        
        print(f"\n✅ Employee schedules saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")


if __name__ == "__main__":
    main()
