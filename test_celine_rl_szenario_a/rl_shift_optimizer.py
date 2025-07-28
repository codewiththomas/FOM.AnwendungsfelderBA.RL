"""
Reinforcement Learning Agent for Equal-Demand Employee Shift Scheduling
Scenario: Demand equals capacity - optimal resource utilization

Key RL Advantages in Equal-Demand:
1. Optimal resource allocation (no waste, no shortage)
2. Constraint satisfaction with perfect utilization
3. Learning best assignment patterns
4. Balanced workload distribution
"""

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path
from copy import deepcopy
import json
import statistics

class EqualDemandRLAgent:
    def __init__(self, forecast_file, employees_file, learning_rate=0.1, epsilon=0.3):
        """
        Initialize RL Agent for equal-demand shift scheduling
        """
        self.forecast_df = pd.read_csv(forecast_file)
        self.employees_df = pd.read_csv(employees_file)
        
        # Basic info
        self.days = self.forecast_df['weekday'].tolist()
        self.shifts = ['shift_morning', 'shift_afternoon', 'shift_evening']
        self.employees = self.employees_df['employee_id'].tolist()
        
        # Employee capacity
        self.employee_capacity = dict(zip(
            self.employees_df['employee_id'], 
            self.employees_df['shifts']
        ))
        
        # Demand requirements
        self.demand = {}
        total_demand = 0
        for _, row in self.forecast_df.iterrows():
            day = row['weekday']
            for shift in self.shifts:
                self.demand[(day, shift)] = row[shift]
                total_demand += row[shift]
        
        total_capacity = sum(self.employee_capacity.values())
        print(f"Equal-demand scenario: {total_demand} demand vs {total_capacity} capacity")
        
        # Valid daily shift patterns (excluding M+E without A)
        self.valid_daily_patterns = [
            [0, 0, 0],  # No shifts
            [1, 0, 0],  # Morning only
            [0, 1, 0],  # Afternoon only  
            [0, 0, 1],  # Evening only
            [1, 1, 0],  # Morning + Afternoon
            [0, 1, 1],  # Afternoon + Evening
            [1, 1, 1]   # All three shifts
            # [1, 0, 1] is EXCLUDED (Morning + Evening without Afternoon)
        ]
        
        # RL Parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = {}
        
        # Experience memory for learning
        self.experience = []
        
        # Initialize Q-table
        self._initialize_q_table()
        
        print("🤖 RL Agent initialized for equal-demand optimization!")
    
    def _initialize_q_table(self):
        """Initialize Q-table for equal-demand scenarios"""
        for day in self.days:
            for shift in self.shifts:
                for emp in self.employees:
                    state_key = f"{day}_{shift}_{emp}"
                    self.q_table[state_key] = {
                        'assign': 0.0,      # Reward for assigning work
                        'skip': 0.0         # Reward for not assigning
                    }
    
    def get_state_key(self, day, shift, employee, current_schedule):
        """Create state representation for equal-demand optimization"""
        # Calculate current employee workload
        emp_workload = sum(
            sum(current_schedule[employee][d][s] for s in self.shifts) 
            for d in self.days
        )
        
        # Calculate utilization balance across all employees
        utilizations = []
        for emp in self.employees:
            capacity = self.employee_capacity[emp]
            assigned = sum(
                sum(current_schedule[emp][d][s] for s in self.shifts) 
                for d in self.days
            )
            utilization = assigned / capacity if capacity > 0 else 0
            utilizations.append(utilization)
        
        util_std = np.std(utilizations) if utilizations else 0
        balance_level = "balanced" if util_std <= 0.1 else "unbalanced"
        
        return f"{day}_{shift}_{employee}_work{emp_workload}_bal{balance_level}"
    
    def calculate_reward(self, action, day, shift, employee, current_schedule):
        """
        Equal-demand reward function focusing on perfect resource utilization
        """
        reward = 0
        
        if action == 'assign':
            # 1. Demand coverage reward (must be perfect)
            required = self.demand[(day, shift)]
            currently_assigned = sum(
                current_schedule[emp][day][shift] for emp in self.employees
            )
            if currently_assigned < required:
                reward += 50  # Good to fill demand
            else:
                reward -= 200  # Cannot overfill in equal demand
            
            # 2. Employee capacity constraint
            emp_total = sum(
                sum(current_schedule[employee][d][s] for s in self.shifts) 
                for d in self.days
            )
            capacity = self.employee_capacity[employee]
            
            if emp_total < capacity:
                # Good to utilize capacity
                utilization = (emp_total + 1) / capacity
                reward += utilization * 30
            else:
                reward -= 100  # Cannot exceed capacity
            
            # 3. Balanced utilization (important for equal demand)
            all_utilizations = []
            for emp in self.employees:
                emp_assigned = sum(
                    sum(current_schedule[emp][d][s] for s in self.shifts) 
                    for d in self.days
                )
                if emp == employee:
                    emp_assigned += 1  # Account for current assignment
                
                emp_capacity = self.employee_capacity[emp]
                utilization = emp_assigned / emp_capacity if emp_capacity > 0 else 0
                all_utilizations.append(utilization)
            
            # Reward balanced utilization
            if all_utilizations:
                util_std = np.std(all_utilizations)
                if util_std <= 0.05:
                    reward += 40  # Excellent balance
                elif util_std <= 0.1:
                    reward += 20  # Good balance
                else:
                    reward -= util_std * 30  # Penalty for imbalance
            
            # 4. Consecutive shift constraint validation
            daily_assignments = [current_schedule[employee][day][s] for s in self.shifts]
            if employee == employee:  # Current assignment
                shift_idx = self.shifts.index(shift)
                daily_assignments[shift_idx] += 1
            
            # Check if pattern is valid
            if daily_assignments not in self.valid_daily_patterns:
                reward -= 500  # Heavy penalty for invalid patterns
            else:
                reward += 10   # Bonus for valid patterns
        
        else:  # action == 'skip'
            # Check if demand would still be met
            required = self.demand[(day, shift)]
            currently_assigned = sum(
                current_schedule[emp][day][shift] for emp in self.employees
            )
            if currently_assigned >= required:
                reward += 10  # Good to not overfill
            else:
                reward -= 30  # Bad to leave demand unmet
        
        return reward
    
    def choose_action(self, state_key):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(['assign', 'skip'])
        else:
            # Exploitation: best known action
            q_values = self.q_table.get(state_key, {'assign': 0, 'skip': 0})
            return 'assign' if q_values['assign'] >= q_values['skip'] else 'skip'
    
    def update_q_table(self, state_key, action, reward):
        """Update Q-values with learning"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {'assign': 0.0, 'skip': 0.0}
        
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + self.learning_rate * (reward - current_q)
    
    def create_initial_schedule(self):
        """Create empty schedule structure"""
        schedule = {}
        for emp in self.employees:
            schedule[emp] = {}
            for day in self.days:
                schedule[emp][day] = {s: 0 for s in self.shifts}
        return schedule
    
    def is_valid_assignment(self, employee, day, shift, current_schedule):
        """Check if assignment is valid under equal-demand constraints"""
        
        # Check employee capacity
        emp_total = sum(
            sum(current_schedule[employee][d][s] for s in self.shifts) 
            for d in self.days
        )
        if emp_total >= self.employee_capacity[employee]:
            return False
        
        # Check if demand is already fulfilled
        required = self.demand[(day, shift)]
        currently_assigned = sum(
            current_schedule[emp][day][shift] for emp in self.employees
        )
        if currently_assigned >= required:
            return False
        
        # Check consecutive shift constraint
        daily_assignments = [current_schedule[employee][day][s] for s in self.shifts]
        shift_idx = self.shifts.index(shift)
        daily_assignments[shift_idx] = 1  # Simulate assignment
        
        if daily_assignments not in self.valid_daily_patterns:
            return False
        
        return True
    
    def generate_schedule_episode(self):
        """Generate one complete schedule using current Q-table"""
        schedule = self.create_initial_schedule()
        
        # Process each shift assignment opportunity
        assignment_decisions = []
        
        for day in self.days:
            for shift in self.shifts:
                required = self.demand[(day, shift)]
                assigned_count = 0
                
                # Try to assign required number of employees
                available_employees = self.employees.copy()
                random.shuffle(available_employees)  # Randomize for exploration
                
                for emp in available_employees:
                    if assigned_count >= required:
                        break
                    
                    if self.is_valid_assignment(emp, day, shift, schedule):
                        state_key = self.get_state_key(day, shift, emp, schedule)
                        action = self.choose_action(state_key)
                        
                        if action == 'assign':
                            schedule[emp][day][shift] = 1
                            assigned_count += 1
                        
                        # Calculate reward and store for learning
                        reward = self.calculate_reward(action, day, shift, emp, schedule)
                        assignment_decisions.append((state_key, action, reward))
        
        return schedule, assignment_decisions
    
    def train(self, episodes=1000):
        """Train the RL agent"""
        print(f"🎓 Training RL agent for {episodes} episodes...")
        
        best_schedule = None
        best_score = float('-inf')
        episode_scores = []
        
        for episode in range(episodes):
            schedule, decisions = self.generate_schedule_episode()
            
            # Update Q-table based on decisions
            for state_key, action, reward in decisions:
                self.update_q_table(state_key, action, reward)
            
            # Evaluate schedule quality
            score = self.evaluate_schedule(schedule)
            episode_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_schedule = deepcopy(schedule)
            
            # Decay epsilon (reduce exploration over time)
            if episode % 100 == 0:
                self.epsilon = max(0.05, self.epsilon * 0.95)
                avg_score = np.mean(episode_scores[-100:]) if len(episode_scores) >= 100 else np.mean(episode_scores)
                print(f"Episode {episode}: Best score = {best_score:.2f}, Avg score = {avg_score:.2f}, Epsilon = {self.epsilon:.3f}")
        
        self.best_schedule = best_schedule
        print(f"✅ Training completed! Best score: {best_score:.2f}")
        return best_schedule
    
    def evaluate_schedule(self, schedule):
        """Evaluate schedule quality focusing on equal-demand optimization"""
        score = 0
        
        # 1. Demand coverage (must be perfect)
        demand_violations = 0
        for day in self.days:
            for shift in self.shifts:
                required = self.demand[(day, shift)]
                assigned = sum(schedule[emp][day][shift] for emp in self.employees)
                if assigned != required:
                    demand_violations += abs(assigned - required)
        
        if demand_violations == 0:
            score += 2000  # Perfect demand coverage
        else:
            score -= demand_violations * 200  # Heavy penalty
        
        # 2. Capacity utilization
        total_capacity = sum(self.employee_capacity.values())
        total_assigned = sum(
            sum(sum(schedule[emp][d][s] for s in self.shifts) for d in self.days)
            for emp in self.employees
        )
        utilization_rate = total_assigned / total_capacity if total_capacity > 0 else 0
        score += utilization_rate * 500  # Reward high utilization
        
        # 3. Constraint compliance (consecutive shifts)
        constraint_violations = 0
        for emp in self.employees:
            # Check capacity constraints
            total_shifts = sum(
                sum(schedule[emp][d][s] for s in self.shifts) 
                for d in self.days
            )
            if total_shifts > self.employee_capacity[emp]:
                constraint_violations += total_shifts - self.employee_capacity[emp]
            
            # Check consecutive shift constraints
            for day in self.days:
                daily_pattern = [schedule[emp][day][s] for s in self.shifts]
                if daily_pattern not in self.valid_daily_patterns:
                    constraint_violations += 1
        
        score -= constraint_violations * 100
        
        # 4. Balanced utilization
        utilizations = []
        for emp in self.employees:
            total_assigned = sum(
                sum(schedule[emp][d][s] for s in self.shifts) 
                for d in self.days
            )
            capacity = self.employee_capacity[emp]
            utilization = total_assigned / capacity if capacity > 0 else 0
            utilizations.append(utilization)
        
        if utilizations:
            # Reward balanced utilization
            util_std = np.std(utilizations)
            score += (1.0 - util_std) * 300
        
        return score
    
    def validate_schedule_constraints(self, schedule):
        """Validate all constraints and return violations"""
        violations = []
        
        # Check consecutive shift constraints
        consecutive_violations = 0
        for emp in self.employees:
            for day in self.days:
                daily_pattern = [schedule[emp][day][s] for s in self.shifts]
                if daily_pattern not in self.valid_daily_patterns:
                    consecutive_violations += 1
                    violations.append(f"{emp} on {day}: Invalid pattern {daily_pattern}")
        
        # Check capacity constraints
        capacity_violations = 0
        for emp in self.employees:
            total_shifts = sum(
                sum(schedule[emp][d][s] for s in self.shifts) 
                for d in self.days
            )
            capacity = self.employee_capacity[emp]
            if total_shifts > capacity:
                capacity_violations += total_shifts - capacity
                violations.append(f"{emp}: {total_shifts} shifts exceeds capacity {capacity}")
        
        # Check demand coverage
        demand_violations = 0
        for day in self.days:
            for shift in self.shifts:
                required = self.demand[(day, shift)]
                assigned = sum(schedule[emp][day][shift] for emp in self.employees)
                if assigned != required:
                    demand_violations += abs(assigned - required)
                    violations.append(f"{day} {shift}: {assigned} assigned vs {required} required")
        
        print(f"\n🔍 CONSTRAINT VALIDATION:")
        print(f"Consecutive shift violations: {consecutive_violations}")
        print(f"Capacity violations: {capacity_violations}")  
        print(f"Demand coverage violations: {demand_violations}")
        print(f"Total violations: {len(violations)}")
        
        if violations:
            print("\nDetailed violations:")
            for violation in violations[:10]:  # Show first 10
                print(f"  - {violation}")
            if len(violations) > 10:
                print(f"  ... and {len(violations) - 10} more")
        
        return len(violations) == 0
    
    def create_employee_schedules(self, output_dir):
        """Create individual employee schedule files"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not hasattr(self, 'best_schedule') or self.best_schedule is None:
            print("❌ No trained schedule available. Run train() first.")
            return
        
        for emp in self.employees:
            schedule_data = []
            
            for day in self.days:
                row = {'weekday': day}
                for shift in self.shifts:
                    row[shift] = self.best_schedule[emp][day][shift]
                schedule_data.append(row)
            
            schedule_df = pd.DataFrame(schedule_data)
            filename = f"{emp}_schedule.csv"
            filepath = os.path.join(output_dir, filename)
            schedule_df.to_csv(filepath, index=False)
    
    def create_metrics_summary(self, output_dir):
        """Create employee metrics CSV for comparison analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not hasattr(self, 'best_schedule') or self.best_schedule is None:
            print("❌ No trained schedule available. Run train() first.")
            return
        
        metrics_data = []
        
        for emp in self.employees:
            total_shifts = sum(
                sum(self.best_schedule[emp][d][s] for s in self.shifts) 
                for d in self.days
            )
            capacity = self.employee_capacity[emp]
            utilization = total_shifts / capacity if capacity > 0 else 0
            shift_difference = capacity - total_shifts
            
            metrics_data.append({
                'employee_id': emp,
                'total_shifts_assigned': total_shifts,
                'capacity': capacity,
                'utilization_rate': round(utilization, 3),
                'shift_difference': shift_difference
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = os.path.join(output_dir, 'employee_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"📊 Employee metrics saved to: {metrics_file}")
        
        return metrics_df
    
    def print_solution_summary(self):
        """Print comprehensive solution summary"""
        if not hasattr(self, 'best_schedule') or self.best_schedule is None:
            print("❌ No trained schedule available. Run train() first.")
            return
        
        print("\n=== RL EQUAL-DEMAND SCHEDULING SOLUTION ===")
        
        # Validate constraints first
        is_valid = self.validate_schedule_constraints(self.best_schedule)
        
        # Employee work summary
        print("\nEmployee Work Distribution:")
        utilizations = []
        total_assigned = 0
        total_capacity = 0
        
        for emp in self.employees:
            assigned = sum(
                sum(self.best_schedule[emp][d][s] for s in self.shifts) 
                for d in self.days
            )
            capacity = self.employee_capacity[emp]
            utilization = assigned / capacity if capacity > 0 else 0
            unused = capacity - assigned
            
            utilizations.append(utilization)
            total_assigned += assigned
            total_capacity += capacity
            
            print(f"{emp}: {assigned}/{capacity} shifts ({utilization:.1%} utilization, {unused} unused)")
        
        # Overall summary
        overall_utilization = total_assigned / total_capacity if total_capacity > 0 else 0
        print(f"\nOverall Summary:")
        print(f"Total shifts assigned: {total_assigned}")
        print(f"Total capacity: {total_capacity}")
        print(f"Overall utilization: {overall_utilization:.1%}")
        print(f"Unused capacity: {total_capacity - total_assigned} shifts")
        
        # Fairness metrics
        if utilizations:
            print(f"\nFairness Metrics:")
            print(f"Min utilization: {min(utilizations):.1%}")
            print(f"Max utilization: {max(utilizations):.1%}")
            print(f"Average utilization: {statistics.mean(utilizations):.1%}")
            print(f"Utilization std dev: {statistics.stdev(utilizations) if len(utilizations) > 1 else 0:.3f}")
            print(f"Utilization range: {max(utilizations) - min(utilizations):.3f}")
        
        # Demand coverage verification
        print("\nDemand Coverage Verification:")
        all_covered = True
        total_demand = 0
        for day in self.days:
            print(f"\n{day.upper()}:")
            for shift in self.shifts:
                required = self.demand[(day, shift)]
                total_demand += required
                assigned_employees = []
                for emp in self.employees:
                    if self.best_schedule[emp][day][shift] == 1:
                        assigned_employees.append(emp)
                
                assigned = len(assigned_employees)
                status = "✅" if assigned == required else "❌"
                if assigned != required:
                    all_covered = False
                print(f"  {status} {shift}: {assigned}/{required} - {assigned_employees}")
        
        print(f"\nTotal demand: {total_demand} shifts")
        if all_covered:
            print(f"✅ Perfect demand coverage achieved!")
        else:
            print(f"❌ Demand coverage issues detected")
        
        if is_valid:
            print(f"✅ All constraints satisfied!")
        else:
            print(f"❌ Constraint violations detected!")


def main():
    # File paths
    base_dir = Path(__file__).parent.parent
    forecast_file = base_dir / "data" / "forecast_szenario_a.csv"
    employees_file = base_dir / "data" / "employees.csv"
    output_dir = Path(__file__).parent / "employee_schedules"
    
    # Create RL agent
    agent = EqualDemandRLAgent(forecast_file, employees_file)
    
    print("Starting RL training for equal-demand scenario...")
    
    # Train the agent
    try:
        best_schedule = agent.train(episodes=1500)
        
        print("✅ RL Training completed successfully!")
        
        # Print solution summary
        agent.print_solution_summary()
        
        # Create individual employee schedules
        agent.create_employee_schedules(output_dir)
        
        # Create metrics summary for comparison
        agent.create_metrics_summary(output_dir)
        
        print(f"\n✅ Employee schedules saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during RL training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
