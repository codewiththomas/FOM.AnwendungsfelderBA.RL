"""
Comparison between Linear Optimization and Reinforcement Learning
for Szenario A Employee Shift Scheduling

This script compares performance metrics between:
- Linear Optimization (test_celine_linear_szenario_a)
- Reinforcement Learning (test_celine_rl_szenario_a)

For the szenario A scenario where demand exactly matches total capacity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def calculate_total_shifts_from_schedules(schedules_dir):
    """Calculate total shifts assigned from individual schedule files"""
    total_shifts = {}
    employees = []
    
    # Get all employee schedule files
    for file in os.listdir(schedules_dir):
        if file.endswith('_schedule.csv'):
            emp_id = file.replace('_schedule.csv', '')
            employees.append(emp_id)
            
            # Read employee schedule
            df = pd.read_csv(schedules_dir / file)
            shifts = ['shift_morning', 'shift_afternoon', 'shift_evening']
            
            total = 0
            for _, row in df.iterrows():
                for shift in shifts:
                    total += row[shift]
            
            total_shifts[emp_id] = total
    
    return total_shifts, employees

def calculate_demand_coverage(schedules_dir, forecast_file):
    """Calculate how well demand is covered"""
    # Load forecast
    forecast_df = pd.read_csv(forecast_file)
    
    # Load all schedules
    shifts = ['shift_morning', 'shift_afternoon', 'shift_evening']
    coverage_data = []
    
    for _, forecast_row in forecast_df.iterrows():
        day = forecast_row['weekday']
        
        for shift in shifts:
            required = forecast_row[shift]
            assigned = 0
            
            # Count assignments across all employees
            for file in os.listdir(schedules_dir):
                if file.endswith('_schedule.csv'):
                    emp_df = pd.read_csv(schedules_dir / file)
                    day_row = emp_df[emp_df['weekday'] == day]
                    if not day_row.empty:
                        assigned += day_row.iloc[0][shift]
            
            violation = abs(assigned - required)
            coverage_data.append({
                'day': day,
                'shift': shift,
                'required': required,
                'assigned': assigned,
                'violation': violation
            })
    
    return pd.DataFrame(coverage_data)

def generate_linear_metrics_from_schedules(schedules_dir):
    """Generate metrics from linear optimization schedule files"""
    
    # Employee capacity data
    employees_data = {
        'MA1': 10, 'MA2': 10, 'MA3': 5, 'MA4': 5, 'MA5': 5, 'MA6': 5
    }
    
    total_shifts, employees = calculate_total_shifts_from_schedules(schedules_dir)
    
    metrics_data = []
    for emp in employees:
        capacity = employees_data.get(emp, 5)  # Default capacity
        assigned = total_shifts.get(emp, 0)
        utilization = assigned / capacity if capacity > 0 else 0
        shift_difference = capacity - assigned
        
        metrics_data.append({
            'employee_id': emp,
            'total_shifts_assigned': assigned,
            'capacity': capacity,
            'utilization_rate': round(utilization, 3),
            'shift_difference': shift_difference
        })
    
    return pd.DataFrame(metrics_data)

def load_rl_metrics(rl_dir):
    """Load RL metrics from employee_metrics.csv"""
    metrics_file = rl_dir / "employee_schedules" / "employee_metrics.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    else:
        print(f"❌ RL metrics file not found: {metrics_file}")
        return None

def calculate_fairness_metrics(df):
    """Calculate fairness metrics from employee data"""
    if df is None or df.empty:
        return None
    
    utilizations = df['utilization_rate'].values
    shift_differences = df['shift_difference'].values
    
    return {
        'utilization_mean': np.mean(utilizations),
        'utilization_std': np.std(utilizations),
        'utilization_min': np.min(utilizations),
        'utilization_max': np.max(utilizations),
        'utilization_range': np.max(utilizations) - np.min(utilizations),
        'shift_diff_mean': np.mean(shift_differences),
        'shift_diff_std': np.std(shift_differences),
        'shift_diff_max': np.max(shift_differences),
        'total_unused_capacity': np.sum(shift_differences)
    }

def main():
    print("🔄 SZENARIO A: Linear vs RL Comparison")
    print("=" * 60)
    
    # Load data
    base_dir = Path("/Users/celineschmnn/Desktop/Uni/FOM.AnwendungsfelderBA.RL-1")
    forecast_file = base_dir / "data" / "forecast_szenario_a.csv"
    
    linear_schedules_dir = base_dir / "test_celine_linear_szenario_a" / "employee_schedules"
    rl_dir = base_dir / "test_celine_rl_szenario_a"
    rl_schedules_dir = rl_dir / "employee_schedules"
    
    # Generate metrics for linear optimization
    print("📊 Generating Linear Optimization Metrics...")
    linear_metrics = generate_linear_metrics_from_schedules(linear_schedules_dir)
    
    # Load RL metrics
    print("📊 Loading RL Metrics...")
    rl_metrics = load_rl_metrics(rl_dir)
    
    if linear_metrics is None or rl_metrics is None:
        print("❌ Failed to load metrics data")
        return
    
    # Calculate demand coverage for both approaches
    print("📊 Calculating Demand Coverage...")
    linear_coverage = calculate_demand_coverage(linear_schedules_dir, forecast_file)
    rl_coverage = calculate_demand_coverage(rl_schedules_dir, forecast_file)
    
    # Calculate fairness metrics
    linear_fairness = calculate_fairness_metrics(linear_metrics)
    rl_fairness = calculate_fairness_metrics(rl_metrics)
    
    # Print detailed comparison
    print("\\n" + "=" * 60)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print("\\n🎯 DEMAND COVERAGE:")
    linear_violations = linear_coverage['violation'].sum()
    rl_violations = rl_coverage['violation'].sum()
    linear_perfect_coverage = (linear_coverage['violation'] == 0).all()
    rl_perfect_coverage = (rl_coverage['violation'] == 0).all()
    
    print(f"Linear Optimization:")
    print(f"  - Total violations: {linear_violations}")
    print(f"  - Perfect coverage: {'✅ YES' if linear_perfect_coverage else '❌ NO'}")
    print(f"  - Success rate: {((linear_coverage['violation'] == 0).sum() / len(linear_coverage)) * 100:.1f}%")
    
    print(f"\\nReinforcement Learning:")
    print(f"  - Total violations: {rl_violations}")
    print(f"  - Perfect coverage: {'✅ YES' if rl_perfect_coverage else '❌ NO'}")
    print(f"  - Success rate: {((rl_coverage['violation'] == 0).sum() / len(rl_coverage)) * 100:.1f}%")
    
    if linear_violations == 0 and rl_violations > 0:
        print(f"\\n🏆 Winner: LINEAR OPTIMIZATION (Perfect coverage)")
    elif rl_violations == 0 and linear_violations > 0:
        print(f"\\n🏆 Winner: REINFORCEMENT LEARNING (Perfect coverage)")
    elif linear_violations < rl_violations:
        print(f"\\n🏆 Winner: LINEAR OPTIMIZATION ({linear_violations} vs {rl_violations} violations)")
    elif rl_violations < linear_violations:
        print(f"\\n🏆 Winner: REINFORCEMENT LEARNING ({rl_violations} vs {linear_violations} violations)")
    else:
        print(f"\\n🤝 Tie: Both have {linear_violations} violations")
    
    print("\\n⚖️ FAIRNESS METRICS:")
    print(f"Linear Optimization:")
    print(f"  - Utilization: {linear_fairness['utilization_mean']:.1%} ± {linear_fairness['utilization_std']:.3f}")
    print(f"  - Range: {linear_fairness['utilization_min']:.1%} - {linear_fairness['utilization_max']:.1%}")
    print(f"  - Max unused capacity: {linear_fairness['shift_diff_max']} shifts")
    print(f"  - Total unused capacity: {linear_fairness['total_unused_capacity']} shifts")
    
    print(f"\\nReinforcement Learning:")
    print(f"  - Utilization: {rl_fairness['utilization_mean']:.1%} ± {rl_fairness['utilization_std']:.3f}")
    print(f"  - Range: {rl_fairness['utilization_min']:.1%} - {rl_fairness['utilization_max']:.1%}")
    print(f"  - Max unused capacity: {rl_fairness['shift_diff_max']} shifts")
    print(f"  - Total unused capacity: {rl_fairness['total_unused_capacity']} shifts")
    
    # Fairness comparison
    linear_fairness_score = 1 / (1 + linear_fairness['utilization_std'])
    rl_fairness_score = 1 / (1 + rl_fairness['utilization_std'])
    
    if linear_fairness_score > rl_fairness_score:
        fairness_winner = "LINEAR OPTIMIZATION"
        fairness_improvement = ((linear_fairness_score - rl_fairness_score) / rl_fairness_score) * 100
    else:
        fairness_winner = "REINFORCEMENT LEARNING"
        fairness_improvement = ((rl_fairness_score - linear_fairness_score) / linear_fairness_score) * 100
    
    print(f"\\n🏆 Fairness Winner: {fairness_winner} ({fairness_improvement:.1f}% better)")
    
    print("\\n" + "=" * 60)
    print("📋 DETAILED VIOLATION BREAKDOWN")
    print("=" * 60)
    
    if linear_violations > 0:
        print("\\nLinear Optimization violations:")
        linear_viols = linear_coverage[linear_coverage['violation'] > 0]
        for _, row in linear_viols.iterrows():
            print(f"  - {row['day']} {row['shift']}: {row['assigned']}/{row['required']} (off by {row['violation']})")
    
    if rl_violations > 0:
        print("\\nRL violations:")
        rl_viols = rl_coverage[rl_coverage['violation'] > 0]
        for _, row in rl_viols.iterrows():
            print(f"  - {row['day']} {row['shift']}: {row['assigned']}/{row['required']} (off by {row['violation']})")
    
    print("\\n" + "=" * 60)
    print("🎯 FINAL VERDICT")
    print("=" * 60)
    
    # Overall scoring
    linear_score = 0
    rl_score = 0
    
    # Demand coverage (most important)
    if linear_violations < rl_violations:
        linear_score += 3
        coverage_verdict = "Linear wins on coverage"
    elif rl_violations < linear_violations:
        rl_score += 3
        coverage_verdict = "RL wins on coverage"
    else:
        linear_score += 1.5
        rl_score += 1.5
        coverage_verdict = "Tie on coverage"
    
    # Fairness
    if linear_fairness_score > rl_fairness_score:
        linear_score += 1
        fairness_verdict = "Linear wins on fairness"
    else:
        rl_score += 1
        fairness_verdict = "RL wins on fairness"
    
    print(f"Coverage: {coverage_verdict}")
    print(f"Fairness: {fairness_verdict}")
    print(f"\\nFinal Scores:")
    print(f"  Linear Optimization: {linear_score:.1f}/4.0")
    print(f"  Reinforcement Learning: {rl_score:.1f}/4.0")
    
    if linear_score > rl_score:
        print(f"\\n🏆 OVERALL WINNER: LINEAR OPTIMIZATION")
        print(f"Linear optimization performs better in this szenario A")
    elif rl_score > linear_score:
        print(f"\\n🏆 OVERALL WINNER: REINFORCEMENT LEARNING")
        print(f"RL performs better despite being a learning-based approach")
    else:
        print(f"\\n🤝 OVERALL TIE")
        print(f"Both approaches perform similarly in this scenario")


if __name__ == "__main__":
    main()
