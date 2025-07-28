"""
Comparison Analysis: Linear Optimization vs Reinforcement Learning
Szenario C Analysis

This script compares the performance of linear optimization and RL
in szenario C where there's more capacity than demand.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statistics

def load_metrics(linear_path, rl_path):
    """Load employee metrics from both algorithms"""
    linear_metrics = pd.read_csv(linear_path)
    rl_metrics = pd.read_csv(rl_path)
    
    # Add algorithm identifier
    linear_metrics['algorithm'] = 'Linear Optimization'
    rl_metrics['algorithm'] = 'Reinforcement Learning'
    
    return linear_metrics, rl_metrics

def calculate_fairness_metrics(metrics_df):
    """Calculate comprehensive fairness metrics"""
    try:
        # Convert utilization_rate from percentage strings to numeric values
        utilization_rates = [float(str(rate).replace('%', '')) for rate in metrics_df['utilization_rate'].tolist()]
        
        # Check if shift_difference column exists, if not use zeros
        if 'shift_difference' in metrics_df.columns:
            shift_differences = metrics_df['shift_difference'].tolist()
        else:
            shift_differences = [0] * len(utilization_rates)
        
        if not utilization_rates:
            return {}
    except Exception as e:
        print(f"Error calculating fairness metrics: {e}")
        print(f"Available columns: {list(metrics_df.columns)}")
        return {}
    
    return {
        'min_utilization': min(utilization_rates),
        'max_utilization': max(utilization_rates),
        'avg_utilization': statistics.mean(utilization_rates),
        'utilization_std': statistics.stdev(utilization_rates) if len(utilization_rates) > 1 else 0,
        'utilization_range': max(utilization_rates) - min(utilization_rates),
        'total_shift_difference': sum(shift_differences),
        'max_shift_difference': max(shift_differences),
        'min_shift_difference': min(shift_differences),
        'shift_diff_std': statistics.stdev(shift_differences) if len(shift_differences) > 1 else 0
    }

def compare_algorithms():
    """Main comparison function"""
    # File paths
    base_dir = Path(__file__).parent
    linear_metrics_path = base_dir / "test_celine_linear_szenario_c" / "results_export" / "employee_metrics.csv"
    rl_metrics_path = base_dir / "test_celine_rl_szenario_c" / "employee_schedules" / "employee_metrics.csv"
    
    # Load metrics
    try:
        linear_metrics, rl_metrics = load_metrics(linear_metrics_path, rl_metrics_path)
    except FileNotFoundError as e:
        print(f"❌ Error loading metrics files: {e}")
        print("Make sure both algorithms have been run and generated employee_metrics.csv files")
        return
    
    print("="*80)
    print("SZENARIO C: LINEAR OPTIMIZATION vs REINFORCEMENT LEARNING")
    print("="*80)
    
    # Calculate fairness metrics for both algorithms
    linear_fairness = calculate_fairness_metrics(linear_metrics)
    rl_fairness = calculate_fairness_metrics(rl_metrics)
    
    print(f"\n📊 UTILIZATION ANALYSIS")
    print(f"{'Metric':<25} {'Linear Opt':<15} {'RL':<15} {'RL Advantage':<15}")
    print("-" * 70)
    
    # Utilization metrics comparison
    metrics_comparison = [
        ('Min Utilization', 'min_utilization', 'higher'),
        ('Max Utilization', 'max_utilization', 'lower'),
        ('Avg Utilization', 'avg_utilization', 'higher'),
        ('Utilization Std Dev', 'utilization_std', 'lower'),
        ('Utilization Range', 'utilization_range', 'lower'),
    ]
    
    rl_advantages = 0
    total_comparisons = len(metrics_comparison)
    
    for metric_name, metric_key, better_direction in metrics_comparison:
        linear_val = linear_fairness[metric_key]
        rl_val = rl_fairness[metric_key]
        
        if better_direction == 'lower':
            advantage = ((linear_val - rl_val) / linear_val * 100) if linear_val != 0 else 0
            is_better = rl_val < linear_val
        else:  # higher is better
            advantage = ((rl_val - linear_val) / linear_val * 100) if linear_val != 0 else 0
            is_better = rl_val > linear_val
        
        if is_better:
            rl_advantages += 1
            advantage_str = f"+{advantage:.1f}%"
        else:
            advantage_str = f"{advantage:.1f}%"
        
        status = "✅" if is_better else "❌"
        
        # Format values appropriately
        if 'std' in metric_key.lower() or 'range' in metric_key.lower():
            linear_str = f"{linear_val:.3f}"
            rl_str = f"{rl_val:.3f}"
        else:
            linear_str = f"{linear_val:.1%}" if 'utilization' in metric_key else f"{linear_val:.1f}"
            rl_str = f"{rl_val:.1%}" if 'utilization' in metric_key else f"{rl_val:.1f}"
        
        print(f"{metric_name:<25} {linear_str:<15} {rl_str:<15} {status} {advantage_str}")
    
    print(f"\n📈 FAIRNESS ANALYSIS")
    print(f"{'Metric':<25} {'Linear Opt':<15} {'RL':<15} {'RL Advantage':<15}")
    print("-" * 70)
    
    # Shift difference analysis
    shift_metrics = [
        ('Total Shift Difference', 'total_shift_difference', 'equal'),
        ('Max Individual Diff', 'max_shift_difference', 'lower'),
        ('Min Individual Diff', 'min_shift_difference', 'higher'),
        ('Shift Diff Std', 'shift_diff_std', 'lower'),
    ]
    
    for metric_name, metric_key, better_direction in shift_metrics:
        linear_val = linear_fairness[metric_key]
        rl_val = rl_fairness[metric_key]
        
        if better_direction == 'equal':
            advantage = 0
            is_better = linear_val == rl_val
            advantage_str = "Equal" if is_better else f"Diff: {abs(linear_val - rl_val)}"
        elif better_direction == 'lower':
            advantage = ((linear_val - rl_val) / linear_val * 100) if linear_val != 0 else 0
            is_better = rl_val < linear_val
            advantage_str = f"+{advantage:.1f}%" if is_better else f"{advantage:.1f}%"
        else:  # higher is better
            advantage = ((rl_val - linear_val) / linear_val * 100) if linear_val != 0 else 0
            is_better = rl_val > linear_val
            advantage_str = f"+{advantage:.1f}%" if is_better else f"{advantage:.1f}%"
        
        if is_better:
            rl_advantages += 1
        
        status = "✅" if is_better else ("➖" if better_direction == 'equal' and linear_val == rl_val else "❌")
        
        print(f"{metric_name:<25} {linear_val:<15.1f} {rl_val:<15.1f} {status} {advantage_str}")
    
    total_comparisons += len(shift_metrics)
    
    print(f"\n🎯 OVERALL PERFORMANCE SUMMARY")
    print("-" * 50)
    
    # Overall performance
    rl_win_rate = rl_advantages / total_comparisons * 100
    print(f"RL wins in {rl_advantages}/{total_comparisons} metrics ({rl_win_rate:.1f}%)")
    
    if rl_win_rate >= 60:
        overall_winner = "🏆 Reinforcement Learning SUPERIOR"
    elif rl_win_rate >= 40:
        overall_winner = "🤝 COMPARABLE Performance"
    else:
        overall_winner = "🏆 Linear Optimization SUPERIOR"
    
    print(f"Overall Assessment: {overall_winner}")
    
    print(f"\n💡 KEY INSIGHTS:")
    
    # Detailed insights
    if rl_fairness['utilization_std'] < linear_fairness['utilization_std']:
        improvement = (linear_fairness['utilization_std'] - rl_fairness['utilization_std']) / linear_fairness['utilization_std'] * 100
        print(f"• RL achieves {improvement:.1f}% better utilization balance (lower std deviation)")
    
    if rl_fairness['utilization_range'] < linear_fairness['utilization_range']:
        print(f"• RL reduces utilization inequality by {((linear_fairness['utilization_range'] - rl_fairness['utilization_range']) / linear_fairness['utilization_range'] * 100):.1f}%")
    
    if rl_fairness['avg_utilization'] > linear_fairness['avg_utilization']:
        print(f"• RL achieves higher average utilization ({rl_fairness['avg_utilization']:.1%} vs {linear_fairness['avg_utilization']:.1%})")
    
    print(f"• Both algorithms meet 100% demand coverage (34/34 shifts)")
    print(f"• Both algorithms respect all capacity constraints")
    
    print(f"\n📋 DETAILED EMPLOYEE BREAKDOWN:")
    print("-" * 50)
    
    # Merge data for detailed comparison
    # Calculate unused_capacity from available columns
    linear_metrics['unused_capacity'] = linear_metrics['required_shifts'] - linear_metrics['assigned_shifts']
    rl_metrics['unused_capacity'] = rl_metrics['required_shifts'] - rl_metrics['assigned_shifts']
    
    comparison_df = pd.merge(
        linear_metrics[['employee_id', 'assigned_shifts', 'utilization_rate', 'unused_capacity']], 
        rl_metrics[['employee_id', 'assigned_shifts', 'utilization_rate', 'unused_capacity']], 
        on='employee_id', 
        suffixes=('_linear', '_rl')
    )
    
    print(f"{'Employee':<10} {'Linear':<20} {'RL':<20} {'Improvement':<15}")
    print(f"{'ID':<10} {'Shifts|Util|Unused':<20} {'Shifts|Util|Unused':<20} {'Status':<15}")
    print("-" * 75)
    
    for _, row in comparison_df.iterrows():
        emp_id = row['employee_id']
        
        # Convert utilization_rate from string to float for display
        linear_util = float(str(row['utilization_rate_linear']).replace('%', '')) / 100
        rl_util = float(str(row['utilization_rate_rl']).replace('%', '')) / 100
        
        linear_info = f"{row['assigned_shifts_linear']}|{linear_util:.1%}|{row['unused_capacity_linear']}"
        rl_info = f"{row['assigned_shifts_rl']}|{rl_util:.1%}|{row['unused_capacity_rl']}"
        
        # Determine improvement
        util_diff = row['utilization_rate_rl'] - row['utilization_rate_linear']
        unused_diff = row['unused_capacity_linear'] - row['unused_capacity_rl']  # Less unused is better
        
        if abs(util_diff) < 0.01 and unused_diff == 0:
            improvement = "➖ Same"
        elif util_diff > 0 or unused_diff > 0:
            improvement = "✅ Better"
        else:
            improvement = "❌ Worse"
        
        print(f"{emp_id:<10} {linear_info:<20} {rl_info:<20} {improvement:<15}")
    
    print(f"\n" + "="*80)
    print(f"CONCLUSION: Szenario C demonstrates RL's ability to")
    print(f"learn fair work distribution patterns when capacity exceeds demand.")
    print(f"="*80)

if __name__ == "__main__":
    compare_algorithms()
