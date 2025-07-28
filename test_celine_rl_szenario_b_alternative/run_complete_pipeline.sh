#!/bin/bash
"""
Complete Pipeline Runner for RL Shift Optimization
Runs training, generates schedules, and creates all analysis files
"""

echo "🚀 COMPLETE RL SHIFT OPTIMIZATION PIPELINE"
echo "=========================================="
echo

# Step 1: Train RL Agent and Generate Schedules
echo "1️⃣ Training RL Agent and Generating Employee Schedules..."
python3 rl_shift_optimizer.py
if [ $? -eq 0 ]; then
    echo "✅ RL training completed successfully!"
else
    echo "❌ RL training failed!"
    exit 1
fi
echo

# Step 2: Generate Analysis Results
echo "2️⃣ Generating Analysis Results and Visualizations..."
python3 results_exporter.py
if [ $? -eq 0 ]; then
    echo "✅ Results export completed successfully!"
else
    echo "❌ Results export failed!"
    exit 1
fi
echo

# Step 3: Verify Results
echo "3️⃣ Verifying Results..."
python3 -c "
import pandas as pd
import os

# Get current directory name to determine scenario
current_dir = os.path.basename(os.getcwd())
print(f'📁 Scenario: {current_dir}')

# Determine forecast file based on directory name
base_dir = '/Users/celineschmnn/Desktop/Uni/FOM.AnwendungsfelderBA.RL-1'
forecast_files = {
    'test_celine_rl_szenario_a': f'{base_dir}/data/forecast_szenario_a.csv',
    'test_celine_rl_szenario_b': f'{base_dir}/data/forecast_szenario_b.csv', 
    'test_celine_rl_szenario_c': f'{base_dir}/data/forecast_szenario_c.csv',
    'test_celine_rl_szenario_b_alternative': f'{base_dir}/data/forecast_szenario_b_alternative.csv'
}

forecast_file = forecast_files.get(current_dir, f'{base_dir}/data/forecast.csv')

try:
    # Check forecast data
    forecast_df = pd.read_csv(forecast_file)
    print('📋 FORECAST (Required):')
    print(forecast_df.to_string(index=False))
    print()

    # Check aggregated schedule
    agg_df = pd.read_csv('results_export/aggregated_schedule.csv')
    print('📊 AGGREGATED SCHEDULE (Assigned):')
    print(agg_df.to_string(index=False))
    print()

    # Calculate totals
    forecast_total = forecast_df[['shift_morning', 'shift_afternoon', 'shift_evening']].sum().sum()
    assigned_total = agg_df[['shift_morning', 'shift_afternoon', 'shift_evening']].sum().sum()

    print(f'🎯 VERIFICATION SUMMARY:')
    print(f'   Required shifts: {forecast_total}')
    print(f'   Assigned shifts: {assigned_total}')
    print(f'   Coverage: {assigned_total}/{forecast_total} ({assigned_total/forecast_total*100:.1f}%)')
    
    if assigned_total == forecast_total:
        print('   ✅ Perfect demand coverage!')
    elif assigned_total < forecast_total:
        print(f'   ⚠️  {forecast_total - assigned_total} shifts under-assigned')
    else:
        print(f'   ⚠️  {assigned_total - forecast_total} shifts over-assigned')
        
    print()
    print('📁 FILES UPDATED:')
    print('   ✅ Individual employee schedules in employee_schedules/')
    print('   ✅ Aggregated schedule in results_export/aggregated_schedule.csv')
    print('   ✅ Analysis metrics in results_export/employee_metrics.csv') 
    print('   ✅ Heatmap visualizations in results_export/')
    print('   ✅ Summary report in results_export/summary_report.csv')

except Exception as e:
    print(f'❌ Verification failed: {e}')
"

echo
echo "🎉 PIPELINE COMPLETE!"
echo "All files have been updated with fresh RL optimization results."
echo "Check the results_export/ folder for analysis files and visualizations."
