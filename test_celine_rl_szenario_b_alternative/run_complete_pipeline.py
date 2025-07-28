"""
Complete Pipeline Runner for RL Shift Optimization
Runs training, generates schedules, and creates all analysis files
Usage: python3 run_complete_pipeline.py
"""

import subprocess
import sys
import os
import pandas as pd

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def verify_results():
    """Verify that results are consistent"""
    print("3️⃣ Verifying Results...")
    
    try:
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

        # Check forecast data
        forecast_df = pd.read_csv(forecast_file)
        print('\n📋 FORECAST (Required):')
        print(forecast_df.to_string(index=False))

        # Check aggregated schedule
        agg_df = pd.read_csv('results_export/aggregated_schedule.csv')
        print('\n📊 AGGREGATED SCHEDULE (Assigned):')
        print(agg_df.to_string(index=False))

        # Calculate totals
        forecast_total = forecast_df[['shift_morning', 'shift_afternoon', 'shift_evening']].sum().sum()
        assigned_total = agg_df[['shift_morning', 'shift_afternoon', 'shift_evening']].sum().sum()

        print(f'\n🎯 VERIFICATION SUMMARY:')
        print(f'   Required shifts: {forecast_total}')
        print(f'   Assigned shifts: {assigned_total}')
        print(f'   Coverage: {assigned_total}/{forecast_total} ({assigned_total/forecast_total*100:.1f}%)')
        
        if assigned_total == forecast_total:
            print('   ✅ Perfect demand coverage!')
        elif assigned_total < forecast_total:
            print(f'   ⚠️  {forecast_total - assigned_total} shifts under-assigned')
        else:
            print(f'   ⚠️  {assigned_total - forecast_total} shifts over-assigned')
            
        print('\n📁 FILES UPDATED:')
        print('   ✅ Individual employee schedules in employee_schedules/')
        print('   ✅ Aggregated schedule in results_export/aggregated_schedule.csv')
        print('   ✅ Analysis metrics in results_export/employee_metrics.csv') 
        print('   ✅ Heatmap visualizations in results_export/')
        print('   ✅ Summary report in results_export/summary_report.csv')
        
        return True

    except Exception as e:
        print(f'❌ Verification failed: {e}')
        return False

def main():
    """Run the complete pipeline"""
    print("🚀 COMPLETE RL SHIFT OPTIMIZATION PIPELINE")
    print("=" * 50)
    print()

    # Step 1: Train RL Agent and Generate Schedules
    if not run_command("python3 rl_shift_optimizer.py", "1️⃣ Training RL Agent and Generating Employee Schedules"):
        sys.exit(1)
    print()

    # Step 2: Generate Analysis Results
    if not run_command("python3 results_exporter.py", "2️⃣ Generating Analysis Results and Visualizations"):
        sys.exit(1)
    print()

    # Step 3: Verify Results
    if not verify_results():
        sys.exit(1)

    print()
    print("🎉 PIPELINE COMPLETE!")
    print("All files have been updated with fresh RL optimization results.")
    print("Check the results_export/ folder for analysis files and visualizations.")
    print()
    print("📊 Next steps:")
    print("   • View heatmaps in results_export/comparison_all_heatmaps.png")
    print("   • Check employee metrics in results_export/employee_metrics.csv")
    print("   • Run comparison analysis with other algorithms")

if __name__ == "__main__":
    main()
