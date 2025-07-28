"""
Complete Pipeline Runner for Linear Shift Optimization
Runs optimization, generates schedules, and creates all analysis files
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
            'test_celine_linear_szenario_a': f'{base_dir}/data/forecast_szenario_a.csv',
            'test_celine_linear_szenario_b': f'{base_dir}/data/forecast_szenario_b.csv', 
            'test_celine_linear_szenario_c': f'{base_dir}/data/forecast_szenario_c.csv',
            'test_celine_linear_szenario_b_alternative': f'{base_dir}/data/forecast_szenario_b_alternative.csv'
        }

        forecast_file = forecast_files.get(current_dir, f'{base_dir}/data/forecast.csv')

        # Check if results_export exists
        if not os.path.exists('results_export'):
            print('❌ results_export folder not found! Results exporter may have failed.')
            return False

        # Check forecast data
        forecast_df = pd.read_csv(forecast_file)
        print('\n📋 FORECAST (Required):')
        print(forecast_df.to_string(index=False))

        # Check aggregated schedule if it exists
        agg_file = 'results_export/aggregated_schedule.csv'
        if os.path.exists(agg_file):
            agg_df = pd.read_csv(agg_file)
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
        else:
            print('\n⚠️  Aggregated schedule not found - optimization may have failed')
            
        print('\n📁 FILES UPDATED:')
        
        # Check what files were actually created
        if os.path.exists('employee_schedules'):
            schedule_files = [f for f in os.listdir('employee_schedules') if f.endswith('_schedule.csv')]
            print(f'   ✅ Individual employee schedules ({len(schedule_files)} files) in employee_schedules/')
        else:
            print('   ❌ No employee_schedules folder found')
            
        if os.path.exists('results_export/aggregated_schedule.csv'):
            print('   ✅ Aggregated schedule in results_export/aggregated_schedule.csv')
        else:
            print('   ❌ Aggregated schedule not found')
            
        if os.path.exists('results_export/employee_metrics.csv'):
            print('   ✅ Analysis metrics in results_export/employee_metrics.csv')
        else:
            print('   ❌ Employee metrics not found')
            
        if os.path.exists('results_export/comparison_all_heatmaps.png'):
            print('   ✅ Heatmap visualizations in results_export/')
        else:
            print('   ❌ Heatmap visualizations not found')
            
        if os.path.exists('results_export/summary_report.csv'):
            print('   ✅ Summary report in results_export/summary_report.csv')
        else:
            print('   ❌ Summary report not found')
        
        return True

    except Exception as e:
        print(f'❌ Verification failed: {e}')
        return False

def main():
    """Run the complete pipeline"""
    print("🚀 COMPLETE LINEAR SHIFT OPTIMIZATION PIPELINE")
    print("=" * 55)
    print()

    # Step 1: Run Linear Optimization and Generate Schedules
    if not run_command("python3 simple_shift_optimizer.py", "1️⃣ Running Linear Optimization and Generating Employee Schedules"):
        print("⚠️  Linear optimization failed, but continuing to results export...")
        print("   This may happen if constraints are too tight (over-demand scenarios)")
    print()

    # Step 2: Generate Analysis Results (even if optimization failed partially)
    if not run_command("python3 results_exporter.py", "2️⃣ Generating Analysis Results and Visualizations"):
        print("❌ Results export failed - cannot continue without schedule files")
        sys.exit(1)
    print()

    # Step 3: Verify Results
    if not verify_results():
        print("⚠️  Verification completed with some issues - check output above")
    else:
        print("✅ Verification completed successfully!")

    print()
    print("🎉 LINEAR OPTIMIZATION PIPELINE COMPLETE!")
    print("All files have been updated with fresh linear optimization results.")
    print("Check the results_export/ folder for analysis files and visualizations.")
    print()
    print("📊 Next steps:")
    print("   • View heatmaps in results_export/comparison_all_heatmaps.png")
    print("   • Check employee metrics in results_export/employee_metrics.csv")
    print("   • Run comparison analysis with RL algorithms")
    print()
    print("💡 Note: Linear optimization may fail in over-constrained scenarios")
    print("   (when demand > capacity). This is expected behavior.")

if __name__ == "__main__":
    main()
