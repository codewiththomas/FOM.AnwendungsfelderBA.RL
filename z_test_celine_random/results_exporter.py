"""
Results Export and Visualization for Employee Shift Scheduling - RANDOM AGENT BENCHMARK
Creates CSV exports and heatmaps for analysis of Monte-Carlo Random Agent results
"""

import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization libraries not available: {e}")
    print("CSV exports will still work, but heatmaps will be skipped.")
    VISUALIZATION_AVAILABLE = False
import os
from pathlib import Path

class ResultsExporter:
    def __init__(self, schedules_dir, forecast_file, employees_file, output_dir):
        self.schedules_dir = schedules_dir
        self.forecast_file = forecast_file
        self.employees_file = employees_file
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.forecast_df = pd.read_csv(forecast_file)
        self.employees_df = pd.read_csv(employees_file)
        
        # Load all employee schedules
        self.schedules = {}
        for filename in os.listdir(schedules_dir):
            if filename.endswith('_schedule.csv'):
                emp_id = filename.replace('_schedule.csv', '')
                self.schedules[emp_id] = pd.read_csv(os.path.join(schedules_dir, filename))
        
        self.days = self.forecast_df['weekday'].tolist()
        self.shifts = ['shift_morning', 'shift_afternoon', 'shift_evening']
    
    def export_all_results(self):
        """
        Export all results: CSV files and heatmaps
        """
        print("Exporting scheduling results...")
        
        # 1. Export performance metrics to CSV
        self._export_performance_metrics()
        
        # 2. Create aggregated results CSV
        self._export_aggregated_results()
        
        # 3. Export individual employee metrics
        self._export_employee_metrics()
        
        # 4. Create heatmaps (if visualization libraries are available)
        if VISUALIZATION_AVAILABLE:
            self._create_heatmaps()
        else:
            print("⚠️ Skipping heatmap creation - visualization libraries not available")
        
        print(f"✅ All results exported to: {self.output_dir}")
    
    def _export_performance_metrics(self):
        """
        Export performance metrics to CSV
        """
        metrics = []
        
        # Overall metrics
        total_shifts_required = self.forecast_df[self.shifts].sum().sum()
        total_shifts_assigned = sum(schedule[self.shifts].sum().sum() for schedule in self.schedules.values())
        
        metrics.append({
            'metric': 'Total Shifts Required',
            'value': total_shifts_required,
            'description': 'Total number of shifts needed according to forecast'
        })
        
        metrics.append({
            'metric': 'Total Shifts Assigned',
            'value': total_shifts_assigned,
            'description': 'Total number of shifts assigned to employees'
        })
        
        metrics.append({
            'metric': 'Coverage Rate',
            'value': f"{(total_shifts_assigned / total_shifts_required * 100):.1f}%",
            'description': 'Percentage of required shifts that are covered'
        })
        
        # Employee metrics
        for _, emp_row in self.employees_df.iterrows():
            emp_id = emp_row['employee_id']
            required = emp_row['shifts']
            assigned = self.schedules[emp_id][self.shifts].sum().sum()
            
            metrics.append({
                'metric': f'{emp_id} Shifts Required',
                'value': required,
                'description': f'Number of shifts required for {emp_id}'
            })
            
            metrics.append({
                'metric': f'{emp_id} Shifts Assigned',
                'value': assigned,
                'description': f'Number of shifts assigned to {emp_id}'
            })
            
            # Working days
            working_days = 0
            for _, row in self.schedules[emp_id].iterrows():
                if row[self.shifts].sum() > 0:
                    working_days += 1
            
            metrics.append({
                'metric': f'{emp_id} Working Days',
                'value': working_days,
                'description': f'Number of days {emp_id} works'
            })
        
        # Save to CSV
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(self.output_dir, 'performance_metrics.csv'), index=False)
        print("✅ Performance metrics exported to performance_metrics.csv")
    
    def _export_aggregated_results(self):
        """
        Create aggregated results in forecast format
        """
        # Create aggregated schedule (sum of all employees)
        aggregated_data = []
        
        for day in self.days:
            row = {'weekday': day}
            
            for shift in self.shifts:
                total_assigned = 0
                for emp_id, schedule in self.schedules.items():
                    day_row = schedule[schedule['weekday'] == day]
                    if not day_row.empty:
                        total_assigned += day_row[shift].iloc[0]
                
                row[shift] = total_assigned
            
            aggregated_data.append(row)
        
        # Save aggregated results
        aggregated_df = pd.DataFrame(aggregated_data)
        aggregated_df.to_csv(os.path.join(self.output_dir, 'aggregated_schedule.csv'), index=False)
        print("✅ Aggregated schedule exported to aggregated_schedule.csv")
        
        # Create comparison with forecast
        comparison_data = []
        
        for day in self.days:
            forecast_row = self.forecast_df[self.forecast_df['weekday'] == day].iloc[0]
            aggregated_row = aggregated_df[aggregated_df['weekday'] == day].iloc[0]
            
            row = {
                'weekday': day,
                'forecast_morning': forecast_row['shift_morning'],
                'assigned_morning': aggregated_row['shift_morning'],
                'diff_morning': aggregated_row['shift_morning'] - forecast_row['shift_morning'],
                'forecast_afternoon': forecast_row['shift_afternoon'],
                'assigned_afternoon': aggregated_row['shift_afternoon'],
                'diff_afternoon': aggregated_row['shift_afternoon'] - forecast_row['shift_afternoon'],
                'forecast_evening': forecast_row['shift_evening'],
                'assigned_evening': aggregated_row['shift_evening'],
                'diff_evening': aggregated_row['shift_evening'] - forecast_row['shift_evening']
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(self.output_dir, 'forecast_vs_assigned.csv'), index=False)
        print("✅ Forecast vs Assigned comparison exported to forecast_vs_assigned.csv")
        
        return aggregated_df
    
    def _export_employee_metrics(self):
        """
        Export detailed employee metrics
        """
        employee_metrics = []
        
        for emp_id, schedule in self.schedules.items():
            # Get employee info
            emp_row = self.employees_df[self.employees_df['employee_id'] == emp_id].iloc[0]
            required_shifts = emp_row['shifts']
            
            # Calculate metrics
            total_assigned = schedule[self.shifts].sum().sum()
            working_days = sum(1 for _, row in schedule.iterrows() if row[self.shifts].sum() > 0)
            
            # Daily workload analysis
            daily_workloads = []
            single_shift_days = 0
            double_shift_days = 0
            
            for _, row in schedule.iterrows():
                daily_shifts = row[self.shifts].sum()
                daily_workloads.append(daily_shifts)
                
                if daily_shifts == 1:
                    single_shift_days += 1
                elif daily_shifts == 2:
                    double_shift_days += 1
            
            avg_daily_workload = np.mean([w for w in daily_workloads if w > 0]) if working_days > 0 else 0
            workload_std = np.std([w for w in daily_workloads if w > 0]) if working_days > 0 else 0
            
            # Shift type distribution
            morning_shifts = schedule['shift_morning'].sum()
            afternoon_shifts = schedule['shift_afternoon'].sum()
            evening_shifts = schedule['shift_evening'].sum()
            
            employee_metrics.append({
                'employee_id': emp_id,
                'required_shifts': required_shifts,
                'assigned_shifts': total_assigned,
                'shift_difference': total_assigned - required_shifts,
                'working_days': working_days,
                'rest_days': len(self.days) - working_days,
                'single_shift_days': single_shift_days,
                'double_shift_days': double_shift_days,
                'avg_daily_workload': round(avg_daily_workload, 2),
                'workload_std_dev': round(workload_std, 2),
                'morning_shifts': morning_shifts,
                'afternoon_shifts': afternoon_shifts,
                'evening_shifts': evening_shifts,
                'utilization_rate': f"{(total_assigned / required_shifts * 100):.1f}%" if required_shifts > 0 else "0%"
            })
        
        employee_metrics_df = pd.DataFrame(employee_metrics)
        employee_metrics_df.to_csv(os.path.join(self.output_dir, 'employee_metrics.csv'), index=False)
        print("✅ Employee metrics exported to employee_metrics.csv")
    
    def _create_heatmaps(self):
        """
        Create heatmaps as requested
        """
        # Prepare data for heatmaps
        forecast_matrix = []
        assigned_matrix = []
        
        for day in self.days:
            # Forecast data
            forecast_row = self.forecast_df[self.forecast_df['weekday'] == day].iloc[0]
            forecast_matrix.append([
                forecast_row['shift_morning'],
                forecast_row['shift_afternoon'],
                forecast_row['shift_evening']
            ])
            
            # Assigned data (sum of all employees)
            assigned_row = [0, 0, 0]
            for emp_id, schedule in self.schedules.items():
                day_row = schedule[schedule['weekday'] == day]
                if not day_row.empty:
                    assigned_row[0] += day_row['shift_morning'].iloc[0]
                    assigned_row[1] += day_row['shift_afternoon'].iloc[0]
                    assigned_row[2] += day_row['shift_evening'].iloc[0]
            
            assigned_matrix.append(assigned_row)
        
        # Convert to numpy arrays
        forecast_matrix = np.array(forecast_matrix)
        assigned_matrix = np.array(assigned_matrix)
        diff_matrix = assigned_matrix - forecast_matrix
        
        # Create labels
        days_labels = [day.capitalize() for day in self.days]
        shift_labels = ['Morning', 'Afternoon', 'Evening']
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # 1. Forecast heatmap
        sns.heatmap(forecast_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=shift_labels,
                   yticklabels=days_labels,
                   ax=axes[0],
                   cbar_kws={'label': 'Number of Shifts'})
        axes[0].set_title('Forecast - Required Shifts', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Shift Type')
        axes[0].set_ylabel('Day of Week')
        
        # 2. Assigned heatmap
        sns.heatmap(assigned_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Greens',
                   xticklabels=shift_labels,
                   yticklabels=days_labels,
                   ax=axes[1],
                   cbar_kws={'label': 'Number of Shifts'})
        axes[1].set_title('Assigned - Actual Employee Assignments', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Shift Type')
        axes[1].set_ylabel('Day of Week')
        
        # 3. Difference heatmap
        sns.heatmap(diff_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='RdBu_r',
                   center=0,
                   xticklabels=shift_labels,
                   yticklabels=days_labels,
                   ax=axes[2],
                   cbar_kws={'label': 'Difference (Assigned - Required)'})
        axes[2].set_title('Difference - Assigned vs Required Shifts', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Shift Type')
        axes[2].set_ylabel('Day of Week')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison_all_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Heatmaps saved to comparison_all_heatmaps.png")
        
        # Also create separate heatmaps
        self._create_separate_heatmaps(forecast_matrix, assigned_matrix, diff_matrix, 
                                     days_labels, shift_labels)
    
    def _create_separate_heatmaps(self, forecast_matrix, assigned_matrix, diff_matrix, 
                                 days_labels, shift_labels):
        """
        Create separate heatmap files
        """
        # Individual heatmaps
        heatmaps = [
            (forecast_matrix, 'Forecast - Required Shifts', 'Blues', 'forecast_heatmap.png'),
            (assigned_matrix, 'Assigned - Employee Assignments', 'Greens', 'assigned_heatmap.png'),
            (diff_matrix, 'Difference (Assigned - Required)', 'RdBu_r', 'difference_heatmap.png')
        ]
        
        for matrix, title, cmap, filename in heatmaps:
            plt.figure(figsize=(8, 6))
            
            if 'Difference' in title:
                sns.heatmap(matrix, 
                           annot=True, 
                           fmt='d', 
                           cmap=cmap,
                           center=0,
                           xticklabels=shift_labels,
                           yticklabels=days_labels,
                           cbar_kws={'label': 'Difference'})
            else:
                sns.heatmap(matrix, 
                           annot=True, 
                           fmt='d', 
                           cmap=cmap,
                           xticklabels=shift_labels,
                           yticklabels=days_labels,
                           cbar_kws={'label': 'Number of Shifts'})
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Shift Type')
            plt.ylabel('Day of Week')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Individual heatmaps saved")
    
    def create_summary_report(self):
        """
        Create a summary report CSV
        """
        summary_data = []
        
        # Overall summary
        total_required = self.forecast_df[self.shifts].sum().sum()
        total_assigned = sum(schedule[self.shifts].sum().sum() for schedule in self.schedules.values())
        
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Total Required Shifts',
            'Value': total_required
        })
        
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Total Assigned Shifts',
            'Value': total_assigned
        })
        
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Perfect Match',
            'Value': 'Yes' if total_required == total_assigned else 'No'
        })
        
        # Daily summary
        for day in self.days:
            forecast_row = self.forecast_df[self.forecast_df['weekday'] == day].iloc[0]
            daily_required = forecast_row[self.shifts].sum()
            
            daily_assigned = 0
            for emp_id, schedule in self.schedules.items():
                day_row = schedule[schedule['weekday'] == day]
                if not day_row.empty:
                    daily_assigned += day_row[self.shifts].sum()
            
            summary_data.append({
                'Category': f'Daily - {day.capitalize()}',
                'Metric': 'Required Shifts',
                'Value': daily_required
            })
            
            summary_data.append({
                'Category': f'Daily - {day.capitalize()}',
                'Metric': 'Assigned Shifts',
                'Value': daily_assigned
            })
        
        # Employee summary
        for emp_id in self.schedules.keys():
            emp_row = self.employees_df[self.employees_df['employee_id'] == emp_id].iloc[0]
            required = emp_row['shifts']
            assigned = self.schedules[emp_id][self.shifts].sum().sum()
            
            summary_data.append({
                'Category': f'Employee - {emp_id}',
                'Metric': 'Required Shifts',
                'Value': required
            })
            
            summary_data.append({
                'Category': f'Employee - {emp_id}',
                'Metric': 'Assigned Shifts',
                'Value': assigned
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, 'summary_report.csv'), index=False)
        print("✅ Summary report saved to summary_report.csv")


def main():
    # File paths
    base_dir = Path(__file__).parent.parent
    forecast_file = base_dir / "data" / "forecast.csv"
    employees_file = base_dir / "data" / "employees.csv"
    schedules_dir = Path(__file__).parent / "employee_schedules"
    output_dir = Path(__file__).parent / "results_export"
    
    # Create exporter
    exporter = ResultsExporter(schedules_dir, forecast_file, employees_file, output_dir)
    
    # Export all results
    exporter.export_all_results()
    
    # Create summary report
    exporter.create_summary_report()
    
    print("\n" + "="*50)
    print("RESULTS EXPORT COMPLETE")
    print("="*50)
    print(f"Check the '{output_dir}' folder for:")
    print("📊 CSV Files:")
    print("  • performance_metrics.csv - Overall performance metrics")
    print("  • aggregated_schedule.csv - Sum of all employee shifts")
    print("  • forecast_vs_assigned.csv - Comparison with forecast")
    print("  • employee_metrics.csv - Individual employee analysis")
    print("  • summary_report.csv - Executive summary")
    print("\n📈 Visualizations:")
    print("  • comparison_all_heatmaps.png - Combined heatmaps")
    print("  • forecast_heatmap.png - Required shifts")
    print("  • assigned_heatmap.png - Assigned shifts") 
    print("  • difference_heatmap.png - Difference analysis")


if __name__ == "__main__":
    main()
