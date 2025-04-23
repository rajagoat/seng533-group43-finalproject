#!/usr/bin/env python3
"""
Visualize Apache Storm benchmark results.
This script reads the results from the Excel file and generates visualizations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def generate_visualizations(excel_file, output_dir='visualizations'):
    """Generate visualizations from benchmark results."""
    # Check if input file exists
    if not os.path.exists(excel_file):
        print(f"Error: Input file {excel_file} does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the results
    print(f"Loading results from {excel_file}...")
    df = pd.read_excel(excel_file)
    print(f"Loaded {len(df)} experiment results.")

    # Convert worker_memory from strings like "2g", "4g" to numeric (in GB)
    df['worker_memory_gb'] = df['worker_memory'].str.replace('g', '', case=False).astype(int)

    # Define performance metrics:
    # Throughput: training throughput in records per second
    df['throughput'] = df['train_throughput_records_s']
    
    # Latency: average prediction latency per record (ms)
    df['latency_ms'] = (df['pred_duration_s'] / df['test_count']) * 1000
    
    # Resource Utilization: peak CPU usage (max of start and end)
    df['cpu_util_percent'] = df[['cpu_usage_start_percent', 'cpu_usage_end_percent']].max(axis=1)

    # Convert file_size to a numerical representation
    file_size_mapping = {'1k': 1000, '500k': 500000, '3m': 3000000, 'unknown': -1}
    df['file_size_numeric'] = df['file_size'].map(file_size_mapping)

    # Create file size bins if multiple sizes are present
    if len(df['file_size'].unique()) > 1:
        try:
            df['file_size_bin'] = pd.qcut(df['file_size_numeric'], 
                                        q=len(df['file_size'].unique()), 
                                        labels=['small', 'medium', 'large'][:len(df['file_size'].unique())],
                                        duplicates='drop')
        except ValueError:
            # If qcut fails (e.g., with only one unique value), use the original values
            df['file_size_bin'] = df['file_size']
    else:
        df['file_size_bin'] = df['file_size']

    # Define factors and performance metrics
    factors = ['worker_count', 'worker_memory_gb', 'parallelism']
    metrics = ['throughput', 'latency_ms', 'cpu_util_percent']

    print("Generating visualizations...")
    
    # Create line graphs for each factor vs each performance metric, grouped by file size
    for factor in factors:
        plt.figure(figsize=(16, 6))
        
        # Plot throughput
        plt.subplot(1, 3, 1)
        if 'file_size_bin' in df.columns:
            for size_bin in df['file_size_bin'].unique():
                subset = df[df['file_size_bin'] == size_bin]
                grouped = subset.groupby(factor)['throughput'].mean().reset_index()
                grouped = grouped.sort_values(by=factor)
                plt.plot(grouped[factor], grouped['throughput'], marker='o', label=f'File Size: {size_bin}')
        else:
            grouped = df.groupby(factor)['throughput'].mean().reset_index()
            grouped = grouped.sort_values(by=factor)
            plt.plot(grouped[factor], grouped['throughput'], marker='o')
            
        plt.xlabel(factor)
        plt.ylabel('Throughput (records/sec)')
        plt.title(f'Throughput vs {factor}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot latency
        plt.subplot(1, 3, 2)
        if 'file_size_bin' in df.columns:
            for size_bin in df['file_size_bin'].unique():
                subset = df[df['file_size_bin'] == size_bin]
                grouped = subset.groupby(factor)['latency_ms'].mean().reset_index()
                grouped = grouped.sort_values(by=factor)
                plt.plot(grouped[factor], grouped['latency_ms'], marker='o', label=f'File Size: {size_bin}')
        else:
            grouped = df.groupby(factor)['latency_ms'].mean().reset_index()
            grouped = grouped.sort_values(by=factor)
            plt.plot(grouped[factor], grouped['latency_ms'], marker='o')
            
        plt.xlabel(factor)
        plt.ylabel('Latency (ms/record)')
        plt.title(f'Latency vs {factor}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot CPU utilization
        plt.subplot(1, 3, 3)
        if 'file_size_bin' in df.columns:
            for size_bin in df['file_size_bin'].unique():
                subset = df[df['file_size_bin'] == size_bin]
                grouped = subset.groupby(factor)['cpu_util_percent'].mean().reset_index()
                grouped = grouped.sort_values(by=factor)
                plt.plot(grouped[factor], grouped['cpu_util_percent'], marker='o', label=f'File Size: {size_bin}')
        else:
            grouped = df.groupby(factor)['cpu_util_percent'].mean().reset_index()
            grouped = grouped.sort_values(by=factor)
            plt.plot(grouped[factor], grouped['cpu_util_percent'], marker='o')
            
        plt.xlabel(factor)
        plt.ylabel('CPU Utilization (%)')
        plt.title(f'CPU Utilization vs {factor}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.suptitle(f"Impact of {factor} on Performance Metrics", fontsize=16, y=1.05)
        
        # Save figure
        output_file = os.path.join(output_dir, f"storm_impact_of_{factor}.png")
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    # Create a bar chart comparing metrics across file sizes
    if len(df['file_size'].unique()) > 1:
        plt.figure(figsize=(15, 5))
        
        # Throughput by file size
        plt.subplot(1, 3, 1)
        throughput_by_size = df.groupby('file_size')['throughput'].mean()
        throughput_by_size.plot(kind='bar', color='skyblue')
        plt.title('Avg Throughput by File Size')
        plt.ylabel('Throughput (records/sec)')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Latency by file size
        plt.subplot(1, 3, 2)
        latency_by_size = df.groupby('file_size')['latency_ms'].mean()
        latency_by_size.plot(kind='bar', color='salmon')
        plt.title('Avg Latency by File Size')
        plt.ylabel('Latency (ms/record)')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # CPU utilization by file size
        plt.subplot(1, 3, 3)
        cpu_by_size = df.groupby('file_size')['cpu_util_percent'].mean()
        cpu_by_size.plot(kind='bar', color='lightgreen')
        plt.title('Avg CPU Utilization by File Size')
        plt.ylabel('CPU Utilization (%)')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle("Performance Metrics by File Size", fontsize=16, y=1.05)
        
        # Save figure
        output_file = os.path.join(output_dir, "storm_metrics_by_file_size.png")
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    # Create a 3D surface plot for parallelism vs worker_count vs throughput
    if len(df['parallelism'].unique()) > 1 and len(df['worker_count'].unique()) > 1:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Group by parallelism and worker_count and calculate mean throughput
            for file_size in df['file_size'].unique():
                subset = df[df['file_size'] == file_size]
                
                # Create mesh grid
                parallelism_values = sorted(subset['parallelism'].unique())
                worker_count_values = sorted(subset['worker_count'].unique())
                
                # Create surface plot only if we have enough data points
                if len(parallelism_values) > 1 and len(worker_count_values) > 1:
                    # Create a pivot table
                    pivot = subset.pivot_table(
                        index='parallelism', 
                        columns='worker_count', 
                        values='throughput',
                        aggfunc='mean'
                    )
                    
                    # Create meshgrid for 3D plot
                    X, Y = np.meshgrid(pivot.columns, pivot.index)
                    Z = pivot.values
                    
                    # Plot surface
                    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
                    
                    # Add colorbar
                    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Throughput (records/sec)')
                    
                    ax.set_xlabel('Worker Count')
                    ax.set_ylabel('Parallelism')
                    ax.set_zlabel('Throughput (records/sec)')
                    ax.set_title(f'Throughput Surface Plot (File Size: {file_size})')
                    
                    # Save figure
                    output_file = os.path.join(output_dir, f"storm_throughput_surface_{file_size}.png")
                    plt.savefig(output_file, bbox_inches='tight')
                    print(f"Saved: {output_file}")
            
            plt.close()
        except Exception as e:
            print(f"Could not create 3D plot: {str(e)}")

    # Create a radar chart comparing the best configuration for each file size
    try:
        if len(df['file_size'].unique()) > 1:
            plt.figure(figsize=(10, 8))
            
            # Find best configuration (highest throughput) for each file size
            best_configs = df.loc[df.groupby('file_size')['throughput'].idxmax()]
            
            # Normalize metrics for radar chart
            metrics = ['throughput', 'rmse', 'r2', 'cpu_util_percent']
            normalized = best_configs[metrics].copy()
            
            # Normalize (0-1 scale) - except for R² which is already 0-1
            for metric in metrics:
                if metric != 'r2':
                    min_val = best_configs[metric].min()
                    max_val = best_configs[metric].max()
                    if max_val > min_val:
                        normalized[metric] = (best_configs[metric] - min_val) / (max_val - min_val)
            
            # For RMSE and CPU, lower is better, so invert
            normalized['rmse'] = 1 - normalized['rmse']
            normalized['cpu_util_percent'] = 1 - normalized['cpu_util_percent']
            
            # Create radar chart
            categories = ['Throughput', 'Model Accuracy (inv. RMSE)', 'R²', 'CPU Efficiency']
            
            # Number of variables
            N = len(categories)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create subplot
            ax = plt.subplot(111, polar=True)
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], categories)
            
            # Draw the chart for each file size
            for i, (idx, row) in enumerate(best_configs.iterrows()):
                values = normalized.loc[idx, metrics].tolist()
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"File Size: {row['file_size']}")
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title("Performance Comparison of Best Configurations by File Size")
            
            # Save figure
            output_file = os.path.join(output_dir, "storm_radar_comparison.png")
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
    except Exception as e:
        print(f"Could not create radar chart: {str(e)}")

    # Create a bar chart showing model quality metrics
    plt.figure(figsize=(12, 5))
    
    # RMSE by configuration
    plt.subplot(1, 2, 1)
    for size in df['file_size'].unique():
        subset = df[df['file_size'] == size]
        best_idx = subset['throughput'].idxmax()
        best_config = subset.loc[best_idx]
        
        config_label = f"{size}: W{best_config['worker_count']}-M{best_config['worker_memory_gb']}-P{best_config['parallelism']}"
        plt.bar(config_label, best_config['rmse'], alpha=0.7)
    
    plt.title('RMSE for Best Throughput Configurations')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # R² by configuration
    plt.subplot(1, 2, 2)
    for size in df['file_size'].unique():
        subset = df[df['file_size'] == size]
        best_idx = subset['throughput'].idxmax()
        best_config = subset.loc[best_idx]
        
        config_label = f"{size}: W{best_config['worker_count']}-M{best_config['worker_memory_gb']}-P{best_config['parallelism']}"
        plt.bar(config_label, best_config['r2'], alpha=0.7)
    
    plt.title('R² for Best Throughput Configurations')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, "storm_model_quality.png")
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    print(f"All visualizations saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Visualize Apache Storm benchmark results')
    parser.add_argument('--input', default='storm_experiment_results.xlsx',
                        help='Input Excel file with benchmark results')
    parser.add_argument('--output-dir', default='visualizations',
                        help='Directory to save visualizations')
    args = parser.parse_args()
    
    generate_visualizations(args.input, args.output_dir)

if __name__ == "__main__":
    main()