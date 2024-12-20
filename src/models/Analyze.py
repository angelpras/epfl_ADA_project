import os
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast

def parse_output_log(file_path):
    """Parse the output.log file to extract training and validation losses."""
    val_losses = []
    with open(file_path, 'r') as f:
        for line in f:
            # Match epoch loss lines
            epoch_match = re.match(r"Epoch (\d+): Train Loss = ([\d.]+), Val Loss = ([\d.]+)", line)
            if epoch_match:
                val_losses.append(float(epoch_match.group(3)))
    return val_losses

def plot_validation_loss_curves(results_dir):
    """Plot validation loss curves for all experiments on a single graph."""
    # Prepare the figure
    fig = go.Figure()
    
    # Process all subdirectories in the results directory
    for subdir in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, subdir)
        
        # Check if it's a directory
        if os.path.isdir(experiment_path):
            output_log_path = os.path.join(experiment_path, "output.log")
            
            # Check if output.log exists
            if os.path.exists(output_log_path):
                print(f"Processing {subdir}...")
                
                # Parse validation losses
                val_losses = parse_output_log(output_log_path)
                
                # Add trace to the figure
                epochs = list(range(1, len(val_losses) + 1))
                fig.add_trace(go.Scatter(
                    x=epochs, 
                    y=val_losses, 
                    mode='lines+markers', 
                    name=subdir.replace("_", " ").capitalize()
                ))
    
    # Update layout
    fig.update_layout(
        title="Validation Loss Curves Across Experiments",
        xaxis_title="Epochs",
        yaxis_title="Validation Loss",
        legend_title="Experiments",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='gray'),
        yaxis=dict(showgrid=True, gridcolor='gray')
    )

    # Save the figure
    output_path = os.path.join(results_dir, "all_experiments_validation_loss.html")
    fig.write_html(output_path)
    print(f"Saved comparative validation loss curves at {output_path}")
    return fig

def plot_experiment_metrics(results_dir):
    """Parse and plot performance metrics across experiments."""
    # Prepare to store metrics
    experiment_metrics = {}
    
    # Process all subdirectories in the results directory
    for subdir in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, subdir)
        
        # Check if it's a directory
        if os.path.isdir(experiment_path):
            output_log_path = os.path.join(experiment_path, "output.log")
            if experiment_path.endswith('jaccard_similarity') or experiment_path.endswith('adamic_adar_index') or experiment_path.endswith('preferential_attachment'):
                continue
            # Check if output.log exists
            if os.path.exists(output_log_path):
                with open(output_log_path, 'r') as f:
                    for line in f:
                        if line.startswith("Test Metrics:"):
                            # Extract metrics using ast.literal_eval to safely parse the dictionary
                            metrics = ast.literal_eval(line.split(": ", 1)[1])
                            experiment_metrics[subdir] = metrics
                            break
    
    # Prepare data for plotting
    metrics_types = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create subplot figure
    fig = make_subplots(rows=1, cols=1)
    
    # Color palette for experiments (using a colorblind-friendly palette)
    colors = [
        'rgb(55,126,184)',   # blue
        'rgb(255,127,0)',    # orange
        'rgb(77,175,74)',    # green
        'rgb(228,26,28)',    # red
        'rgb(152,78,163)',   # purple
        'rgb(255,255,51)'    # yellow
    ]
    
    # Prepare for bar positioning
    num_experiments = len(experiment_metrics)
    bar_width = 0.8 / num_experiments
    
    # Track which experiments have been added to the legend
    legend_added = set()
    
    # Add bars for each metric type
    for i, metric in enumerate(metrics_types):
        for j, (exp, metrics) in enumerate(experiment_metrics.items()):
            metric_value = metrics[metric]
            
            # Determine if this is the first time this experiment appears in the legend
            show_legend = exp not in legend_added
            if show_legend:
                legend_added.add(exp)
            
            fig.add_trace(
                go.Bar(
                    x=[metric.capitalize()],
                    y=[metric_value],
                    name=exp.replace("_", " ").capitalize(),
                    showlegend=show_legend,
                    offset=j * bar_width - 0.4,
                    width=bar_width,
                    marker=dict(
                        color=colors[j % len(colors)], 
                        line=dict(width=1, color='black')
                    ),
                    opacity=0.9
                )
            )
    
    # Calculate y-axis range
    all_metric_values = []
    for metrics in experiment_metrics.values():
        all_metric_values.extend(metrics.values())
    
    y_min = max(0, min(all_metric_values) - 0.01)
    y_max = min(1, max(all_metric_values) + 0.01)
    
    # Update layout with black background
    fig.update_layout(
        title="Performance Metrics Across Experiments",
        xaxis_title="Metrics",
        yaxis_title="Metric Value",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='gray'),
        yaxis=dict(
            showgrid=True, 
            gridcolor='gray', 
            range=[y_min, y_max]
        ),
        barmode='group'
    )

    # Save the figure
    output_path = os.path.join(results_dir, "experiment_metrics_comparison.html")
    fig.write_html(output_path)
    print(f"Saved metrics comparison plot at {output_path}")
    return fig

if __name__ == "__main__":
    # This script expects a directory path as a command-line argument
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python Analyze.py <path_to_results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    plot_validation_loss_curves(results_dir)
    plot_experiment_metrics(results_dir)