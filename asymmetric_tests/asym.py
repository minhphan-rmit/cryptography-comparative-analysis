import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f_oneway

# Data extracted from tables
data = {
    "Algorithm": ["Diffie-Hellman", "RSA", "ElGamal", "ECC"],
    "Encryption Time (5KB)": [0.002, 0.01519, 0.00202, 0.00202],
    "Encryption Time (20KB)": [0.009, 0.04463, 0.0094, 0.0094],
    "Encryption Time (50KB)": [0.019, 0.09339, 0.056837, 0.056837],
    "Decryption Time (5KB)": [0.001, 0.01724, 0.003622, 0.003622],
    "Decryption Time (20KB)": [0.008, 0.04629, 0.012963, 0.012963],
    "Decryption Time (50KB)": [0.022, 0.10927, 0.031949, 0.031949],
    "Throughput (5KB)": [2500, 367, 2475.25, 2475.25],
    "Throughput (20KB)": [2222.22, 477, 2127.65, 2127.65],
    "Throughput (50KB)": [2631.5789, 592, 879.7, 879.7],
    "Memory Utilization (5KB)": [5.123, 5.581, 5581, 5581],
    "Memory Utilization (20KB)": [20.361, 22.178, 22178, 22178],
    "Memory Utilization (50KB)": [50.483, 53.369, 55369, 55369],
}

# Converting data to DataFrame
df = pd.DataFrame(data)

# Set font style for IEEE style
plt.rcParams['font.family'] = 'serif'

# Create output directory
output_dir = "algorithm_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Metrics and sizes
metrics = ["Encryption Time", "Decryption Time", "Throughput", "Memory Utilization"]
sizes = ["5KB", "20KB", "50KB"]

# Generate and save plots
for metric in metrics:
    fig, ax = plt.subplots(figsize=(6, 4))
    for size in sizes:
        ax.plot(df["Algorithm"], df[f"{metric} ({size})"], label=size, marker='o')
    ax.set_title(f"{metric} Comparison", fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.legend(title="File Size", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)
    plot_filename = os.path.join(output_dir, f"{metric.replace(' ', '_').lower()}_comparison.png")
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)

# Conduct ANOVA analysis
anova_results = {}
for metric in metrics:
    metric_data = [
        df[f"{metric} (5KB)"].values,
        df[f"{metric} (20KB)"].values,
        df[f"{metric} (50KB)"].values,
    ]
    f_stat, p_value = f_oneway(*metric_data)
    anova_results[metric] = {"F-Statistic": f_stat, "P-Value": p_value}

# Save ANOVA results to a text file
anova_output_path = os.path.join(output_dir, "anova_results.txt")
with open(anova_output_path, "w") as f:
    f.write("ANOVA Analysis Results\n")
    f.write("="*30 + "\n\n")
    for metric, result in anova_results.items():
        f.write(f"{metric}:\n")
        f.write(f"  F-Statistic: {result['F-Statistic']:.4f}\n")
        f.write(f"  P-Value: {result['P-Value']:.4e}\n")
        f.write("\n")

# Save the data to a CSV file
csv_output_path = os.path.join(output_dir, "algorithm_data.csv")
df.to_csv(csv_output_path, index=False)

print("Plots, ANOVA results, and data saved successfully!")