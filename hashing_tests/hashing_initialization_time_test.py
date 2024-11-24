import os
import hashlib
import pandas as pd
import time
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set global style for plots
rcParams['font.family'] = 'serif'  # Set serif font
rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'gray'])  # Grayscale colors
rcParams['text.color'] = 'black'

def measure_initialization_time(algorithm, iterations=1000):
    """
    Measure the time taken to initialize a hashing algorithm repeatedly.

    Parameters:
        algorithm (callable): Hashing algorithm constructor (e.g., hashlib.md5).
        iterations (int): Number of times to initialize the algorithm.

    Returns:
        list: List of initialization times (in microseconds).
    """
    times = []
    for _ in range(iterations):
        start_time = time.time()  # Record start time
        algo_instance = algorithm()  # Initialize the hashing algorithm
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        times.append(elapsed_time * 1e6)  # Convert seconds to microseconds
    return times

def test_initialization_time(algorithms, iterations=1000):
    """
    Test initialization time for multiple hashing algorithms.

    Parameters:
        algorithms (dict): Dictionary of algorithm names and constructors.
        iterations (int): Number of iterations for each algorithm.

    Returns:
        pd.DataFrame: DataFrame containing initialization time results.
    """
    results = []

    for algo_name, algo_constructor in algorithms.items():
        times = measure_initialization_time(algo_constructor, iterations)  # Measure times for the algorithm
        for time_value in times:
            results.append({"Algorithm": algo_name, "Initialization Time (µs)": time_value})

    return pd.DataFrame(results)

# Define the hashing algorithms to test
algorithms = {
    "MD5": hashlib.md5,
    "SHA-1": hashlib.sha1,
    "SHA-256": hashlib.sha256,
    "SHA3-256": hashlib.sha3_256,
    "BLAKE2b": hashlib.blake2b,
}

# Create output directories
results_folder = "hashing_results"  # Folder for storing CSV results
visualizations_folder = "hashing_visualization"  # Folder for storing plots
anova_results_folder = "hashing_anova_result"  # Folder for storing ANOVA results
os.makedirs(results_folder, exist_ok=True)  # Ensure results folder exists
os.makedirs(visualizations_folder, exist_ok=True)  # Ensure visualizations folder exists
os.makedirs(anova_results_folder, exist_ok=True)  # Ensure ANOVA results folder exists

# Run the initialization time tests
iterations = 1000  # Number of iterations per algorithm
initialization_df = test_initialization_time(algorithms, iterations)

# Perform ANOVA test to compare initialization times across algorithms
anova_results = f_oneway(
    *[initialization_df[initialization_df["Algorithm"] == algo]["Initialization Time (µs)"].values
      for algo in algorithms.keys()]
)

# Save ANOVA results to a text file
anova_results_path = os.path.join(anova_results_folder, "anova_results.txt")
with open(anova_results_path, "w") as f:
    f.write(f"ANOVA Results:\n")
    f.write(f"F-statistic: {anova_results.statistic:.4f}\n")
    f.write(f"p-value: {anova_results.pvalue:.4e}\n")

# Calculate mean and standard deviation for each algorithm
summary_stats = initialization_df.groupby("Algorithm")["Initialization Time (µs)"].agg(["mean", "std"])

# Save results to a CSV for further analysis
initialization_csv_path = os.path.join(results_folder, "hashing_initialization_times.csv")
initialization_df.to_csv(initialization_csv_path, index=False)  # Save detailed times

# Save summary statistics to a CSV
summary_csv_path = os.path.join(results_folder, "hashing_initialization_time_summary.csv")
summary_stats.to_csv(summary_csv_path, index=True)  # Save summary statistics

# Sort the summary statistics by mean initialization time
summary_stats = summary_stats.sort_values(by="mean")

# Create a horizontal bar chart in grayscale
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

# Plot mean initialization time without error bars
ax.barh(
    summary_stats.index,  # Algorithm names
    summary_stats["mean"],  # Mean initialization times
    color="gray", edgecolor="black", alpha=0.7  # Grayscale with distinct edges
)

# Add chart titles and labels
ax.set_title("Mean Initialization Time by Algorithm", fontsize=14)
ax.set_xlabel("Initialization Time (µs)", fontsize=12)
ax.set_ylabel("Algorithm", fontsize=12)

# Add gridlines for better readability
ax.grid(axis="x", linestyle="--", alpha=0.5)

# Save the chart as a high-resolution image
plot_path = os.path.join(visualizations_folder, "mean_initialization_time_plot.png")
plt.savefig(plot_path, dpi=300)  # Save plot to file

# Display the chart
# plt.show()

# Print file paths for reference
print(f"\nInitialization times saved to: {initialization_csv_path}")
print(f"Summary statistics saved to: {summary_csv_path}")
print(f"Visualization saved to: {plot_path}")
print(f"ANOVA results saved to: {anova_results_path}")
