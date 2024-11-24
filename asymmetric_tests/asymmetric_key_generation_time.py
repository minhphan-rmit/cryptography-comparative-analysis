import os
import time
from cryptography.hazmat.primitives.asymmetric import rsa, ec, dsa
import pandas as pd
from tabulate import tabulate
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure plot style for IEEE compliance
rcParams['font.family'] = 'serif'  # Use serif font

# Define the algorithms to test
algorithms_to_test = {
    "RSA-2048": lambda: rsa.generate_private_key(public_exponent=65537, key_size=2048),
    "ECC-P256": lambda: ec.generate_private_key(ec.SECP256R1()),
    "DSA-2048": lambda: dsa.generate_private_key(key_size=2048),
}

# Create result folders
results_folder = "asymmetric_results"
visualizations_folder = "asymmetric_visualization"
anova_results_folder = "asymmetric_anova_results"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(visualizations_folder, exist_ok=True)
os.makedirs(anova_results_folder, exist_ok=True)

def measure_key_generation_time(algorithms, iterations=10):
    """
    Measure the key generation time for asymmetric algorithms.

    Parameters:
        algorithms (dict): Dictionary of algorithm names and constructors.
        iterations (int): Number of iterations for each algorithm.

    Returns:
        pd.DataFrame: DataFrame containing key generation times.
    """
    results = []

    for algo_name, algo_fn in algorithms.items():
        for _ in range(iterations):
            start_time = time.time()
            algo_fn()  # Generate key pair
            elapsed_time = time.time() - start_time
            results.append({"Algorithm": algo_name, "Key Generation Time (s)": elapsed_time})

    return pd.DataFrame(results)

# Measure key generation times
iterations = 10
key_generation_df = measure_key_generation_time(algorithms_to_test, iterations)

# Save full statistics
full_results_path = os.path.join(results_folder, "key_generation_full_results.csv")
key_generation_df.to_csv(full_results_path, index=False)

# Calculate average times grouped by algorithm
average_times = key_generation_df.groupby("Algorithm")["Key Generation Time (s)"].agg(["mean", "std"])

# Save summary statistics
summary_results_path = os.path.join(results_folder, "key_generation_summary_results.csv")
average_times.to_csv(summary_results_path)

# Perform ANOVA analysis
anova_results = f_oneway(
    *[key_generation_df[key_generation_df["Algorithm"] == algo]["Key Generation Time (s)"].values
      for algo in algorithms_to_test.keys()]
)

# Save ANOVA results
anova_results_path = os.path.join(anova_results_folder, "key_generation_anova_results.txt")
with open(anova_results_path, "w") as f:
    f.write("ANOVA Results:\n")
    f.write(f"F-statistic: {anova_results.statistic:.4f}\n")
    f.write(f"p-value: {anova_results.pvalue:.4e}\n")

# Visualize average key generation times
plt.figure(figsize=(8, 6))
plt.bar(average_times.index, average_times["mean"], yerr=average_times["std"], capsize=5, color="skyblue")
plt.title("Average Key Generation Time by Algorithm", fontsize=14)
plt.xlabel("Algorithm", fontsize=12)
plt.ylabel("Time (s)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Save the visualization
visualization_path = os.path.join(visualizations_folder, "key_generation_time.png")
plt.savefig(visualization_path, dpi=300)
plt.close()

# Print completion message
print(f"\nFull results saved to: {full_results_path}")
print(f"Summary statistics saved to: {summary_results_path}")
print(f"ANOVA results saved to: {anova_results_path}")
print(f"Visualization saved to: {visualization_path}")
