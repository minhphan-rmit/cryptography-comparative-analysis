import os
import hashlib
import pandas as pd
import time
from scipy.stats import f_oneway
from tabulate import tabulate
import matplotlib.pyplot as plt

# Function to measure hashing speed for a file
def measure_hashing_speed(file_path, algorithm):
    """
    Measure the time taken to hash a file with the specified algorithm.
    
    Parameters:
        file_path (str): Path to the file to be hashed.
        algorithm (callable): Hashing algorithm constructor (e.g., hashlib.md5).
    
    Returns:
        float: Time taken to hash the file in seconds.
    """
    hasher = algorithm()  # Initialize the hashing algorithm
    start_time = time.time()
    
    # Read the file in chunks and update the hash
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):  # Read file in 8KB chunks
            hasher.update(chunk)
    
    elapsed_time = time.time() - start_time
    return elapsed_time

# Function to test hashing speed for multiple algorithms
def test_hashing_algorithms(test_folder, algorithms):
    """
    Test hashing speed for multiple algorithms on files in the test folder.
    
    Parameters:
        test_folder (str): Folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.
    
    Returns:
        pd.DataFrame: DataFrame containing hashing speed results.
    """
    results = []

    # Iterate through all files in the test folder
    for file_name in sorted(os.listdir(test_folder)):
        file_path = os.path.join(test_folder, file_name)
        file_size = os.path.getsize(file_path)

        # Measure hashing time for each algorithm
        for algo_name, algo_constructor in algorithms.items():
            time_taken = measure_hashing_speed(file_path, algo_constructor)
            results.append({
                "File Size (bytes)": file_size,
                "Algorithm": algo_name,
                "Time Taken (s)": time_taken
            })

    return pd.DataFrame(results)

# Define hashing algorithms to test
algorithms = {
    "MD5": hashlib.md5,
    "SHA-1": hashlib.sha1,
    "SHA-256": hashlib.sha256,
    "SHA3-256": hashlib.sha3_256,
    "BLAKE2b": hashlib.blake2b,
}

# Path to test data folder
test_data_folder = "test_data"

# Run hashing speed tests and collect results
results_df = test_hashing_algorithms(test_data_folder, algorithms)

# Perform ANOVA analysis on the results
anova_results = f_oneway(
    *[results_df[results_df['Algorithm'] == algo]["Time Taken (s)"].values for algo in algorithms.keys()]
)

# Save ANOVA results to a text file with a meaningful name in a dedicated folder
os.makedirs("hashing_anova_result", exist_ok=True)
anova_file_path = "hashing_anova_result/hash_algorithm_anova_results.txt"
with open(anova_file_path, "w") as f:
    f.write(f"ANOVA Results:\n")
    f.write(f"F-statistic: {anova_results.statistic:.4f}\n")
    f.write(f"p-value: {anova_results.pvalue:.4e}\n")

# Compute average hashing time per file size and algorithm
avg_per_file_size = results_df.groupby(["File Size (bytes)", "Algorithm"])["Time Taken (s)"].mean().unstack()

# Save detailed results to a CSV in a dedicated folder
os.makedirs("hashing_results", exist_ok=True)
results_df.to_csv("hashing_results/hashing_speed_results.csv", index=False)

# Save average statistics to another CSV
avg_per_file_size.to_csv("hashing_results/hashing_speed_statistics.csv", index=True)

# Convert file sizes to readable format: KB for <1 MB, MB for >=1 MB
def readable_file_size(size_in_bytes):
    if size_in_bytes < 1024 ** 2:  # Less than 1 MB
        return f"{size_in_bytes / 1024:.1f} KB"
    else:
        return f"{size_in_bytes / (1024 ** 2):.1f} MB"

avg_per_file_size.index = [readable_file_size(size) for size in avg_per_file_size.index]

# Plot average hashing time per file size
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each algorithm's performance
for algorithm in avg_per_file_size.columns:
    ax.plot(
        avg_per_file_size.index,
        avg_per_file_size[algorithm],
        marker='o',
        label=algorithm,
        linewidth=1.5
    )

# Format plot
ax.set_title("Average Hashing Time per File Size", fontsize=14)
ax.set_xlabel("File Size", fontsize=12)
ax.set_ylabel("Average Time (seconds)", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.legend(title="Algorithm", fontsize=10, title_fontsize=12, loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot to a visualization folder
os.makedirs("hashing_visualization", exist_ok=True)
plt.savefig("hashing_visualization/average_hashing_time_per_file_size.png", dpi=300)

# Display the plot
plt.show()
