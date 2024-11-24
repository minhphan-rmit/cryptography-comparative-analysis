import psutil
import hashlib
import os
import pandas as pd
import time
from tabulate import tabulate
import matplotlib.pyplot as plt

def measure_resource_usage(file_path, algorithm):
    """
    Measure CPU, peak memory, and average memory usage during hashing.

    Parameters:
        file_path (str): Path to the file to hash.
        algorithm (callable): Hashing algorithm constructor (e.g., hashlib.md5).

    Returns:
        dict: Resource usage statistics (CPU %, peak memory, average memory, duration).
    """
    # Initialize process monitoring
    process = psutil.Process(os.getpid())
    cpu_usage = []
    memory_usage = []

    hasher = algorithm()  # Initialize hashing algorithm
    start_time = time.time()

    # Process file in chunks and monitor resource usage
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):  # 64 KB chunks
            hasher.update(chunk)
            # Record CPU and memory usage
            cpu_usage.append(process.cpu_percent(interval=None))
            memory_usage.append(process.memory_info().rss / (1024 ** 2))  # Memory in MB

    elapsed_time = time.time() - start_time

    return {
        "Average CPU Usage (%)": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
        "Peak Memory Usage (MB)": max(memory_usage) if memory_usage else 0,
        "Average Memory Usage (MB)": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
        "Hashing Duration (s)": elapsed_time
    }

def test_resource_usage(test_folder, algorithms):
    """
    Test CPU, memory usage, and duration for hashing algorithms using files in the test folder.

    Parameters:
        test_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.

    Returns:
        pd.DataFrame: DataFrame containing resource usage statistics.
    """
    results = []

    # Iterate over all files in the test folder
    for file_name in sorted(os.listdir(test_folder)):
        file_path = os.path.join(test_folder, file_name)
        file_size = os.path.getsize(file_path)

        # Test each algorithm
        for algo_name, algo_constructor in algorithms.items():
            usage_stats = measure_resource_usage(file_path, algo_constructor)
            results.append({
                "File Name": file_name,
                "File Size (bytes)": file_size,
                "Algorithm": algo_name,
                **usage_stats
            })

    return pd.DataFrame(results)

# Define the hashing algorithms to test
algorithms = {
    "MD5": hashlib.md5,
    "SHA-1": hashlib.sha1,
    "SHA-256": hashlib.sha256,
    "SHA3-256": hashlib.sha3_256,
    "BLAKE2b": hashlib.blake2b,
}

# Path to test data folder
test_data_folder = "test_data"

# Run the resource usage tests
resource_usage_df = test_resource_usage(test_data_folder, algorithms)

# Create results folder
os.makedirs("hashing_results", exist_ok=True)

# Save detailed results to a CSV
resource_usage_df.to_csv("hashing_results/hash_resource_usage_results.csv", index=False)

# Display detailed resource usage statistics using tabulate
print("\nResource Usage Statistics:")
print(tabulate(resource_usage_df, headers="keys", tablefmt="grid"))

# Calculate and display average resource usage per algorithm
average_usage = resource_usage_df.groupby("Algorithm")[
    ["Average CPU Usage (%)", "Peak Memory Usage (MB)", "Average Memory Usage (MB)", "Hashing Duration (s)"]
].mean()

# Save summary statistics to a CSV
average_usage.to_csv("hashing_results/hash_resource_usage_statistics.csv")

print("\nAverage Resource Usage by Algorithm:")
print(tabulate(average_usage, headers=["Algorithm", "Avg CPU (%)", "Peak Memory (MB)", "Avg Memory (MB)", "Avg Duration (s)"], tablefmt="grid"))

# Visualization: Separate plots for CPU, Memory, and Duration
os.makedirs("hashing_visualization", exist_ok=True)

# Average CPU Usage Bar Chart
plt.figure(figsize=(8, 6))
plt.bar(average_usage.index, average_usage["Average CPU Usage (%)"], color="skyblue", edgecolor="black")
plt.title("Average CPU Usage by Algorithm", fontsize=14)
plt.ylabel("CPU Usage (%)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("hashing_visualization/average_cpu_usage.png", dpi=300)
plt.close()

# Peak Memory Usage Bar Chart
plt.figure(figsize=(8, 6))
plt.bar(average_usage.index, average_usage["Peak Memory Usage (MB)"], color="lightcoral", edgecolor="black")
plt.title("Peak Memory Usage by Algorithm", fontsize=14)
plt.ylabel("Memory Usage (MB)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("hashing_visualization/peak_memory_usage.png", dpi=300)
plt.close()

# Hashing Duration Bar Chart
plt.figure(figsize=(8, 6))
plt.bar(average_usage.index, average_usage["Hashing Duration (s)"], color="limegreen", edgecolor="black")
plt.title("Hashing Duration by Algorithm", fontsize=14)
plt.ylabel("Duration (s)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("hashing_visualization/hashing_duration.png", dpi=300)
plt.close()

print("\nDetailed results saved to 'hashing_results/hash_resource_usage_results.csv'.")
print("Summary statistics saved to 'hashing_results/hash_resource_usage_statistics.csv'.")
print("Visualizations saved as separate files in 'hashing_visualization/'.")
