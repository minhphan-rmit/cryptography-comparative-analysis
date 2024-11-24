import os
import hashlib
import pandas as pd
import time
import matplotlib.pyplot as plt

def measure_throughput(file_path, algorithm):
    """
    Measure the throughput of a hashing algorithm in MB/s.

    Parameters:
        file_path (str): Path to the file to hash.
        algorithm (callable): Hashing algorithm constructor (e.g., hashlib.md5).

    Returns:
        float: Throughput in MB/s.
    """
    hasher = algorithm()
    file_size = os.path.getsize(file_path)  # File size in bytes
    start_time = time.time()
    
    # Hash file contents in chunks
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):  # Read file in 8KB chunks
            hasher.update(chunk)
    
    elapsed_time = time.time() - start_time
    throughput = (file_size / (1024 ** 2)) / elapsed_time  # Convert bytes to MB, divide by time
    return throughput

def test_algorithm_throughput(test_folder, algorithms):
    """
    Test throughput for multiple algorithms on files in the test folder.

    Parameters:
        test_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.

    Returns:
        pd.DataFrame: DataFrame containing throughput results.
    """
    results = []

    # Iterate over all files in the folder
    for file_name in sorted(os.listdir(test_folder)):
        file_path = os.path.join(test_folder, file_name)
        file_size = os.path.getsize(file_path)

        # Test each algorithm
        for algo_name, algo_constructor in algorithms.items():
            throughput = measure_throughput(file_path, algo_constructor)
            results.append({
                "File Name": file_name,
                "File Size (bytes)": file_size,
                "Algorithm": algo_name,
                "Throughput (MB/s)": throughput
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

# Run the throughput tests
throughput_df = test_algorithm_throughput(test_data_folder, algorithms)

# Save throughput results to a dedicated folder
os.makedirs("hashing_results", exist_ok=True)
throughput_df.to_csv("hashing_results/hashing_throughput_results.csv", index=False)

# Compute average throughput per algorithm and file size
avg_throughput = throughput_df.groupby(["File Size (bytes)", "Algorithm"])["Throughput (MB/s)"].mean().unstack()

# Save average throughput table to a CSV
avg_throughput.to_csv("hashing_results/hashing_throughput_averages.csv", index=True)

# Display the throughput averages
print("\nAverage Throughput per File Size and Algorithm (MB/s):")
print(avg_throughput)

# Create a readable file size label: KB for <1 MB, MB otherwise
def readable_file_size(size_in_bytes):
    if size_in_bytes < 1024 ** 2:  # Files smaller than 1 MB
        return f"{size_in_bytes / 1024:.1f} KB"
    else:
        return f"{size_in_bytes / (1024 ** 2):.1f} MB"

# Convert file sizes in the index for easier reading
avg_throughput.index = [readable_file_size(size) for size in avg_throughput.index]

# Calculate average throughput for each algorithm
average_throughput = throughput_df.groupby("Algorithm")["Throughput (MB/s)"].mean().sort_values(ascending=True)

# Create a horizontal bar chart to visualize average throughput
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars
average_throughput.plot(kind="barh", color="tab:blue", ax=ax)

# Formatting
ax.set_title("Average Throughput of Hashing Algorithms", fontsize=14)
ax.set_xlabel("Average Throughput (MB/s)", fontsize=12)
ax.set_ylabel("Algorithm", fontsize=12)
ax.grid(axis="x", linestyle="--", alpha=0.7)

# Add throughput values on the bars
for i, value in enumerate(average_throughput):
    ax.text(value + 5, i, f"{value:.1f}", va="center", fontsize=10)

# Ensure tight layout for better formatting
plt.tight_layout()

# Save the bar chart to a dedicated folder
os.makedirs("hashing_visualization", exist_ok=True)
plt.savefig("hashing_visualization/average_hashing_throughput_bar_chart.png", dpi=300)

# Display the plot
plt.show()
