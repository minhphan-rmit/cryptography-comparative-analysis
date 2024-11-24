import os
import time
import psutil
from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pandas as pd
from tabulate import tabulate
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Configure plot style
rcParams['font.family'] = 'serif'  # Use serif font

def measure_cpu_memory(file_path, algorithm, key_size=None):
    """
    Measure CPU and memory usage during encryption and decryption.

    Parameters:
        file_path (str): Path to the file to encrypt and decrypt.
        algorithm: Encryption algorithm (AES, 3DES, ChaCha20, or Camellia).
        key_size (int): Key size in bits for AES, Camellia, or ChaCha20.

    Returns:
        dict: CPU and memory usage metrics.
    """
    process = psutil.Process(os.getpid())
    cpu_usage = []
    memory_usage = []

    try:
        # Generate random key and IV/nonce
        if algorithm == TripleDES:
            key = os.urandom(24)
            iv = os.urandom(8)
            cipher = Cipher(TripleDES(key), modes.CFB(iv), backend=default_backend())
        elif algorithm in [algorithms.AES, algorithms.Camellia]:
            key = os.urandom(key_size // 8)
            iv = os.urandom(16)
            cipher = Cipher(algorithm(key), modes.CFB(iv), backend=default_backend())
        elif algorithm == algorithms.ChaCha20:
            key = os.urandom(32)
            nonce = os.urandom(16)
            cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
        else:
            raise ValueError("Unsupported algorithm")

        encryptor = cipher.encryptor()
        decryptor = cipher.decryptor()

        # Read file into memory
        with open(file_path, 'rb') as f:
            data = f.read()

        # Measure CPU and memory during encryption
        start_time = time.time()
        encrypted_data = b""
        for i in range(0, len(data), 65536):  # 64 KB chunks
            encrypted_data += encryptor.update(data[i:i + 65536])
            cpu_usage.append(process.cpu_percent(interval=None))
            memory_usage.append(process.memory_info().rss / (1024 ** 2))  # Convert to MB
        encryptor.finalize()
        encryption_time = time.time() - start_time

        avg_cpu_encrypt = sum(cpu_usage) / len(cpu_usage)
        peak_memory_encrypt = max(memory_usage)

        # Measure CPU and memory during decryption
        cpu_usage.clear()
        memory_usage.clear()
        start_time = time.time()
        decrypted_data = b""
        for i in range(0, len(encrypted_data), 65536):  # 64 KB chunks
            decrypted_data += decryptor.update(encrypted_data[i:i + 65536])
            cpu_usage.append(process.cpu_percent(interval=None))
            memory_usage.append(process.memory_info().rss / (1024 ** 2))  # Convert to MB
        decryptor.finalize()
        decryption_time = time.time() - start_time

        avg_cpu_decrypt = sum(cpu_usage) / len(cpu_usage)
        peak_memory_decrypt = max(memory_usage)

        return {
            "Encryption Time (s)": encryption_time,
            "Decryption Time (s)": decryption_time,
            "Encryption Avg CPU (%)": avg_cpu_encrypt,
            "Decryption Avg CPU (%)": avg_cpu_decrypt,
            "Encryption Peak Memory (MB)": peak_memory_encrypt,
            "Decryption Peak Memory (MB)": peak_memory_decrypt,
        }
    except Exception as e:
        print(f"Error processing file {file_path} with algorithm {algorithm}: {e}")
        return None


def test_cpu_memory(data_folder, algorithms, max_file_size_mb=20):
    """
    Test CPU and memory usage for all files in the folder.

    Parameters:
        data_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.
        max_file_size_mb (int): Maximum file size in MB to process.

    Returns:
        pd.DataFrame: DataFrame containing CPU and memory usage results.
    """
    results = []
    max_file_size_bytes = max_file_size_mb * 1024 * 1024

    # Process each file in the folder
    for file_name in sorted(os.listdir(data_folder)):
        file_path = os.path.join(data_folder, file_name)
        file_size = os.path.getsize(file_path)

        # Skip files larger than the maximum size
        if file_size > max_file_size_bytes:
            continue

        # Test each algorithm
        for algo_name, (algo_constructor, key_size) in algorithms.items():
            metrics = measure_cpu_memory(file_path, algo_constructor, key_size)
            if metrics:
                metrics.update({"File Size (bytes)": file_size, "Algorithm": algo_name})
                results.append(metrics)

    return pd.DataFrame(results)


# Define the algorithms to test
algorithms_to_test = {
    "AES (128-bit)": (algorithms.AES, 128),
    "3DES": (TripleDES, None),
    "ChaCha20": (algorithms.ChaCha20, None),
    "Camellia (128-bit)": (algorithms.Camellia, 128),
}

# Path to the test data folder
test_data_folder = "test_data"

# Measure CPU and memory usage
results_df = test_cpu_memory(test_data_folder, algorithms_to_test)

# Create folders for results and visualizations
results_folder = "symmetric_results_cpu_memory"
visualizations_folder = "symmetric_visualization_cpu_memory"
anova_results_folder = "symmetric_anova_cpu_memory"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(visualizations_folder, exist_ok=True)
os.makedirs(anova_results_folder, exist_ok=True)

# Save full statistics to a CSV
results_csv_path = os.path.join(results_folder, "cpu_memory_full_results.csv")
results_df.to_csv(results_csv_path, index=False)

# Calculate average CPU and memory usage grouped by file size and algorithm
average_usage = results_df.groupby(["File Size (bytes)", "Algorithm"])[
    ["Encryption Avg CPU (%)", "Decryption Avg CPU (%)", "Encryption Peak Memory (MB)", "Decryption Peak Memory (MB)"]
].mean()

# Save average usage to a CSV
average_usage_csv_path = os.path.join(results_folder, "cpu_memory_average_results.csv")
average_usage.to_csv(average_usage_csv_path)

# ANOVA analysis for CPU and memory usage
anova_encrypt_cpu = f_oneway(
    *[results_df[results_df["Algorithm"] == algo]["Encryption Avg CPU (%)"].values
      for algo in algorithms_to_test.keys()]
)
anova_decrypt_cpu = f_oneway(
    *[results_df[results_df["Algorithm"] == algo]["Decryption Avg CPU (%)"].values
      for algo in algorithms_to_test.keys()]
)

# Save ANOVA results to a text file
anova_results_path = os.path.join(anova_results_folder, "cpu_memory_anova_results.txt")
with open(anova_results_path, "w") as f:
    f.write("ANOVA Results:\n")
    f.write(f"Encryption Avg CPU - F-statistic: {anova_encrypt_cpu.statistic:.4f}, p-value: {anova_encrypt_cpu.pvalue:.4e}\n")
    f.write(f"Decryption Avg CPU - F-statistic: {anova_decrypt_cpu.statistic:.4f}, p-value: {anova_decrypt_cpu.pvalue:.4e}\n")

# Visualize average CPU and memory usage with barcharts
metrics_to_plot = [
    "Encryption Avg CPU (%)",
    "Decryption Avg CPU (%)",
    "Encryption Peak Memory (MB)",
    "Decryption Peak Memory (MB)"
]

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))

    # Prepare data for barchart
    grouped = average_usage[metric].unstack()  # File Size x Algorithms
    x_positions = np.arange(len(grouped.index))  # File sizes
    bar_width = 0.15  # Width of each bar
    offset = 0

    # Plot bars for each algorithm
    colors = plt.cm.tab20.colors  # Use tab colors
    for i, algorithm in enumerate(grouped.columns):
        plt.bar(
            x_positions + offset,
            grouped[algorithm],
            width=bar_width,
            label=algorithm,
            color=colors[i % len(colors)]
        )
        offset += bar_width

    # Add labels and legend
    plt.title(f"Average {metric} by Algorithm and File Size", fontsize=14)
    plt.xlabel("File Size (bytes)", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(x_positions + (len(grouped.columns) - 1) * bar_width / 2, grouped.index, rotation=45)
    plt.legend(title="Algorithm", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the barchart
    metric_cleaned = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").lower()
    output_path = os.path.join(visualizations_folder, f"{metric_cleaned}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

# Print completion message
print(f"\nFull results saved to: {results_csv_path}")
print(f"Average usage saved to: {average_usage_csv_path}")
print(f"ANOVA results saved to: {anova_results_path}")
print(f"Visualizations saved to: {visualizations_folder}/")
