import os
import time
from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pandas as pd
from tabulate import tabulate
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Configure plot style for IEEE compliance
rcParams['font.family'] = 'serif'  # Use serif font
rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'gray'])  # Grayscale colors
rcParams['text.color'] = 'black'
rcParams['grid.alpha'] = 0.5  # Subtle gridlines

def measure_latency(algorithm, chunk_size, key_size=None):
    """
    Measure latency for encryption and decryption of a single chunk of data.

    Parameters:
        algorithm: Encryption algorithm (AES, 3DES, ChaCha20, or Camellia).
        chunk_size (int): Size of the chunk to encrypt and decrypt in bytes.
        key_size (int): Key size in bits for AES, Camellia, or ChaCha20.

    Returns:
        dict: Latency (in microseconds) for encryption and decryption.
    """
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

        # Create random data of the specified chunk size
        chunk = os.urandom(chunk_size)

        # Measure encryption latency
        start_encrypt_time = time.time()
        encrypted_chunk = encryptor.update(chunk) + encryptor.finalize()
        encryption_latency = (time.time() - start_encrypt_time) * 1e6  # Convert to microseconds

        # Measure decryption latency
        start_decrypt_time = time.time()
        decryptor.update(encrypted_chunk) + decryptor.finalize()
        decryption_latency = (time.time() - start_decrypt_time) * 1e6  # Convert to microseconds

        return {
            "Encryption Latency (µs)": encryption_latency,
            "Decryption Latency (µs)": decryption_latency,
        }
    except Exception as e:
        print(f"Error processing latency for algorithm {algorithm}: {e}")
        return None


def test_latency(algorithms, chunk_sizes):
    """
    Test latency for different algorithms and chunk sizes.

    Parameters:
        algorithms (dict): Dictionary of algorithm names and constructors.
        chunk_sizes (list): List of chunk sizes to test (in bytes).

    Returns:
        pd.DataFrame: DataFrame containing latency results.
    """
    results = []

    # Iterate over each algorithm and chunk size
    for algo_name, (algo_constructor, key_size) in algorithms.items():
        for chunk_size in chunk_sizes:
            metrics = measure_latency(algo_constructor, chunk_size, key_size)
            if metrics:
                metrics.update({"Chunk Size (bytes)": chunk_size, "Algorithm": algo_name})
                results.append(metrics)

    return pd.DataFrame(results)


# Define the algorithms to test
algorithms_to_test = {
    "AES (128-bit)": (algorithms.AES, 128),
    "3DES": (TripleDES, None),
    "ChaCha20": (algorithms.ChaCha20, None),
    "Camellia (128-bit)": (algorithms.Camellia, 128),
}

# Define chunk sizes to test (e.g., 1KB, 4KB, 8KB, etc.)
chunk_sizes_to_test = [1024, 4096, 8192, 16384, 32768]  # 1KB to 32KB

# Measure latency
results_df = test_latency(algorithms_to_test, chunk_sizes_to_test)

# Create folders for results and visualizations
results_folder = "symmetric_results_latency"
visualizations_folder = "symmetric_visualization_latency"
anova_results_folder = "symmetric_anova_latency"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(visualizations_folder, exist_ok=True)
os.makedirs(anova_results_folder, exist_ok=True)

# Save full statistics to a CSV
results_csv_path = os.path.join(results_folder, "latency_full_results.csv")
results_df.to_csv(results_csv_path, index=False)

# Calculate average latency grouped by chunk size and algorithm
average_latency = results_df.groupby(["Chunk Size (bytes)", "Algorithm"])[
    ["Encryption Latency (µs)", "Decryption Latency (µs)"]
].mean()

# Save average latency to a CSV
average_latency_csv_path = os.path.join(results_folder, "latency_average_results.csv")
average_latency.to_csv(average_latency_csv_path)

# ANOVA analysis for latency
anova_encrypt = f_oneway(
    *[results_df[results_df["Algorithm"] == algo]["Encryption Latency (µs)"].values
      for algo in algorithms_to_test.keys()]
)
anova_decrypt = f_oneway(
    *[results_df[results_df["Algorithm"] == algo]["Decryption Latency (µs)"].values
      for algo in algorithms_to_test.keys()]
)

# Save ANOVA results to a text file
anova_results_path = os.path.join(anova_results_folder, "latency_anova_results.txt")
with open(anova_results_path, "w") as f:
    f.write("ANOVA Results:\n")
    f.write(f"Encryption Latency - F-statistic: {anova_encrypt.statistic:.4f}, p-value: {anova_encrypt.pvalue:.4e}\n")
    f.write(f"Decryption Latency - F-statistic: {anova_decrypt.statistic:.4f}, p-value: {anova_decrypt.pvalue:.4e}\n")

# Visualize latency with barcharts
for metric in ["Encryption Latency (µs)", "Decryption Latency (µs)"]:
    plt.figure(figsize=(10, 6))

    # Prepare data for barchart
    grouped = average_latency[metric].unstack()  # Chunk Size x Algorithms
    x_positions = np.arange(len(grouped.index))  # Chunk sizes
    bar_width = 0.15  # Width of each bar
    offset = 0

    # Plot bars for each algorithm
    for i, algorithm in enumerate(grouped.columns):
        plt.bar(
            x_positions + offset,
            grouped[algorithm],
            width=bar_width,
            label=algorithm
        )
        offset += bar_width

    # Add labels and legend
    plt.title(f"Average {metric} by Algorithm and Chunk Size", fontsize=14)
    plt.xlabel("Chunk Size (bytes)", fontsize=12)
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
print(f"Average latency saved to: {average_latency_csv_path}")
print(f"ANOVA results saved to: {anova_results_path}")
print(f"Visualizations saved to: {visualizations_folder}/")
