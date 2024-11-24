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
from concurrent.futures import ThreadPoolExecutor

# Configure plot style for IEEE compliance
rcParams['font.family'] = 'serif'  # Use serif font
rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'gray'])  # Grayscale colors
rcParams['text.color'] = 'black'
rcParams['grid.alpha'] = 0.5  # Subtle gridlines


def measure_throughput(file_path, algorithm, key_size=None):
    """
    Measure throughput for encryption and decryption with optimizations.

    Parameters:
        file_path (str): Path to the file to encrypt and decrypt.
        algorithm: Encryption algorithm (AES, 3DES, ChaCha20, or Camellia).
        key_size (int): Key size in bits for AES, Camellia, or ChaCha20.

    Returns:
        dict: Throughput (MB/s) for encryption and decryption.
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

        # Cache file in memory
        with open(file_path, 'rb') as f:
            plaintext = f.read()

        # Encryption throughput
        start_encrypt_time = time.time()
        encrypted_data = b""
        chunk_size = 65536  # 64 KB
        for i in range(0, len(plaintext), chunk_size):
            encrypted_data += encryptor.update(plaintext[i:i + chunk_size])
        encryptor.finalize()
        encryption_time = time.time() - start_encrypt_time
        encryption_throughput = len(plaintext) / (encryption_time * 1024 ** 2)  # MB/s

        # Decryption throughput
        start_decrypt_time = time.time()
        decrypted_data = b""
        for i in range(0, len(encrypted_data), chunk_size):
            decrypted_data += decryptor.update(encrypted_data[i:i + chunk_size])
        decryptor.finalize()
        decryption_time = time.time() - start_decrypt_time
        decryption_throughput = len(plaintext) / (decryption_time * 1024 ** 2)  # MB/s

        return {
            "Encryption Throughput (MB/s)": encryption_throughput,
            "Decryption Throughput (MB/s)": decryption_throughput,
        }
    except Exception as e:
        print(f"Error processing file {file_path} with algorithm {algorithm}: {e}")
        return None


def test_throughput(data_folder, algorithms, max_file_size_mb=20):
    """
    Test throughput for files under the size limit.

    Parameters:
        data_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.
        max_file_size_mb (int): Maximum file size in MB to process.

    Returns:
        pd.DataFrame: DataFrame containing throughput results.
    """
    results = []
    max_file_size_bytes = max_file_size_mb * 1024 * 1024

    test_files = [
        os.path.join(data_folder, file_name)
        for file_name in sorted(os.listdir(data_folder))
        if os.path.getsize(os.path.join(data_folder, file_name)) <= max_file_size_bytes
    ]

    def process_file(file_info):
        file_path, algo_name, algo_constructor, key_size = file_info
        metrics = measure_throughput(file_path, algo_constructor, key_size)
        if metrics:
            metrics.update({"File Size (bytes)": os.path.getsize(file_path), "Algorithm": algo_name})
            return metrics

    with ThreadPoolExecutor() as executor:
        file_infos = [
            (file_path, algo_name, algo_constructor, key_size)
            for file_path in test_files
            for algo_name, (algo_constructor, key_size) in algorithms.items()
        ]
        for result in executor.map(process_file, file_infos):
            if result:
                results.append(result)

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

# Measure throughput
results_df = test_throughput(test_data_folder, algorithms_to_test)

# Create folders for results and visualizations
results_folder = "symmetric_results_throughput"
visualizations_folder = "symmetric_visualization_throughput"
anova_results_folder = "symmetric_anova_throughput"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(visualizations_folder, exist_ok=True)
os.makedirs(anova_results_folder, exist_ok=True)

# Save full statistics to a CSV
results_csv_path = os.path.join(results_folder, "throughput_full_results.csv")
results_df.to_csv(results_csv_path, index=False)

# Calculate average throughput grouped by file size and algorithm
average_throughput = results_df.groupby(["File Size (bytes)", "Algorithm"])[
    ["Encryption Throughput (MB/s)", "Decryption Throughput (MB/s)"]
].mean()

# Save average throughput to a CSV
average_throughput_csv_path = os.path.join(results_folder, "throughput_average_results.csv")
average_throughput.to_csv(average_throughput_csv_path)

# ANOVA analysis for throughput
anova_encrypt = f_oneway(
    *[results_df[results_df["Algorithm"] == algo]["Encryption Throughput (MB/s)"].values
      for algo in algorithms_to_test.keys()]
)
anova_decrypt = f_oneway(
    *[results_df[results_df["Algorithm"] == algo]["Decryption Throughput (MB/s)"].values
      for algo in algorithms_to_test.keys()]
)

# Save ANOVA results to a text file
anova_results_path = os.path.join(anova_results_folder, "throughput_anova_results.txt")
with open(anova_results_path, "w") as f:
    f.write("ANOVA Results:\n")
    f.write(f"Encryption Throughput - F-statistic: {anova_encrypt.statistic:.4f}, p-value: {anova_encrypt.pvalue:.4e}\n")
    f.write(f"Decryption Throughput - F-statistic: {anova_decrypt.statistic:.4f}, p-value: {anova_decrypt.pvalue:.4e}\n")

# Visualize average throughput with barcharts
for metric in ["Encryption Throughput (MB/s)", "Decryption Throughput (MB/s)"]:
    plt.figure(figsize=(10, 6))

    # Prepare data for barchart
    grouped = average_throughput[metric].unstack()  # File Size x Algorithms
    x_positions = np.arange(len(grouped.index))  # File sizes
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
print(f"Average throughput saved to: {average_throughput_csv_path}")
print(f"ANOVA results saved to: {anova_results_path}")
print(f"Visualizations saved to: {visualizations_folder}/")
