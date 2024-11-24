import os
import math
from collections import Counter
from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Configure plot style
rcParams['font.family'] = 'serif'  # Use serif font

def calculate_entropy(data):
    """
    Calculate the Shannon entropy of a byte sequence.

    Parameters:
        data (bytes): The data to calculate entropy for.

    Returns:
        float: Shannon entropy value.
    """
    if not data:
        return 0

    # Count occurrences of each byte
    byte_counts = Counter(data)
    total_bytes = len(data)

    # Calculate probabilities and entropy
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_counts.values())
    return entropy

def measure_entropy(file_path, algorithm, key_size=None):
    """
    Encrypt a file and measure the entropy of the encrypted output.

    Parameters:
        file_path (str): Path to the file to encrypt.
        algorithm: Encryption algorithm (AES, 3DES, ChaCha20, or Camellia).
        key_size (int): Key size in bits for AES, Camellia, or ChaCha20.

    Returns:
        dict: File size and entropy of the encrypted output.
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

        # Encrypt the file
        encrypted_data = b""
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):  # 64 KB chunks
                encrypted_data += encryptor.update(chunk)
        encryptor.finalize()

        # Calculate entropy of the encrypted output
        entropy = calculate_entropy(encrypted_data)
        return {"Entropy": entropy}
    except Exception as e:
        print(f"Error processing file {file_path} with algorithm {algorithm}: {e}")
        return None

def test_entropy(data_folder, algorithms, max_file_size_mb=20):
    """
    Test entropy for encrypted outputs of all files in the folder.

    Parameters:
        data_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.
        max_file_size_mb (int): Maximum file size in MB to process.

    Returns:
        pd.DataFrame: DataFrame containing entropy results.
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
            metrics = measure_entropy(file_path, algo_constructor, key_size)
            if metrics:
                metrics.update({"File Size (bytes)": file_size, "Algorithm": algo_name, "File Name": file_name})
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

# Measure entropy
results_df = test_entropy(test_data_folder, algorithms_to_test)

# Create folders for results and visualizations
results_folder = "symmetric_results_entropy"
visualizations_folder = "symmetric_visualization_entropy"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(visualizations_folder, exist_ok=True)

# Save full statistics to a CSV
results_csv_path = os.path.join(results_folder, "entropy_full_results.csv")
results_df.to_csv(results_csv_path, index=False)

# Calculate average entropy grouped by file size and algorithm
average_entropy = results_df.groupby(["File Size (bytes)", "Algorithm"])["Entropy"].mean()

# Save average entropy to a CSV
average_entropy_csv_path = os.path.join(results_folder, "entropy_average_results.csv")
average_entropy.to_csv(average_entropy_csv_path)

# Visualize entropy with barcharts
plt.figure(figsize=(10, 6))

# Prepare data for barchart
grouped = average_entropy.unstack()  # File Size x Algorithms
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
plt.title("Average Entropy by Algorithm and File Size", fontsize=14)
plt.xlabel("File Size (bytes)", fontsize=12)
plt.ylabel("Entropy", fontsize=12)
plt.xticks(x_positions + (len(grouped.columns) - 1) * bar_width / 2, grouped.index, rotation=45)
plt.legend(title="Algorithm", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Save the barchart
output_path = os.path.join(visualizations_folder, "average_entropy.png")
plt.savefig(output_path, dpi=300)
plt.close()

# Print completion message
print(f"\nFull results saved to: {results_csv_path}")
print(f"Average entropy saved to: {average_entropy_csv_path}")
print(f"Visualization saved to: {visualizations_folder}/")
