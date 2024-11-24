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

# Configure plot style for IEEE compliance
rcParams['font.family'] = 'serif'  # Use serif font
rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'gray'])  # Grayscale colors
rcParams['text.color'] = 'black'
rcParams['grid.alpha'] = 0.5  # Subtle gridlines

def encrypt_decrypt_data(file_path, algorithm, key_size=None):
    """
    Measure encryption and decryption times for a given file and algorithm.

    Parameters:
        file_path (str): Path to the file to encrypt and decrypt.
        algorithm: Encryption algorithm (AES, 3DES, ChaCha20, or Camellia).
        key_size (int): Key size in bits for AES, Camellia, or ChaCha20.

    Returns:
        dict: Encryption and decryption times in seconds.
    """
    try:
        # Generate random key and IV/nonce based on the algorithm
        if algorithm == TripleDES:
            key = os.urandom(24)  # 3DES requires a 192-bit key
            iv = os.urandom(8)  # Block size for 3DES is 8 bytes
            cipher = Cipher(TripleDES(key), modes.CFB(iv), backend=default_backend())
        elif algorithm in [algorithms.AES, algorithms.Camellia]:
            key = os.urandom(key_size // 8)
            iv = os.urandom(16)  # Block size for AES and Camellia is 16 bytes
            cipher = Cipher(algorithm(key), modes.CFB(iv), backend=default_backend())
        elif algorithm == algorithms.ChaCha20:
            key = os.urandom(32)  # ChaCha20 requires a 256-bit key
            nonce = os.urandom(16)
            cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
        else:
            raise ValueError("Unsupported algorithm")

        encryptor = cipher.encryptor()
        decryptor = cipher.decryptor()

        # Measure encryption time
        start_encrypt_time = time.time()
        encrypted_data = b""
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):  # Process in 8 KB chunks
                encrypted_data += encryptor.update(chunk)
        encryptor.finalize()
        end_encrypt_time = time.time()

        # Measure decryption time
        start_decrypt_time = time.time()
        decrypted_data = b""
        while encrypted_data:
            chunk, encrypted_data = encrypted_data[:8192], encrypted_data[8192:]
            decrypted_data += decryptor.update(chunk)
        decryptor.finalize()
        end_decrypt_time = time.time()

        return {
            "Encryption Time (s)": end_encrypt_time - start_encrypt_time,
            "Decryption Time (s)": end_decrypt_time - start_decrypt_time,
        }
    except Exception as e:
        print(f"Error processing file {file_path} with algorithm {algorithm}: {e}")
        return {"Encryption Time (s)": None, "Decryption Time (s)": None}

# Define the algorithms to test
algorithms_to_test = {
    "AES (128-bit)": (algorithms.AES, 128),
    "3DES": (TripleDES, None),
    "ChaCha20": (algorithms.ChaCha20, None),
    "Camellia (128-bit)": (algorithms.Camellia, 128),
}

def test_encryption_decryption(data_folder, algorithms, max_file_size_mb=20):
    """
    Test encryption and decryption times for files under the size limit.

    Parameters:
        data_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.
        max_file_size_mb (int): Maximum file size in MB to process.

    Returns:
        pd.DataFrame: DataFrame containing encryption and decryption times.
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
            timings = encrypt_decrypt_data(file_path, algo_constructor, key_size)
            if timings["Encryption Time (s)"] is not None and timings["Decryption Time (s)"] is not None:
                results.append({
                    "File Size (bytes)": file_size,
                    "Algorithm": algo_name,
                    **timings,
                })

    return pd.DataFrame(results)

# Path to the test data folder
test_data_folder = "test_data"

# Measure encryption and decryption times
results_df = test_encryption_decryption(test_data_folder, algorithms_to_test)

# Create folders for results and visualizations
results_folder = "symmetric_results"
visualizations_folder = "symmetric_visualization"
anova_results_folder = "symmetric_anova_result"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(visualizations_folder, exist_ok=True)
os.makedirs(anova_results_folder, exist_ok=True)

# Save full statistics to a CSV
results_csv_path = os.path.join(results_folder, "encryption_decryption_full_results.csv")
results_df.to_csv(results_csv_path, index=False)

# Calculate average times grouped by file size and algorithm
average_times = results_df.groupby(["File Size (bytes)", "Algorithm"])[
    ["Encryption Time (s)", "Decryption Time (s)"]
].mean()

# Save average times to a CSV
average_times_csv_path = os.path.join(results_folder, "encryption_decryption_average_times.csv")
average_times.to_csv(average_times_csv_path)

# ANOVA analysis for encryption and decryption times
anova_encrypt = f_oneway(
    *[results_df[results_df["Algorithm"] == algo]["Encryption Time (s)"].values
      for algo in algorithms_to_test.keys()]
)
anova_decrypt = f_oneway(
    *[results_df[results_df["Algorithm"] == algo]["Decryption Time (s)"].values
      for algo in algorithms_to_test.keys()]
)

# Save ANOVA results to a text file
anova_results_path = os.path.join(anova_results_folder, "encryption_decryption_anova_results.txt")
with open(anova_results_path, "w") as f:
    f.write("ANOVA Results:\n")
    f.write(f"Encryption Time - F-statistic: {anova_encrypt.statistic:.4f}, p-value: {anova_encrypt.pvalue:.4e}\n")
    f.write(f"Decryption Time - F-statistic: {anova_decrypt.statistic:.4f}, p-value: {anova_decrypt.pvalue:.4e}\n")

# Define file size units for labels
def format_file_size(size_in_bytes):
    """
    Convert file size to human-readable format (KB or MB).
    """
    if size_in_bytes < 1024 ** 2:  # Less than 1MB
        return f"{size_in_bytes // 1024} KB"
    else:  # 1MB or larger
        return f"{size_in_bytes / (1024 ** 2):.0f} MB"

# Generate human-readable file sizes
file_sizes_readable = sorted(set(results_df["File Size (bytes)"]))
file_sizes_labels = [format_file_size(size) for size in file_sizes_readable]

# Convert file sizes to megabytes for consistent scale
results_df["File Size (MB)"] = results_df["File Size (bytes)"] / (1024 ** 2)

# Update index for average_times to use MB
average_times.index = pd.MultiIndex.from_tuples(
    [(size / (1024 ** 2), algo) for size, algo in average_times.index],
    names=["File Size (MB)", "Algorithm"]
)

# Visualize average encryption and decryption times
markers = ["o", "s", "D", "^"]
for metric in ["Encryption Time (s)", "Decryption Time (s)"]:
    plt.figure(figsize=(8, 6))

    for i, (algorithm, data) in enumerate(average_times[metric].unstack().items()):
        plt.plot(
            data.index,  # File sizes in MB
            data.values,
            label=algorithm,
            marker=markers[i],
            linestyle="--"
        )

    # Add x-axis ticks and labels
    plt.xticks(
        [size / (1024 ** 2) for size in file_sizes_readable],  # File sizes in MB
        file_sizes_labels,  # Human-readable labels (KB, MB)
        rotation=45,  # Rotate labels for readability
        fontsize=10
    )

    plt.title(f"Average {metric} by Algorithm and File Size", fontsize=14)
    plt.xlabel("File Size", fontsize=12)  # Use mixed KB/MB labels
    plt.ylabel(metric, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Algorithm", fontsize=10)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(visualizations_folder, f"average_{metric.replace(' ', '_').lower()}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

# Print completion message
print(f"\nFull results saved to: {results_csv_path}")
print(f"Average times saved to: {average_times_csv_path}")
print(f"ANOVA results saved to: {anova_results_path}")
print(f"Visualizations saved to: {visualizations_folder}/")
