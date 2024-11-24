import os
import time
from cryptography.hazmat.primitives.asymmetric import rsa, ec, dh, padding
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.dh import generate_parameters, DHParameterNumbers
from cryptography.hazmat.primitives.asymmetric import dh
import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure plot style for IEEE compliance
rcParams['font.family'] = 'serif'  # Use serif font

# Define algorithms for testing
algorithms_to_test = {
    "RSA-2048": lambda: rsa.generate_private_key(public_exponent=65537, key_size=2048),
    "ECC-P256": lambda: ec.generate_private_key(ec.SECP256R1()),
    "Diffie-Hellman": lambda: generate_parameters(generator=2, key_size=2048),
}

# Create result folders
results_folder = "asymmetric_results"
visualizations_folder = "asymmetric_visualization"
anova_results_folder = "asymmetric_anova_results"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(visualizations_folder, exist_ok=True)
os.makedirs(anova_results_folder, exist_ok=True)

def diffie_hellman_key_exchange(parameters):
    """
    Perform a Diffie-Hellman key exchange and measure its time.
    """
    private_key = parameters.generate_private_key()
    peer_private_key = parameters.generate_private_key()

    # Perform key exchange
    start_time = time.time()
    shared_key = private_key.exchange(peer_private_key.public_key())
    end_time = time.time()

    # Derive a key from the shared secret
    HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"handshake data").derive(shared_key)

    return end_time - start_time

def benchmark_asymmetric_operations(data_folder, algorithms, max_file_size_mb=1, iterations=10):
    """
    Benchmark encryption/decryption and key exchange times for asymmetric algorithms.

    Parameters:
        data_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.
        max_file_size_mb (int): Maximum file size in MB to process.
        iterations (int): Number of iterations for benchmarking.

    Returns:
        pd.DataFrame: DataFrame containing encryption/decryption or key exchange times.
    """
    results = []
    max_file_size_bytes = max_file_size_mb * 1024 * 1024  # Convert MB to bytes

    for file_name in sorted(os.listdir(data_folder)):
        file_path = os.path.join(data_folder, file_name)
        file_size = os.path.getsize(file_path)

        # Skip files larger than the maximum size
        if file_size > max_file_size_bytes:
            print(f"Skipping file {file_name} (size {file_size} bytes > {max_file_size_bytes} bytes)")
            continue

        # Read the file data
        with open(file_path, 'rb') as f:
            data = f.read()

        for algo_name, algo_fn in algorithms.items():
            private_key = algo_fn()  # Generate key or parameters for the algorithm

            for _ in range(iterations):
                if isinstance(private_key, rsa.RSAPrivateKey):
                    # RSA Encryption/Decryption
                    public_key = private_key.public_key()
                    max_chunk_size = 190  # Max size for 2048-bit RSA with OAEP and SHA-256 padding
                    start_encrypt = time.time()
                    ciphertext = public_key.encrypt(
                        data[:max_chunk_size],  # Encrypt the first chunk
                        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
                    )
                    end_encrypt = time.time()

                    start_decrypt = time.time()
                    private_key.decrypt(
                        ciphertext,
                        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
                    )
                    end_decrypt = time.time()

                    results.append({
                        "File Name": file_name,
                        "File Size (bytes)": file_size,
                        "Algorithm": algo_name,
                        "Operation": "Encryption",
                        "Time (s)": end_encrypt - start_encrypt,
                    })
                    results.append({
                        "File Name": file_name,
                        "File Size (bytes)": file_size,
                        "Algorithm": algo_name,
                        "Operation": "Decryption",
                        "Time (s)": end_decrypt - start_decrypt,
                    })

                elif isinstance(private_key, ec.EllipticCurvePrivateKey):
                    # ECC Signing/Verification
                    start_encrypt = time.time()
                    signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
                    end_encrypt = time.time()

                    public_key = private_key.public_key()
                    start_decrypt = time.time()
                    public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
                    end_decrypt = time.time()

                    results.append({
                        "File Name": file_name,
                        "File Size (bytes)": file_size,
                        "Algorithm": algo_name,
                        "Operation": "Signing",
                        "Time (s)": end_encrypt - start_encrypt,
                    })
                    results.append({
                        "File Name": file_name,
                        "File Size (bytes)": file_size,
                        "Algorithm": algo_name,
                        "Operation": "Verification",
                        "Time (s)": end_decrypt - start_decrypt,
                    })

                elif isinstance(private_key, dh.DHParameters):
                    # Diffie-Hellman Key Exchange
                    exchange_time = diffie_hellman_key_exchange(private_key)
                    results.append({
                        "File Name": file_name,
                        "File Size (bytes)": file_size,
                        "Algorithm": algo_name,
                        "Operation": "Key Exchange",
                        "Time (s)": exchange_time,
                    })

    return pd.DataFrame(results)


# Path to the test_data folder
data_folder = "test_data"

# Benchmark operations
iterations = 10
operations_df = benchmark_asymmetric_operations(data_folder, algorithms_to_test, max_file_size_mb=1, iterations=iterations)

# Save full statistics
full_results_path = os.path.join(results_folder, "asymmetric_operations_full_results.csv")
operations_df.to_csv(full_results_path, index=False)

# Calculate average times grouped by algorithm and operation
average_times = operations_df.groupby(["Algorithm", "Operation"])["Time (s)"].agg(["mean", "std"])

# Save summary statistics
summary_results_path = os.path.join(results_folder, "asymmetric_operations_summary_results.csv")
average_times.to_csv(summary_results_path)

# Perform ANOVA analysis for operations
anova_results = {}
for operation in operations_df["Operation"].unique():
    operation_data = [
        operations_df[(operations_df["Algorithm"] == algo) & (operations_df["Operation"] == operation)]["Time (s)"].values
        for algo in algorithms_to_test.keys()
    ]
    anova_results[operation] = f_oneway(*operation_data)

# Save ANOVA results
anova_results_path = os.path.join(anova_results_folder, "asymmetric_operations_anova_results.txt")
with open(anova_results_path, "w") as f:
    f.write("ANOVA Results:\n")
    for operation, result in anova_results.items():
        f.write(f"{operation} - F-statistic: {result.statistic:.4f}, p-value: {result.pvalue:.4e}\n")

# Visualize average operation times
for operation in operations_df["Operation"].unique():
    subset = average_times.xs(operation, level=1)
    plt.figure(figsize=(8, 6))
    plt.bar(subset.index, subset["mean"], yerr=subset["std"], capsize=5, color="skyblue")
    plt.title(f"Average {operation} Time by Algorithm", fontsize=14)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Time (s)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the visualization
    output_path = os.path.join(visualizations_folder, f"{operation.lower().replace(' ', '_')}_time.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

# Print completion message
print(f"\nFull results saved to: {full_results_path}")
print(f"Summary statistics saved to: {summary_results_path}")
print(f"ANOVA results saved to: {anova_results_path}")
print(f"Visualizations saved to: {visualizations_folder}/")
