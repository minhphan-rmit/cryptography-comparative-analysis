import os
import hashlib
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

def calculate_hamming_distance(hash1, hash2):
    """
    Calculate the Hamming distance between two hexadecimal hash outputs.

    Parameters:
        hash1 (str): First hash output as a hexadecimal string.
        hash2 (str): Second hash output as a hexadecimal string.

    Returns:
        int: The Hamming distance (number of differing bits).
    """
    # Convert hex to binary and calculate differing bits
    bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
    return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))

def test_avalanche_effect(test_folder, algorithms):
    """
    Test the avalanche effect of hashing algorithms using files in the test folder.

    Parameters:
        test_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.

    Returns:
        pd.DataFrame: DataFrame containing avalanche effect results.
    """
    results = []

    # Iterate over all files in the folder
    for file_name in sorted(os.listdir(test_folder)):
        file_path = os.path.join(test_folder, file_name)

        # Read file content
        with open(file_path, 'rb') as f:
            original_data = f.read()

        # Modify the data slightly (flip one bit)
        modified_data = bytearray(original_data)
        modified_data[0] ^= 1  # Flip the first bit of the first byte

        # Test each algorithm
        for algo_name, algo_constructor in algorithms.items():
            hasher_original = algo_constructor()
            hasher_original.update(original_data)
            hash_original = hasher_original.hexdigest()

            hasher_modified = algo_constructor()
            hasher_modified.update(modified_data)
            hash_modified = hasher_modified.hexdigest()

            # Calculate Hamming distance
            hamming_distance = calculate_hamming_distance(hash_original, hash_modified)
            total_bits = len(hash_original) * 4  # Total bits in the hash
            avalanche_effect = (hamming_distance / total_bits) * 100  # Percentage

            results.append({
                "File Name": file_name,
                "Algorithm": algo_name,
                "Hamming Distance": hamming_distance,
                "Avalanche Effect (%)": avalanche_effect
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

# Run the avalanche effect tests
avalanche_df = test_avalanche_effect(test_data_folder, algorithms)

# Create folders for results and visualizations
os.makedirs("hashing_results", exist_ok=True)
os.makedirs("hashing_visualization", exist_ok=True)

# Save detailed results to a CSV
avalanche_df.to_csv("hashing_results/hash_avalanche_effect_results.csv", index=False)

# Calculate average avalanche effect per algorithm
avalanche_stats = avalanche_df.groupby("Algorithm")["Avalanche Effect (%)"].agg(["mean", "std"])

# Save summary statistics to a CSV
avalanche_stats.to_csv("hashing_results/hash_avalanche_effect_statistics.csv")

# Display results using tabulate
print("\nAvalanche Effect Statistics by Algorithm:")
print(tabulate(avalanche_stats, headers=["Algorithm", "Average (%)", "Std Dev"], tablefmt="grid"))

# Visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Bar chart with error bars for average avalanche effect
ax.bar(
    avalanche_stats.index,
    avalanche_stats["mean"],
    yerr=avalanche_stats["std"],
    capsize=5,
    color="skyblue",
    edgecolor="black"
)

# Add labels and formatting
ax.set_title("Avalanche Effect of Hashing Algorithms", fontsize=14)
ax.set_xlabel("Hashing Algorithm", fontsize=12)
ax.set_ylabel("Avalanche Effect (%)", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save the bar chart to a file
plt.tight_layout()
plt.savefig("hashing_visualization/hash_avalanche_effect_bar_chart.png", dpi=300)

# Display the plot
plt.show()

print("\nResults saved to 'hashing_results/hash_avalanche_effect_results.csv' and summary statistics to 'hashing_results/hash_avalanche_effect_statistics.csv'.")
print("Visualization saved to 'hashing_visualization/hash_avalanche_effect_bar_chart.png'.")
