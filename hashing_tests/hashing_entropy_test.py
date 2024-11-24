import os
import hashlib
import pandas as pd
from scipy.stats import entropy, f_oneway
from tabulate import tabulate

def calculate_entropy(hash_output):
    """
    Calculate the Shannon entropy of a hash output.

    Parameters:
        hash_output (str): The hash output as a hexadecimal string.

    Returns:
        float: The Shannon entropy.
    """
    # Convert the hexadecimal hash to bytes
    byte_array = bytes.fromhex(hash_output)
    # Calculate the frequency of each byte
    byte_frequencies = [byte_array.count(byte) for byte in range(256)]
    # Normalize frequencies
    total = sum(byte_frequencies)
    probabilities = [freq / total for freq in byte_frequencies if freq > 0]
    # Compute Shannon entropy
    return entropy(probabilities, base=2)

def test_entropy(test_folder, algorithms):
    """
    Test the entropy of hashing algorithms using files in the test folder.

    Parameters:
        test_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.

    Returns:
        pd.DataFrame: DataFrame containing entropy results.
    """
    results = []

    # Iterate over all files in the folder
    for file_name in sorted(os.listdir(test_folder)):
        file_path = os.path.join(test_folder, file_name)

        # Read file content
        with open(file_path, 'rb') as f:
            file_data = f.read()

        # Test each algorithm
        for algo_name, algo_constructor in algorithms.items():
            hasher = algo_constructor()
            hasher.update(file_data)
            hash_output = hasher.hexdigest()  # Get hash as a hexadecimal string
            hash_entropy = calculate_entropy(hash_output)  # Calculate entropy
            results.append({
                "File Name": file_name,
                "Algorithm": algo_name,
                "Entropy": hash_entropy
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

# Run the entropy tests
entropy_df = test_entropy(test_data_folder, algorithms)

# Create folders for results
os.makedirs("hashing_results", exist_ok=True)
os.makedirs("hashing_visualization", exist_ok=True)

# Save entropy results to a CSV
entropy_df.to_csv("hashing_results/hash_entropy_results.csv", index=False)

# Calculate average entropy and standard deviation per algorithm
entropy_stats = entropy_df.groupby("Algorithm")["Entropy"].agg(["mean", "std"])

# Save entropy statistics to a CSV
entropy_stats.to_csv("hashing_results/hash_entropy_statistics.csv")

# Perform ANOVA test on the entropy values
anova_results = f_oneway(
    *[entropy_df[entropy_df['Algorithm'] == algo]["Entropy"].values for algo in algorithms.keys()]
)

# Save ANOVA results to a text file
with open("hashing_results/hash_entropy_anova_results.txt", "w") as f:
    f.write("ANOVA Results:\n")
    f.write(f"F-statistic: {anova_results.statistic:.4f}\n")
    f.write(f"p-value: {anova_results.pvalue:.4e}\n")

# Display results using tabulate
print("\nEntropy Statistics by Algorithm:")
print(tabulate(entropy_stats, headers=["Algorithm", "Average Entropy", "Std Dev"], tablefmt="grid"))

print("\nANOVA Results:")
print(f"F-statistic: {anova_results.statistic:.4f}, p-value: {anova_results.pvalue:.4e}")

# Visualization
import matplotlib.pyplot as plt

# Plot average entropy with error bars (standard deviation)
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(
    entropy_stats.index,
    entropy_stats["mean"],
    yerr=entropy_stats["std"],
    capsize=5,
    color="skyblue",
    edgecolor="black"
)

# Add labels and formatting
ax.set_title("Average Entropy of Hashing Algorithms", fontsize=14)
ax.set_xlabel("Hashing Algorithm", fontsize=12)
ax.set_ylabel("Entropy (bits)", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save the plot
plt.tight_layout()
plt.savefig("hashing_visualization/hash_entropy_bar_chart.png", dpi=300)

# Show the plot
plt.show()
