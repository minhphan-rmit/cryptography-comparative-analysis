import os
import hashlib
import pandas as pd
from scipy.stats import chisquare
from tabulate import tabulate

def calculate_byte_distribution(hash_output):
    """
    Calculate the frequency distribution of bytes in a hash output.

    Parameters:
        hash_output (str): The hash output as a hexadecimal string.

    Returns:
        list: Frequency distribution of bytes (0-255).
    """
    # Convert the hexadecimal hash to bytes
    byte_array = bytes.fromhex(hash_output)
    # Count occurrences of each byte (0-255)
    byte_frequencies = [byte_array.count(byte) for byte in range(256)]
    return byte_frequencies

def test_uniform_distribution(test_folder, algorithms):
    """
    Test the uniformity of byte distributions in hashing algorithms.

    Parameters:
        test_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.

    Returns:
        pd.DataFrame: DataFrame containing chi-square test results.
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
            hash_output = hasher.hexdigest()

            # Calculate byte distribution
            observed_frequencies = calculate_byte_distribution(hash_output)

            # Expected uniform distribution
            total_bytes = sum(observed_frequencies)
            expected_frequencies = [total_bytes / 256] * 256

            # Perform Chi-Square Goodness of Fit Test
            chi2_stat, p_value = chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

            results.append({
                "File Name": file_name,
                "Algorithm": algo_name,
                "Chi-Square Statistic": chi2_stat,
                "p-value": p_value
            })

    # Convert results to a DataFrame
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

# Run the uniform distribution tests
uniformity_df = test_uniform_distribution(test_data_folder, algorithms)

# Create folders for results
os.makedirs("hashing_results", exist_ok=True)

# Save detailed results to a CSV
uniformity_df.to_csv("hashing_results/hash_uniformity_test_results.csv", index=False)

# Calculate average Chi-Square Statistic and p-value per algorithm
uniformity_stats = uniformity_df.groupby("Algorithm")[["Chi-Square Statistic", "p-value"]].mean()

# Save summary statistics to a CSV
uniformity_stats.to_csv("hashing_results/hash_uniformity_test_statistics.csv")

# Display results using tabulate
print("\nUniformity Test Statistics by Algorithm:")
print(tabulate(uniformity_stats, headers=["Algorithm", "Avg Chi-Square Statistic", "Avg p-value"], tablefmt="grid"))

# Visualization
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

# Bar chart for average Chi-Square Statistic
ax.bar(
    uniformity_stats.index,
    uniformity_stats["Chi-Square Statistic"],
    color="skyblue",
    edgecolor="black"
)

# Add labels and formatting
ax.set_title("Chi-Square Statistic for Byte Uniformity", fontsize=14)
ax.set_xlabel("Hashing Algorithm", fontsize=12)
ax.set_ylabel("Average Chi-Square Statistic", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save the bar chart to a file
os.makedirs("hashing_visualization", exist_ok=True)
plt.tight_layout()
plt.savefig("hashing_visualization/hash_uniformity_test_chi_square_chart.png", dpi=300)

# Display the plot
plt.show()

print("\nResults saved to 'hashing_results/hash_uniformity_test_results.csv'.")
print("Summary statistics saved to 'hashing_results/hash_uniformity_test_statistics.csv'.")
print("Visualization saved to 'hashing_visualization/hash_uniformity_test_chi_square_chart.png'.")
