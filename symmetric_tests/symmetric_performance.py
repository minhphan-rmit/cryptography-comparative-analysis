import os
import time
from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Configure plot style
rcParams['font.family'] = 'serif'  # Use serif font


def measure_initialization_time(algorithm, key_size=None):
    """
    Measure the time to initialize the encryption engine.

    Parameters:
        algorithm: Encryption algorithm (AES, 3DES, ChaCha20, or Camellia).
        key_size (int): Key size in bits for AES, Camellia, or ChaCha20.

    Returns:
        float: Initialization time in microseconds.
    """
    try:
        start_time = time.time()
        if algorithm == TripleDES:
            key = os.urandom(24)
            iv = os.urandom(8)
            Cipher(TripleDES(key), modes.CFB(iv), backend=default_backend())
        elif algorithm in [algorithms.AES, algorithms.Camellia]:
            key = os.urandom(key_size // 8)
            iv = os.urandom(16)
            Cipher(algorithm(key), modes.CFB(iv), backend=default_backend())
        elif algorithm == algorithms.ChaCha20:
            key = os.urandom(32)
            nonce = os.urandom(16)
            Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
        else:
            raise ValueError("Unsupported algorithm")
        return (time.time() - start_time) * 1e6  # Convert to microseconds
    except Exception as e:
        print(f"Error initializing algorithm {algorithm}: {e}")
        return None


def measure_block_processing_time(algorithm, block_size, key_size=None):
    """
    Measure the time to process a single block of data.

    Parameters:
        algorithm: Encryption algorithm (AES, 3DES, ChaCha20, or Camellia).
        block_size (int): Size of the block to process in bytes.
        key_size (int): Key size in bits for AES, Camellia, or ChaCha20.

    Returns:
        float: Block processing time in microseconds.
    """
    try:
        key = os.urandom(key_size // 8 if key_size else 24)
        if algorithm == TripleDES:
            iv = os.urandom(8)
            cipher = Cipher(TripleDES(key), modes.CFB(iv), backend=default_backend())
        elif algorithm in [algorithms.AES, algorithms.Camellia]:
            iv = os.urandom(16)
            cipher = Cipher(algorithm(key), modes.CFB(iv), backend=default_backend())
        elif algorithm == algorithms.ChaCha20:
            nonce = os.urandom(16)
            cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
        else:
            raise ValueError("Unsupported algorithm")

        encryptor = cipher.encryptor()
        block = os.urandom(block_size)

        start_time = time.time()
        encryptor.update(block)
        return (time.time() - start_time) * 1e6  # Convert to microseconds
    except Exception as e:
        print(f"Error processing block for algorithm {algorithm}: {e}")
        return None


def measure_key_scheduling_time(algorithm, key_size=None):
    """
    Measure the time required for key scheduling.

    Parameters:
        algorithm: Encryption algorithm (AES, 3DES, ChaCha20, or Camellia).
        key_size (int): Key size in bits for AES, Camellia, or ChaCha20.

    Returns:
        float: Key scheduling time in microseconds.
    """
    try:
        key = os.urandom(key_size // 8 if key_size else 24)
        if algorithm == TripleDES:
            iv = os.urandom(8)
            start_time = time.time()
            Cipher(TripleDES(key), modes.CFB(iv), backend=default_backend())
        elif algorithm in [algorithms.AES, algorithms.Camellia]:
            iv = os.urandom(16)
            start_time = time.time()
            Cipher(algorithm(key), modes.CFB(iv), backend=default_backend())
        elif algorithm == algorithms.ChaCha20:
            nonce = os.urandom(16)  # ChaCha20 requires a 16-byte nonce
            start_time = time.time()
            Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
        else:
            raise ValueError("Unsupported algorithm")
        return (time.time() - start_time) * 1e6  # Convert to microseconds
    except Exception as e:
        print(f"Error during key scheduling for algorithm {algorithm}: {e}")
        return None


def measure_file_processing_time(file_path, algorithm, key_size=None):
    """
    Measure the time to process an entire file (encryption).

    Parameters:
        file_path (str): Path to the file to process.
        algorithm: Encryption algorithm (AES, 3DES, ChaCha20, or Camellia).
        key_size (int): Key size in bits for AES, Camellia, or ChaCha20.

    Returns:
        float: File processing time in seconds.
    """
    try:
        key = os.urandom(key_size // 8 if key_size else 24)
        if algorithm == TripleDES:
            iv = os.urandom(8)
            cipher = Cipher(TripleDES(key), modes.CFB(iv), backend=default_backend())
        elif algorithm in [algorithms.AES, algorithms.Camellia]:
            iv = os.urandom(16)
            cipher = Cipher(algorithm(key), modes.CFB(iv), backend=default_backend())
        elif algorithm == algorithms.ChaCha20:
            nonce = os.urandom(16)
            cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
        else:
            raise ValueError("Unsupported algorithm")

        encryptor = cipher.encryptor()
        start_time = time.time()
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):  # 64 KB chunks
                encryptor.update(chunk)
        encryptor.finalize()
        return time.time() - start_time  # Time in seconds
    except Exception as e:
        print(f"Error processing file {file_path} with algorithm {algorithm}: {e}")
        return None


def test_performance(data_folder, algorithms, max_file_size_mb=20):
    """
    Test performance metrics for all algorithms and files.

    Parameters:
        data_folder (str): Path to the folder containing test files.
        algorithms (dict): Dictionary of algorithm names and constructors.
        max_file_size_mb (int): Maximum file size in MB to process.

    Returns:
        pd.DataFrame: DataFrame containing performance metrics.
    """
    results = []
    max_file_size_bytes = max_file_size_mb * 1024 * 1024

    for file_name in sorted(os.listdir(data_folder)):
        file_path = os.path.join(data_folder, file_name)
        file_size = os.path.getsize(file_path)

        if file_size > max_file_size_bytes:
            continue

        for algo_name, (algo_constructor, key_size) in algorithms.items():
            initialization_time = measure_initialization_time(algo_constructor, key_size)
            block_time = measure_block_processing_time(algo_constructor, 16, key_size)  # 16-byte block
            key_schedule_time = measure_key_scheduling_time(algo_constructor, key_size)
            file_time = measure_file_processing_time(file_path, algo_constructor, key_size)
            results.append({
                "File Size (bytes)": file_size,
                "Algorithm": algo_name,
                "Initialization Time (µs)": initialization_time,
                "Block Processing Time (µs)": block_time,
                "Key Scheduling Time (µs)": key_schedule_time,
                "File Processing Time (s)": file_time
            })

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

# Measure performance metrics
results_df = test_performance(test_data_folder, algorithms_to_test)

# Save results
results_folder = "symmetric_results_performance"
os.makedirs(results_folder, exist_ok=True)
results_csv_path = os.path.join(results_folder, "performance_metrics_results.csv")
results_df.to_csv(results_csv_path, index=False)

# Print completion message
print(f"Performance metrics saved to: {results_csv_path}")
