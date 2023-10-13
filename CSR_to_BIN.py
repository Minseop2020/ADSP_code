import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csr_matrix, find

def load_csr_from_mtx(file_path):
    dense_matrix = mmread(file_path)
    csr_data = csr_matrix(dense_matrix)
    return csr_data

def save_csr_to_bin(csr_data, bin_file_path):
    # Convert CSR to dense matrix
    dense_data = csr_data.toarray()
    # Save dense matrix to .bin file
    dense_data.tofile(bin_file_path)
    return dense_data

def load_dense_from_bin(bin_file_path, shape):
    # Load dense matrix from .bin file
    dense_data = np.fromfile(bin_file_path, dtype=float)
    # Reshape the data
    dense_data = dense_data.reshape(shape)
    return dense_data

def plot_csr_matrix_comparison(original, loaded):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    row_orig, col_orig, val_orig = find(original)
    row_loaded, col_loaded, val_loaded = find(loaded)
    
    sc1 = axs[0].scatter(col_orig, -row_orig, c=val_orig, marker='s', cmap='viridis')
    axs[0].set_title('Original Matrix')
    axs[0].set_xlabel('Column Index')
    axs[0].set_ylabel('Row Index')
    
    sc2 = axs[1].imshow(loaded, cmap='viridis', aspect='auto', interpolation='none')
    axs[1].set_title('Binary')   
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    print(sc2)

    plt.colorbar(sc1, ax=axs[0], label='Value')
    plt.colorbar(sc2, ax=axs[1], label='Value')
    
    plt.show()

# Example usage:
csr_original = load_csr_from_mtx('C:\\Work\\School\\Coding\\494_bus\\494_bus_noisy.mtx')

# Save the CSR matrix to .bin file
bin_data = save_csr_to_bin(csr_original, 'C:\\Work\\School\\Coding\\494_bus\\494_bus_noisy.bin')

# Load the dense matrix from .bin file
dense_loaded = load_dense_from_bin('C:\\Work\\School\\Coding\\494_bus\\494_bus_noisy.bin', csr_original.shape)

# Plotting the comparison
plot_csr_matrix_comparison(csr_original, bin_data)