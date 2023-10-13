import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, find



def load_csr_from_mtx(file_path):
    """
    Load a CSR matrix from a .mtx file.
    
    Parameters:
        file_path (str): Path to the .mtx file containing the matrix data.
        
    Returns:
        csr_matrix: The loaded CSR matrix.
    """
    # Load the dense matrix from the .mtx file
    dense_matrix = mmread(file_path)
    
    # Convert the dense matrix to a CSR matrix
    csr_data = csr_matrix(dense_matrix)
    
    return csr_data

def add_gaussian_noise_to_csr(csr_data, percentage, noise_ratio):
    """
    Add Gaussian noise to a percentage of non-zero elements in a CSR matrix.
    
    Parameters:
        csr_data (csr_matrix): The original CSR matrix.
        percentage (float): Percentage of non-zero elements to add noise to.
        noise_ratio (float): Ratio of the Gaussian noise to the original data.
        
    Returns:
        csr_matrix: The CSR matrix with added Gaussian noise.
    """
    # Extract non-zero elements and their indices from the CSR matrix
    data = np.array(csr_data.data, dtype=float)
    indices = csr_data.indices
    indptr = csr_data.indptr
    
    # Determine the number of non-zero elements to add noise to
    num_noisy_elements = int(len(data) * percentage / 100)
    print("num_noisy_elements:")
    print(num_noisy_elements)
    
    # Randomly select indices of the elements to add noise to
    noisy_indices = np.random.choice(len(data), num_noisy_elements, replace=False)
    
    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=noise_ratio, size=num_noisy_elements) * data[noisy_indices]
    print("noise:")
    print(noise)
    
    # Add Gaussian noise to the selected elements
    data[noisy_indices] += noise

    
    # Create a new CSR matrix with the noisy data
    noisy_csr = csr_matrix((data, indices, indptr), shape=csr_data.shape)


    return noisy_csr

def save_csr_to_mtx(csr_data, file_path):
    mmwrite(file_path, csr_data)


def plot_csr_matrix_comparison(original, noisy):
    """
    Plot a comparison between the original and noisy CSR matrices using scatter plots.
    
    Parameters:
        original (csr_matrix): The original CSR matrix.
        noisy (csr_matrix): The noisy CSR matrix.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Find the row indices, column indices, and values of the non-zero elements
    row_orig, col_orig, val_orig = find(original)
    row_noisy, col_noisy, val_noisy = find(noisy)
    
    # Plot original matrix
    sc1 = axs[0].scatter(col_orig, -row_orig, c=val_orig, marker='s', cmap='viridis')
    axs[0].set_title('Original Matrix')
    axs[0].set_xlabel('Column Index')
    axs[0].set_ylabel('Row Index')
    
    # Plot noisy matrix
    sc2 = axs[1].scatter(col_noisy, -row_noisy, c=val_noisy, marker='s', cmap='viridis')
    axs[1].set_title('Noisy Matrix')
    axs[1].set_xlabel('Column Index')
    axs[1].set_ylabel('Row Index')
    
    # Add colorbars
    plt.colorbar(sc1, ax=axs[0], label='Value')
    plt.colorbar(sc2, ax=axs[1], label='Value')
    
    # Display the plots
    plt.show()


# Example usage:
# Creating a CSR matrix
# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
# csr_original = csr_matrix((data, (row, col)), shape=(3, 3))
csr_original = load_csr_from_mtx('C:\\Work\\School\\Coding\\494_bus\\494_bus.mtx')

# Adding Gaussian noise
percentage = 1  # Percentage of non-zero elements to add noise to
noise_ratio = 0.1  # Ratio of the Gaussian noise to the original data
csr_noisy = add_gaussian_noise_to_csr(csr_original, percentage, noise_ratio)

# Displaying the original and noisy CSR matrices
print("Original CSR matrix:")
print(csr_original)
print("\nNoisy CSR matrix:")
print(csr_noisy)

# Save the noisy CSR matrix to .mtx file
save_csr_to_mtx(csr_noisy, 'C:\\Work\\School\\Coding\\494_bus\\494_bus_noisy.mtx')

# Plotting the comparison
plot_csr_matrix_comparison(csr_original, csr_noisy)