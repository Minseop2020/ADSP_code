def load_csr_from_mtx(file_path):
#     """
#     Load a CSR matrix from a .mtx file.
    
#     Parameters:
#         file_path (str): Path to the .mtx file containing the matrix data.
        
#     Returns:
#         csr_matrix: The loaded CSR matrix.
#     """
#     # Load the dense matrix from the .mtx file
#     dense_matrix = mmread(file_path)
    
#     # Convert the dense matrix to a CSR matrix
#     csr_data = csr_matrix(dense_matrix)
    
#     return csr_data