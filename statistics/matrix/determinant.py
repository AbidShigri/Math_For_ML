# Checking Matrix Invertibility: If the determinant is exactly 0, the matrix is "singular,"
#  meaning it cannot be inverted (there is no way to undo the transformation). 
# This causes errors in algorithms like Linear Regression if your data features are redundant.
# Eigenvalues & Optimization: It is used behind the scenes to find eigenvalues,
#  which help compress data in Principal Component Analysis (PCA).
# matrix is invertible or not
# scaling factor of transformation

# Determinant tells how matrix changes area/volume.
# Then matrix expands space.
# ML Meaning

# Expansion helps:

# amplify features
# separate classes
# increase representation power

# Neural networks constantly expand and compress feature spaces.

import numpy as np
import torch

print(f" \n \n Caculating determinant of 2 x 2 matrix manually and with numpy ")
matrix_A = np.array([[3.0, 5.0],
                     [6.0, 1.0]])
# Manual formula for a 2x2 matrix [[a, b], [c, d]] is: (ad - bc)
det_manual = (matrix_A[0, 0] * matrix_A[1, 1]) - (matrix_A[1, 0] * matrix_A[0, 1])
print(det_manual)
print(f"\ndeterminant of matrix by numpy: {det_manual}")

# numpy buildin function for derterminant(under the linear algebra module "linalg")
det_numpy = np.linalg.det(matrix_A)
print(f"\n determinant of matrix by numpy: {det_numpy}")

print(f" \n ===2 Checking if matrix can be inverted ===")
# if determinat is zero matrix has no inverse called singular matrix

matrix_singular = np.array([[1.0, 2.0],
                            [2.0, 4.0]])
det_singular = np.linalg.det(matrix_singular)

print(f"\n Matrix with redundant data: {matrix_singular}")
print(f"\n Determinant: {det_singular}")

if np.isclose(det_singular, 0.0):
    print(f"\n => Determinant is 0! Matrix cannot be inverted")
else:
    print(f"\n => This matrix can be inverted")

print("\n === Matrix Determinant by Pytorch ====")

matrix_ten = torch.tensor([[1.0, 3.0],
                           [2.0, 4.0]])

ten_m_dent = torch.linalg.det(matrix_ten)
#  also can use torch.det()
print(f"Determinant: {ten_m_dent: .1f}")