# A matrix multiplication is a mathematical operation where we combine two matrices to produce a new matrix.
# It is one of the most important operations in Machine Learning because ML models process data mostly in the form of matrices.
# 2. Condition for Matrix Multiplication
# Matrix multiplication only works when:
# columns of first matrix=rows of second matrix

# 3. Why Matrix Multiplication is Important in ML

# Machine Learning models process:

# features
# weights
# images
# embeddings
# neural networks

# using matrix multiplication.

# Because it allows:

# many calculations together
# very fast computation
# GPU acceleration
# compact representation of data


# Program practice/ or implementation

import numpy as np
import torch
A = np.array([[1, 2, 3],
              [4, 5, 5]])
B = np.array([[2, 3],
              [4, 5],
              [8, 4]])
# using numpy
mul_m = A @ B # The '@' operator is the modern standard for matrix/vector multiplication
# mul_m1 = A * B
# print(f"matrix multiplicaiton by element wise \n {mul_m1}")
print(f"\n multiplication matrix A and B:\n {mul_m} \n")
# using pytorch (torch)

t1 = torch.tensor([[1, 2, 3],
              [4, 5, 5]])
t2 = torch.tensor([[2, 3],
              [4, 5],
              [8, 4]])
output = torch.matmul(t1, t2)
print(f"matric multiplication by using torch: \n {output}")
