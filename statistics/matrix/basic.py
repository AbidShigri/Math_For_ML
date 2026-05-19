# matrix is the collection of vector
# matix is a structured collection of numbers used in ml to store data and perform fast mathematica 
# operation on dataset
# data is feed to model as a matrix
# each row is a sample /datapoint/ vector
# each column is feature
# matrix help for
# fast computation
# handling large dataset
# efficient learning 
# GPU acceleration 

import numpy as np
# Create a dataset matrix (2 samples, 3 features)
matrix = np.array([[1, 8, 3],
                  [9, 2, 5]])

print(f" type {type(matrix)}")
print(f" size {matrix.size}")
print(f" {matrix.ndim}")
print(f" {matrix.shape}")




# zero matrix
z = np.zeros((2, 3)) #this will create zero metrix 
print(f"Zero matrix: {z}") 
# output =[[0. 0. 0.]
#   [0. 0. 0.]]

# identiy matrix
# Why is it called "Identity"?

# Because it behaves like the number 1 in multiplication:

# A⋅I=A

# So if you multiply any matrix with identity matrix → it stays the same.
# 5 × 1 = 5
# A × I = A
# np.eye(3) → more flexible (can create non-square diagonals)
# np.identity(3) → only square identity matrix
identity_m = np.eye(3) #is flexible can create any identity matix

Identity_m = np.identity(3)
print(f"identity matrix {identity_m}")
print(f"identity matrix {Identity_m}")
# identity matrix is ALWAYS square?
# Because identity matrix is defined by this rule:

# A⋅I=A
# This only works when:

# number of rows = number of columns

#  So identity MUST be square.
identity_m = np.eye(3, 5)
# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]]

#  This is NOT identity matrix
#  This is called rectangular identity-like matrix
# It’s used in advanced ML / linear algebra tricks.

# 7. Big mental model (VERY IMPORTANT)
# Square matrix → transformation system (like rotation, scaling)
# Rectangular matrix → data (ML input)
# Identity matrix → “do nothing transformation”