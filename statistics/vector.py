#          === VECTORER For ML ========
# A vector is one of the most important concepts in Machine Learning.
# If scalars are “single numbers”, then vectors are “collections of numbers arranged in order”.
# A vector can be collection of scalar, 1 dimension array, Rank 1 tensor or list of numbers
# for the vector operation two library will be used numpy and torch via Pytorch framwork

#  ====== Programing Practice ==========
import numpy as np
vector_array = np.array([1, 2, 3, 4, 5])
print(type(vector_array)) #output => <class 'numpy.ndarray'> n = number, d = dimenstional
print(vector_array.shape) # output => (5, ) 1 dimenstion array containing 5 elements or
# (n,) means in numpy Rank 1 array or tensor  (called: vector)
print(vector_array.ndim) # output give no of dimension
print(vector_array.size) # how many value inside the array


# Step 3 — Learn Operations

# zero matrix
z = np.array()