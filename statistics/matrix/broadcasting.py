# Broadcasting is one of the most powerful features in both NumPy and PyTorch. 
# It allows you to perform arithmetic operations (like addition, subtraction, or multiplication)
#  on matrices of different shapes without manually copying data to make them match.
# Instead of wasting memory replicating rows or columns, the underlying library handles it virtually, 
# making the operation incredibly fast.

# Here is a practical Python program showing how broadcasting works when you
#  add a 1D vector to a 2D matrix.
import numpy as np
import torch
print("1: ==> Print numpy broadcasting \n")
# shape of this matrix 3 x 3 3D =(3, 3)
matrix_np = np.array([[2, 1, 3],
                    [5, 3, 1],
                    [2, 4, 6]])
# 1D vector of shape (3, )
vector_np = np.array([3, 5, 4])
print(np.shape(vector_np))

print(f"print 3D matrix of shape 3 x3 (3, 3)\n {matrix_np}")
print("print 1D vector of shape(3,) \n ", vector_np)
result_np = matrix_np @ vector_np
print(f"print broadcasting result => {result_np}")

print("1: ==> Print pytorch(torch) broadcasting \n")

# now creating matix using torch 
# Rank3 tensor or 3D matrix with shape of 3x3 or (3, 3)

print("1: ==> Print troach broadcasting \n")
t_matrix = torch.tensor([[2, 1, 3],
                        [5, 3, 1],
                        [2, 4, 6]])

# Rank1 tensor or 1D vector shape = (3,)
t_vector = torch.tensor([3, 5, 4])

t_result = torch.matmul(t_matrix, t_vector)

print(f"3D tensor or matrix shape = (3, 3) \n {t_matrix}")
print(f"1D tensor or vector shape = (3, ) \n {t_vector}")
print(f"broadcasting result using torch \n {t_result}")
