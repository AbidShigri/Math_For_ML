# In Machine Learning, matrix addition and subtraction are the quiet workhorses.
#  They might seem basic compared to matrix multiplication, but you cannot train a neural network, 
# optimize a model, or preprocess data without them
#  The Rule (The "How" and "When")You can only add or subtract two matrices
#  if they have the exact same dimensions (shape). If Matrix $A$ is $3 \times 2$, Matrix $B$ must 
# also be $3 \times 2$.
# The operation itself is simple: you just add or subtract the corresponding elements (element-wise).

import numpy as np
m1 = np.array([[1, 2, 3],  #here 2x3 matrix
               [3, 4, 5]])
m2 = np.array([[4, 5, 2],
               [1, 4, 6]])

print(f"addition of two matrix is: \n {m1 + m2}")
print(f"subtraction of matrix is: \n {m1 - m2}")


# when addition and subtraction used in ML
# A. Adding the "Bias" in Neural Networks (Addition)
# B. Gradient Descent: Updating Weights (Subtraction)
# C. Calculating Loss/Error (Subtraction)
# D. Data Normalization and Mean Centering (Subtraction)

import numpy as np
import math 
# calculate model error
y_actual = np.array([100, 200, 300])
y_pred = np.array([90, 210, 312])

loss = (y_actual - y_pred) 
MSE = (y_actual - y_pred)**2
print(f"how much model is wrong (Loss): {loss}")

# 2. Layer Output with Bias Addition (Broadcasting)

# The NumPy Exception: Broadcasting
# In Python, NumPy allows a special shortcut called broadcasting. If you try to add a single row vector or
#  a single number (scalar) to a massive matrix, NumPy automatically "stretches" or copies that row across 
#  the whole matrix so the shapes match.
# A 2x2 matrix of weighted inputs (X * W

xw_output = np.array([[5.0, 2.0],
                      [4.0, 3.0]])

# 1x2 bias vector
bias = np.array([0.5, -1.0])
# NumPy automatically adds the bias row to BOTH rows of xw_output
final_output = xw_output + bias
print(f"\nFinal Output with Bias Added:\n {final_output}")


import numpy as np

# Actual values (3 houses, 2 features each) -> A 3x2 Matrix
# Column 0: Actual Price ($K), Column 1: Actual SqFt (Hundreds)
y_actual1 = np.array([[100, 14],
                      [200, 20],
                      [300, 29]])
y_pred1 = np.array([[90, 11], #prediction of house 1
                    [210, 21], #prediction of house2
                    [320, 33]]) #prediction of house3

# Matrix Subtraction (Element-wise)
# Both matrices are exactly 3x2, so they match perfectly.
error_m = y_actual1 - y_pred1
print("Error Matrix:\n", error_m)
square_error = np.square(error_m)
print(square_error)
# Calculate MSE (The single number we try to minimize)
mse = np.mean(square_error)
print(mse)
print(f"final mse loss: {mse:.2f}")
#  :.2f this called format specification
#  : start for formating
#  .2 show two digit after two decimal
# f floating-point number