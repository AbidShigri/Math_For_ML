# x = 5
# print(type(x))

# 1. The Core Concept (Keep it Simple)In machine learning, a scalar is just a single number. 
# It has no direction, no rows, and no columns. It is a Tensor of Rank 0 (or dimension 0).
# In Math: Written as a lowercase letter (e.g., $x = 5$).
# In Python: Just a standard float or int (e.g., x = 5.0).
# 1. To Represent Single Quantities

# Many things in ML are just one value.
# Examples:
# Age = 20
# Temperature = 36.5
# Salary = 50000
# Loss = 0.25
# Each of these is a scalar.
# 2. Model Output Can Be Scalar
# 3. Loss Function Gives a Scalar

import numpy as np
x = np.array(5) #This creates a NumPy array with a single value (5).
# x : It is not a normal Python integer
# It becomes a 0-dimensional NumPy array (scalar array)
print(x)
print(f"this is a 0 dimension numpy array or 0 rank tensor: {x.shape}") #output = () The empty shape () means scalar (0D).or () This means 0-dimensional array

# .shape is used to know the structure (dimensions) of data inside a NumPy array.
# It tells you:
# how many dimensions the array has
# how many values exist in each dimension

learning_rate = 0.01 #sclar
loss = 0.25 #sclar
print(type(loss))

v = np.array([5])
print(v) #vector or 
print(f" {v.shape}") #output (1,)meas one dimention numpy array or tensor , contain one row of element

v = np.array([1, 2, 3])
print(v)
print(f" this is a 1 dimension numpy array or rank 1 tensor: {v.shape}" ) # 


# Scalar with torch library 
# Torch is the core component or library of pytorch(framework of ML and DL to train and build model with mathe and tensor etc)
# Torch is used for math and tensor

# create a scalar tensor
import torch 
scalar_tensor = torch.tensor(5.0) #it will create a scalar tensor
print(scalar_tensor)
print(f"rank of tensor: {scalar_tensor.ndim}") # output 0 (Rank0 tensor) ,b/c it has no dimension
   #ndim means number of dimension /How many axes/dimensions a tensor or array has.
print(f"dimention of scalar is: {scalar_tensor.shape}") # output: torch.size([]) means array with 0 dimension

# 2. Extract the raw Python number from a loss value
# (Crucial for tracking loss without melting your GPU memory)
loss_tensor = torch.tensor(0.2451)
print(loss_tensor) #output will be tensor(0.2451)
print(type(loss_tensor)) #here "torch.Tensor" show in outpu
raw_loss = loss_tensor.item() #.item() converts a tensor containing one value into a normal Python number.
print(f"raw_loss: {raw_loss}") #output  0.2451000064611435
print(type(raw_loss)) #output : class float 
