import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# STEP 1: "convert actual or real world data... into vector"
# ==========================================
# Let's simulate 4 houses. Each house has 2 features: [Size in sq ft, Age in years]
# This is our raw real-world data matrix (Shape: [4, 2])
X_raw = np.array([
    [1500.0, 5.0],
    [2500.0, 2.0],
    [1200.0, 15.0],
    [3000.0, 1.0]
], dtype=np.float32)

# These are the actual target prices in thousands of dollars (Our true 'y' scalars)
y_raw = np.array([[300.0], [500.0], [200.0], [600.0]], dtype=np.float32)


# ==========================================
# STEP 2: "prepare that data... by scalarstandard, normalization"
# ==========================================
# Neural networks hate huge differences in scales (e.g., 3000 vs 1). 
# We use sklearn's StandardScaler to make the mean 0 and variance 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Convert our cleaned NumPy matrices into PyTorch Tensors
X_tensor = torch.tensor(X_scaled)
y_tensor = torch.tensor(y_raw)


# ==========================================
# STEP 3: "apply rule like y = mx + b" (The Neural Network)
# ==========================================
# We define a network with 1 layer that takes 2 inputs and outputs 1 prediction
class LinearNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear automatically creates the weight matrix (m) and bias vector (b)
        self.layer = nn.Linear(in_features=2, out_features=1)
        
    def forward(self, x):
        # This line performs the exact math: y = xW^T + b
        return self.layer(x)

# Instantiate our model
model = LinearNeuralNetwork()


# ==========================================
# STEP 4: "calculate error, or loss, gradient descent, update weight, mse"
# ==========================================
# 4a. Define Mean Squared Error (MSE) as our loss calculator
criterion = nn.MSELoss()

# 4b. Define Stochastic Gradient Descent (SGD) as our optimizer to update weights
# We give it our model's parameters (weights and biases) and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("--- Starting the Training Loop ---")

# We train for 100 epochs (loops over the data)
for epoch in range(100):
    # 1. Forward Pass: Make a prediction using the current weight and bias rules
    predictions = model(X_tensor)
    
    # 2. Calculate Error: Use MSE to see how far off our predictions are from actual prices
    loss = criterion(predictions, y_tensor)
    
    # 3. Gradient Descent: Calculate the slope of the error (gradients)
    optimizer.zero_grad() # Clear out old gradients first
    loss.backward()       # Backpropagation: calculate gradients using the chain rule
    
    # 4. Update Weights: Move the weights and biases in the direction that lowers the error
    optimizer.step()
    
    # Print the error every 20 steps to watch it learn
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Current MSE Loss: {loss.item():.4f}")

print("\nTraining complete! The network has learned its own optimized rules.")