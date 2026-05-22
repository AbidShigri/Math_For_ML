import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HousingDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32) # most common pytorch dtype is float32
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # unsqueeze(1) adds an extra dimension at position 1 (column dimension)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx): # this tells pytorch how to get one sample from dataset
        return self.x[idx], self.y[idx]


def get_data_loaders(batch_size=32, test_size=0.2, random_state=42): # it defines how to feed data to model
    """
    Generates synthetic regression data, scales it, and returns 
    Train and Test DataLoaders.
    """
    # 1. Generate synthetic housing data (1000 samples, 10 features)
    x, y = make_regression(n_samples=1000, n_features=10, noise=0.5, random_state=random_state)
    
    # 2. Split into train/test data set
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    
    # 3. Scale features (crucial for neural network convergence)
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)
    
    # 4. Create Dataset objects
    train_dataset = HousingDataset(x_train, y_train)
    test_dataset = HousingDataset(x_test, y_test)
    
    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, x_train.shape[1]