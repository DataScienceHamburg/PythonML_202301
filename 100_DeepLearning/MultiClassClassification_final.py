#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import
iris = load_iris()
X = iris.data
y = iris.target

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# %% dataset
class IrisData(Dataset):
    def __init__(self, X_train, y_train) -> None:
        super().__init__()
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.y = self.y.type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len



iris_data = IrisData(X_train=X_train, y_train=y_train)

# %% dataloader
train_loader = DataLoader(iris_data, batch_size=2)

# %% check dims
iris_data.X.shape
# %% define class
class MultiClassNet(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x


# %% hyper parameters
# NUM_FEATURES = ...
# HIDDEN = ...
# NUM_CLASSES = ...
# %% create model instance

# %% loss function
criterion = nn.CrossEntropyLoss()
# %% optimizer

# %% training
     
# %% show losses over epochs


# %% test the model


# %% Accuracy

