import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# RCNN 
class CNN0(nn.Module):
    def __init__(self,L):
        super(CNN0,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32*int(L/4)*int(L/4),1)
    
    def forward(self,x):
        x = x.float() 
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x

# ======================================================
# MRCNN
class CNN1(nn.Module):
    def __init__(self,L):
        super(CNN1,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32*int(L/4)*int(L/4),1) 
    
    def forward(self,x):
        x = x.float() 
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x

# ================================
# 1D-CGS
class HybridModel(nn.Module):
    def __init__(self, feature_dim=2, hidden_dim=64):
        super(HybridModel, self).__init__()
        
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1, stride=1),  
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  
            nn.Flatten()  # [N, 32]
        )
        
        self.sage1 = SAGEConv(32, hidden_dim)  
        self.sage2 = SAGEConv(hidden_dim, 64)
        
        self.head = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Graph connectivity [2, num_edges]
        Returns:
            Node importance scores [num_nodes]
        """
        x_cnn = x.unsqueeze(1)  # [N, 1, 2]
        x_shared = self.cnn_feature_extractor(x_cnn)  # [N, 32]
        
        x_sage = F.relu(self.sage1(x_shared, edge_index))  # [N, hidden_dim]
        x_sage = self.sage2(x_sage, edge_index)  # [N, 64]
        
        return self.head(x_sage).squeeze(-1)  # [N]