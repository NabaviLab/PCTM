import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformer import *

# Custom FocalLoss definition
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.45, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class TwinRESTransformer(nn.Module):
    pass

dataset = "priorCurrent/"
# Main training function
def main():
    # Initialize the model and set up devices and parameters
    model = TwinRESTransformer().to(device)
    criterion = FocalLoss(alpha=0.45, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    dataloader = DataLoader(dataset, batch_size=4)

    # Training loop
    model.train()
    for epoch in range(60):  # 60 epochs for example
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
