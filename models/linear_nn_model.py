import torch
from models.model import BaseModel
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import itertools

class MultiClassNNModel(BaseModel):
    def __init__(self, num_features, num_classes, include_interaction=False):
        super().__init__()
        self.model_type = "MultiClassLogisticNN"
        self.include_interaction = include_interaction

        if self.include_interaction:
            self.net = nn.Linear(num_features + int(num_features*(num_features-1)/2), num_classes)  # Interaction features included - increase number of inputs
        else:
            self.net = nn.Linear(num_features, num_classes)  # No interaction features

        self.criterion = nn.CrossEntropyLoss() # use cross entropy loss as our loss function
        self.optimizer = optim.Adam(self.net.parameters()) # use Adam as our optimizer (idk what that is yet, but it's standard)

        # use the same standard score normalization that we did in the linear model
        self.feature_means = None
        self.feature_stds = None

    def _normalize_data(self, data):
        if self.feature_means is None or self.feature_stds is None:
            raise RuntimeError("Model must be trained before normalization.")
        return (data - self.feature_means) / self.feature_stds

    def _add_interaction_features(self, data):
        interactions = [data[:, i] * data[:, j] for i, j in itertools.combinations(range(data.shape[1]), 2)]
        interactions = [inter.unsqueeze(1) for inter in interactions]  # Convert each 1D tensor to 2D tensor with shape [n, 1]
        return torch.cat([data] + interactions, dim=1)

    def train_model(self, train_data, train_labels, epochs=100, batch_size=128):
        self.trained = False
        
        # Convert the data to tensors
        train_data = torch.FloatTensor(train_data)
        
        # Conditionally add interaction features
        if self.include_interaction:
            train_data = self._add_interaction_features(train_data)
        
        # Compute and store the normalization parameters
        self.feature_means = train_data.mean(dim=0)
        self.feature_stds = train_data.std(dim=0)
        
        # To avoid division by zero, replace any zero std values with 1
        self.feature_stds[self.feature_stds == 0] = 1
        
        # Normalize the training data
        train_data = self._normalize_data(train_data)
        
        train_labels_indices = torch.argmax(train_labels, dim=1)
        train_labels = torch.LongTensor(train_labels_indices)
        
        # Create DataLoader for mini-batch processing
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the scheduler to reduce the learning rate over time
        scheduler = StepLR(self.optimizer, step_size=100, gamma=0.99)  # Reduces the learning rate by a factor of 0.99 every 100 batches

        for epoch in range(epochs):
            # Every epoch, iterate through the entire dataset in batches
            for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
                self.optimizer.zero_grad() # zero the gradients
                output = self.net(batch_data) # forward pass
                loss = self.criterion(output, batch_labels) # calculate loss function
                loss.backward() # calculate gradients
                self.optimizer.step() # update weights
            
                # Step the learning rate scheduler in the batch loop to reduce LR more quickly
                scheduler.step()

            # Every 10 epochs, print the loss and current learning rate
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Current LR: {scheduler.get_last_lr()[0]}')
        
        self.trained = True

    def predict(self, data):
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")
        
        # Convert to tensor and normalize the data
        data = torch.FloatTensor(data)

        # Conditionally add interaction features
        if self.include_interaction:
            data = self._add_interaction_features(data)
        
        data = self._normalize_data(data)
        
        with torch.no_grad():
            output = self.net(data)
            probabilities = F.softmax(output, dim=1)  # Apply softmax to convert output to probabilities
        
        return probabilities
