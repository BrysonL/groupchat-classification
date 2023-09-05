import torch
from models.model import BaseModel

class MostFrequentClassClassifier(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model_type = "MostFrequentClass"
        self.most_frequent_class = None
        self.num_classes = num_classes
    
    def train_model(self, train_data, train_labels, **kwargs):
        """
        Trains the MostFrequentClassClassifier.
        
        For this classifier, training simply involves identifying the most frequent class 
        in the training labels and storing it.
        """
        # Convert one-hot encoded labels back to class indices
        label_indices = torch.argmax(train_labels, dim=1)
        
        # Identify the most frequent class
        unique, counts = label_indices.unique(return_counts=True)
        self.most_frequent_class = unique[torch.argmax(counts)]
        
        self.trained = True

    def predict(self, data):
        """
        Predicts labels for the given data.
        
        For this classifier, prediction involves returning the most frequent class 
        as one-hot encoded vectors for all the given data points.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")
        
        # Return one-hot encoded predictions
        predictions = torch.zeros(len(data), self.num_classes)
        predictions[:, self.most_frequent_class] = 1
        return predictions