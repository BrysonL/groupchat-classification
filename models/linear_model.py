import torch
from scipy.optimize import minimize
from models.model import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

class MultiClassLinearModel(BaseModel):
    def __init__(self, num_classes, include_interaction=False, degree=2):
        super().__init__()
        self.model_type = "MulticlassLinear"
        self.num_classes = num_classes
        # Instead of storing weights, store models for each class
        self.models = [LogisticRegression(max_iter=1000, C=100.0) for _ in range(num_classes)]
        self.interaction = include_interaction
        if self.interaction:
            self.poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)


    def _binary_train(self, train_data, train_labels):
        """
        Trains a binary linear classifier using scikit-learn's LogisticRegression.
        """
        # Train the model
        model = LogisticRegression(max_iter=1000, C=100.0)  # You can adjust hyperparameters as needed
        model.fit(train_data, train_labels)
        return model
    
    def train_model(self, train_data, train_labels, **kwargs):
        """
        Trains the MultiClassLinearModel using the one-vs-all strategy.
        """
        train_data = train_data.numpy().astype('float32')  # Convert torch tensor to numpy array for sklearn

        # Compute and store the normalization parameters
        self.feature_means = train_data.mean(axis=0)
        self.feature_stds = train_data.std(axis=0)
        
        # To avoid division by zero, replace any zero std values with 1
        self.feature_stds[self.feature_stds == 0] = 1
        
        # Normalize the training data
        train_data = (train_data - self.feature_means) / self.feature_stds
        
        if self.interaction:
            # Add interaction features
            train_data = self.poly.fit_transform(train_data)

        for i in range(self.num_classes):
            # For each class, set labels to 1 for that class and 0 for all others
            print(f"Training binary classifier for class {i}")
            binary_labels = train_labels[:, i].numpy()
            self.models[i] = self._binary_train(train_data, binary_labels)
        self.trained = True

    def predict(self, data):
        """
        Predicts the class label for given data.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")
        
        data = data.numpy().astype('float32')
        # Normalize data using the stored parameters
        data = (data - self.feature_means) / self.feature_stds

        if self.interaction:
            # Add interaction features
            data = self.poly.transform(data)
        
        # Compute scores for each class using the trained models
        scores = [model.decision_function(data) for model in self.models]
        
        # Convert scores to torch tensor and return tensor of predictions
        return torch.tensor(scores).T