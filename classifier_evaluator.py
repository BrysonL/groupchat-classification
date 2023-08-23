import torch
import torch.nn.functional as F

class ClassifierEvaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute_confusion_matrix(self, true, pred):
        n = true.shape[0]
        # Create an index tensor of shape [n, 2] where the first column is the true class and the second is the predicted class
        indices = torch.stack((true, pred), dim=1)
        
        # Use bincount to count occurrences of each pair of indices
        conf_matrix = torch.bincount(indices[:, 0] * self.num_classes + indices[:, 1], minlength=self.num_classes**2)
        
        # Reshape the resulting vector to get the 2D confusion matrix
        conf_matrix = conf_matrix.reshape(self.num_classes, self.num_classes)
        
        return conf_matrix


    def evaluate(self, ys, yhat):
        """
        Evaluates the model's performance.

        Parameters:
        - ys (torch.Tensor): Ground truth labels in one-hot format.
        - yhat (torch.Tensor): Predicted scores or logits for each class.

        Returns:
        - accuracy (float): Classification accuracy.
        - confusion_matrix (torch.Tensor): Confusion matrix.
        """
        # Convert yhat scores/logits to one-hot format by taking the argmax
        yhat_labels = yhat.argmax(dim=1)
        true_labels = ys.argmax(dim=1)

        correct_predictions = (true_labels == yhat_labels).float().sum()
        accuracy = correct_predictions / ys.shape[0]

        confusion_matrix = self.compute_confusion_matrix(true_labels, yhat_labels)
        
        return accuracy.item(), confusion_matrix
