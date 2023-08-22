import torch
import torch.nn.functional as F

class ClassifierEvaluator:
    def __init__(self):
        pass

    def evaluate(self, ys, yhat):
        """
        Evaluates the model's performance.

        Parameters:
        - ys (torch.Tensor): Ground truth labels in one-hot format.
        - yhat (torch.Tensor): Predicted scores or logits for each class.

        Returns:
        - accuracy (float): Classification accuracy.
        """
        # Convert yhat scores/logits to one-hot format by taking the argmax
        yhat_one_hot = F.one_hot(yhat.argmax(dim=1), num_classes=ys.shape[1])

        correct_predictions = (ys == yhat_one_hot).all(dim=1).float().sum()
        accuracy = correct_predictions / ys.shape[0]

        return accuracy.item()