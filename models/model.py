class BaseModel:
    def __init__(self):
        self.model_type = None
        self.trained = False
        # We'll add more variables later as we identify commonalities between models

    def train_model(self, train_data, train_labels, **kwargs):
        raise NotImplementedError("train_model method must be implemented by subclasses.")

    def predict(self, data):
        raise NotImplementedError("predict method must be implemented by subclasses.")

    # Optional: Save and load model methods. This will be useful once we get to more advanced models that take a while to train.
    def save_model(self, path):
        raise NotImplementedError("save_model method must be implemented by subclasses.")

    def load_model(self, path):
        raise NotImplementedError("load_model method must be implemented by subclasses.")
