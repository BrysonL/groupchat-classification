from datetime import datetime, timezone, timedelta
from message import Message
import torch

class TextMessage(Message):
    def __init__(self, sender_name, timestamp_ms, content):
        super().__init__(sender_name, timestamp_ms)
        self.content = content
        self.message_type = "text"

    def __str__(self):
        est = timezone(timedelta(hours=-5))
        # Convert from milliseconds to seconds and then to a datetime object
        dt = datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc).astimezone(est)
        formatted_time = dt.strftime('%B %d, %Y %I:%M.%S %p %Z')
        return (
            f"Sender: {self.sender}\n"
            f"Timestamp: {formatted_time}\n"
            f"Message: {self.content}"
        )
    
    def get_sender_vector(self):
        """
        Returns a one-hot encoded vector representing the sender using torch.
        """
        # Using torch for numerical operations
        vector = torch.zeros(len(self.VALID_SENDERS))
        sender_index = self.VALID_SENDERS.index(self.sender)
        vector[sender_index] = 1
        return vector

    def extract_features(self):
        """
        Extracts features from the message content using torch.
        For now:
          - Feature 1: Message length
          - Feature 2: Number of words
        """
        # Extract message length
        message_length = len(self.content)
        
        # Extract number of words
        num_words = len(self.content.split())

        return torch.tensor([message_length, num_words])