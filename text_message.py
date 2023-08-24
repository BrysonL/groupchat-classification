from datetime import datetime, timezone, timedelta
from message import Message
import torch
from emoji import unicode_codes
from textblob import TextBlob

class TextMessage(Message):
    def __init__(self, sender_name, timestamp_ms, content):
        super().__init__(sender_name, timestamp_ms)
        if len(content) > 1401:
            self.content = content[:1401] # Truncate messages longer than 1401 characters - this only impacts 3 out of 56k messages in my dataset
            # print(f"Truncated message from {self.sender} at {datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc).astimezone(timezone(timedelta(hours=-5))) } with length {len(content)} and content: {content}")
        else:
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
        vector = torch.zeros(len(self.SENDER_ALIASES))
        sender_index = self.SENDER_ALIASES.index(self.sender)
        vector[sender_index] = 1
        return vector

    def extract_features(self):
        """
        Extracts features from the message content using torch.
        Features:
        - Feature 1: Message length
        - Feature 2: Number of words
        - Feature 3: Sentiment Score (Polarity)
        - Feature 4: Presence of URLs
        - Feature 5: Presence of Numbers
        - Feature 6: Message Hour
        - Feature 7: Day of the Week
        - Feature 8: Presence of Question Marks
        - Feature 9: Month of the message
        - Feature 10: Number of lowercase letters
        - Feature 11: Ratio of lowercase to all letters
        - Feature 12: Average word length
        """
        # Feature 1: Extract message length
        message_length = len(self.content)
        
        # Feature 2: Extract number of words
        num_words = len(self.content.split())

        # Feature 3: Sentiment Score (Polarity)
        blob = TextBlob(self.content)
        sentiment_score = blob.sentiment.polarity

        # Feature 4: Presence of URLs
        has_url = 1 if "http://" in self.content or "https://" in self.content else 0

        # Feature 5: Presence of Numbers
        has_number = 1 if any(char.isdigit() for char in self.content) else 0

        # Feature 6: Message Hour
        est = timezone(timedelta(hours=-5))
        dt = datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc).astimezone(est)
        message_hour = dt.hour

        # Feature 7: Day of the Week (0=Monday, 6=Sunday)
        day_of_week = dt.weekday()

        # Feature 8: Presence of Question Marks
        has_question = 1 if "?" in self.content else 0

        # Feature 9: Month of the message
        message_month = dt.month

        # Feature 10: Number of lowercase letters
        num_lowercase = sum(1 for char in self.content if char.islower())

        # Feature 11: Ratio of lowercase to all letters
        lowercase_ratio = num_lowercase / message_length

        # Feature 12: Average word length
        words = self.content.split()
        avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
        

        return torch.tensor([message_length, num_words, sentiment_score, has_url, has_number, message_hour, day_of_week, has_question, message_month, num_lowercase, lowercase_ratio, avg_word_length])
    