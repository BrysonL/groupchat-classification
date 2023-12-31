from datetime import datetime, timezone, timedelta
from message import Message
import torch
from emoji import unicode_codes
from textblob import TextBlob

class TextMessage(Message):
    def __init__(self, sender_name, timestamp_ms, content):
        super().__init__(sender_name, timestamp_ms)
        self.content = content
        self.message_type = "text"

        self.features = None
        self.onehot_label = None

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
        if self.onehot_label is None:
            vector = torch.zeros(len(self.SENDER_ALIASES))
            sender_index = self.SENDER_ALIASES.index(self.sender)
            vector[sender_index] = 1
            self.onehot_label = vector
        return self.onehot_label

    def extract_features(self, prev_msg=None):
        """
        Extracts features from the message content using torch.
        Features (note: with conversion to onehot, the number no longer corresponds to the feature's index in the feature vector, and it is too complicated to keep tracking):
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
        - Feature 13: Previous sender
        - Feature 14: Previous message was a question
        - Feature 15: Time since last message
        - Feature 16: Previous message more than 200 seconds ago
        """
        if self.features is not None: # if we've already extracted the features, don't extract them again
            return self.features
        
        num_senders = len(self.SENDER_ALIASES)

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
        num_letters = sum(1 for char in self.content if char.isalpha())

        # Feature 11: Ratio of lowercase to all letters
        lowercase_ratio = num_lowercase / num_letters if num_letters > 0 else 0

        # Feature 12: Average word length
        words = self.content.split()
        avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0

        if prev_msg:
            sender_onehot = torch.zeros(num_senders)
            # Feature 13: Previous sender
            sender_onehot[prev_msg.sender_idx] = 1
            
            # Feature 14: Previous message was a question
            prev_msg_was_question = 1 if "?" in prev_msg.content else 0
            
            # Feature 15: Time since last message
            time_since_last_msg = (self.timestamp - prev_msg.timestamp) / 1000

            # Feature 16: Previous message more than 20 seconds ago
            prev_msg_more_than_200s_ago = 1 if time_since_last_msg > 200 else 0
        else:
            sender_onehot = torch.zeros(num_senders)
            prev_msg_was_question = 0
            time_since_last_msg = 0  # or a large arbitrary value if you prefer
            prev_msg_more_than_200s_ago = 0

        days_onehot = torch.zeros(7)
        days_onehot[day_of_week] = 1

        month_onehot = torch.zeros(12)
        month_onehot[message_month - 1] = 1  # -1 since month range is 1-12 and index is 0-based

        self.features = torch.cat([
            torch.tensor([message_length, num_words, sentiment_score, has_url, has_number, message_hour, has_question, num_lowercase, lowercase_ratio, avg_word_length, prev_msg_was_question, time_since_last_msg, prev_msg_more_than_200s_ago]),
            sender_onehot,
            days_onehot,
            month_onehot
        ])
        return self.features
    