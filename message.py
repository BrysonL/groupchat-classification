from datetime import datetime, timezone, timedelta
from functools import total_ordering
import os
import dotenv

dotenv.load_dotenv()

@total_ordering
class Message:
    VALID_SENDERS = os.getenv("VALID_SENDERS").split(",")
    SENDER_ALIASES = os.getenv("SENDER_ALIASES").split(",")

    def __init__(self, sender, timestamp):
        # Check if the sender is valid
        if sender not in self.VALID_SENDERS:
            raise ValueError(f"Invalid sender: {sender}. Must be one of {', '.join(self.VALID_SENDERS)}.")

        # Find the index of the sender in VALID_SENDERS
        sender_idx = self.VALID_SENDERS.index(sender)

        # Use the index to find the corresponding alias
        self.sender = self.SENDER_ALIASES[sender_idx]
        self.sender_idx = sender_idx # Store the index for creating one-hot vectors
        self.timestamp = timestamp
        self.message_type = 'unknown'
        self.content = None

    @classmethod
    def from_json(cls, msg_data):
        # Lazily import the required message types
        from text_message import TextMessage
        from multimedia_message import MultimediaMessage

        # Factory method to determine message type and create an instance
        if 'content' in msg_data:
            return TextMessage(msg_data['sender_name'], msg_data['timestamp_ms'], msg_data['content'])
        elif 'photos' in msg_data:
            return MultimediaMessage(msg_data['sender_name'], msg_data['timestamp_ms'], msg_data['photos'], 'photo')
        elif 'gifs' in msg_data:
            return MultimediaMessage(msg_data['sender_name'], msg_data['timestamp_ms'], msg_data['gifs'], 'gif')
        elif 'sticker' in msg_data:
            return MultimediaMessage(msg_data['sender_name'], msg_data['timestamp_ms'], msg_data['sticker'], 'sticker')
        elif 'videos' in msg_data:
            return MultimediaMessage(msg_data['sender_name'], msg_data['timestamp_ms'], msg_data['videos'], 'video')
        elif 'audio_files' in msg_data:
            return MultimediaMessage(msg_data['sender_name'], msg_data['timestamp_ms'], msg_data['audio_files'], 'audio')
        elif 'files' in msg_data:
            return MultimediaMessage(msg_data['sender_name'], msg_data['timestamp_ms'], msg_data['files'], 'files')
        elif 'timestamp_ms' in msg_data:
            return Message(msg_data['sender_name'], msg_data['timestamp_ms'])
        else:
            raise ValueError("Unknown message type")

    def __str__(self):
        est = timezone(timedelta(hours=-5))
        # Convert from milliseconds to seconds and then to a datetime object
        dt = datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc).astimezone(est)
        formatted_time = dt.strftime('%B %d, %Y %I:%M.%S %p %Z')
        return (
            f"Sender: {self.sender}\n"
            f"Timestamp: {formatted_time}\n"
        )

    def __repr__(self):
        return f"Message(sender={self.sender}, timestamp={self.timestamp})"
    
    def __eq__(self, other):
        if not isinstance(other, Message):
            return NotImplemented
        return self.timestamp == other.timestamp

    def __lt__(self, other):
        if not isinstance(other, Message):
            return NotImplemented
        return self.timestamp < other.timestamp
