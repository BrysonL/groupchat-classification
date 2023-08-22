from datetime import datetime, timezone, timedelta
from message import Message

class MultimediaMessage(Message):
    def __init__(self, sender_name, timestamp_ms, content, message_type):
        super().__init__(sender_name, timestamp_ms)
        self.content = content
        self.message_type = message_type

    def __str__(self):
        est = timezone(timedelta(hours=-5))
        # Convert from milliseconds to seconds and then to a datetime object
        dt = datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc).astimezone(est)
        formatted_time = dt.strftime('%B %d, %Y %I:%M.%S %p %Z')
        return (
            f"Sender: {self.sender}\n"
            f"Timestamp: {formatted_time}\n"
            f"Content path: {self.content}"
        )
    