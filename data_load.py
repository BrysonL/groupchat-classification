import dotenv
import os
import json
import re
from message import Message

dotenv.load_dotenv()

messages_folder = os.getenv("MESSAGES_FOLDER_PATH")
print(messages_folder)

def load_messages_from_directory(directory):
    all_messages = []
    text_messages = []

    # List all files in the directory
    filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".json")]

    for filename in filenames:
        with open(os.path.join(directory, filename), "r") as file:
            print(f"Loading {filename}")
            data = json.load(file)
            for msg_data in data["messages"]:
                try:
                    message = Message.from_json(msg_data)
                    all_messages.append(message)
                    if message.message_type == "text":
                        text_messages.append(message)
                except ValueError as e:
                    print(f"Error loading message: {msg_data}")

    return sorted(all_messages), sorted(text_messages)

def clean_and_filter_messages(messages, min_words=3):
    # Filters to remove common system messages
    filter_photo = re.compile("changed the group's photo", re.I)
    filter_reacted = re.compile("reacted .* to your message", re.I)
    
    cleaned_messages = []

    for message in messages:
        if not (filter_photo.search(message.content) or filter_reacted.search(message.content)):
            if len(message.content.split()) > min_words:  # Check if message has more than min_words words
                cleaned_messages.append(message)

    return cleaned_messages

messages, text_messages = load_messages_from_directory(messages_folder)
print(len(messages), len(text_messages))
print(messages[0])

cleaned_text_messages = clean_and_filter_messages(text_messages)
print(len(cleaned_text_messages))
