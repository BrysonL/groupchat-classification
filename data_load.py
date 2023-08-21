import dotenv
import os
import json
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

    return all_messages, text_messages

messages, text_messages = load_messages_from_directory(messages_folder)
print(len(messages), len(text_messages))
print(messages[0])