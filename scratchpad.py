from data_load import load_messages_from_directory, clean_and_filter_messages
import dotenv
import os

dotenv.load_dotenv()

messages_folder = os.getenv("MESSAGES_FOLDER_PATH")
print(messages_folder)

messages, text_messages = load_messages_from_directory(messages_folder)
print(len(messages), len(text_messages))
print(messages[0])

cleaned_text_messages = clean_and_filter_messages(text_messages)
print(len(cleaned_text_messages))

msg = cleaned_text_messages[4]
sender_vector = msg.get_sender_vector()
message_features = msg.extract_features()

print(sender_vector)
print(message_features)
