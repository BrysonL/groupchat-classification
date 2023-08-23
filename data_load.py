import os
import json
import re
from message import Message
import random
import torch

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
            if len(message.content.split()) >= min_words:  # Check if message has more than min_words words
                cleaned_messages.append(message)

    return cleaned_messages


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    # Assert that the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    # Seed the random number generator for reproducibility
    if seed is not None:
        random.seed(seed)

    # Shuffle the data
    random.shuffle(data)

    # Calculate split indices
    train_idx = int(len(data) * train_ratio)
    val_idx = train_idx + int(len(data) * val_ratio)

    # Split the data
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]

    return train_data, val_data, test_data

def extract_features_and_labels(messages):
    features_list = [msg.extract_features() for msg in messages]
    features_tensor = torch.stack(features_list)

    sender_vectors = [msg.get_sender_vector() for msg in messages]
    sender_vectors_tensor = torch.stack(sender_vectors)

    return features_tensor, sender_vectors_tensor