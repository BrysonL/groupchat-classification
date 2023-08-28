import dotenv
import os
import torch
from data_load import load_messages_from_directory, split_data, extract_features_and_labels, clean_and_filter_messages
from most_frequent_classifier import MostFrequentClassClassifier
from linear_model import MultiClassLinearModel
from classifier_evaluator import ClassifierEvaluator
import torch.nn.functional as F

RAND_SEED = 42

# 1. Load the message data
dotenv.load_dotenv()
messages_folder = os.getenv("MESSAGES_FOLDER_PATH")
print(f"Loading messages from: {messages_folder}")
_, text_messages = load_messages_from_directory(messages_folder)
text_messages = clean_and_filter_messages(text_messages, min_words=1)

# 2. Split the data into training and test sets
train_data, _, test_data = split_data(text_messages, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, seed=RAND_SEED)
train_features, train_labels = extract_features_and_labels(train_data)
test_features, test_labels = extract_features_and_labels(test_data)

num_classes = train_labels.size(1)
num_features = train_features.size(1)

# # 3a. Train the MostFrequentClassClassifier
# model = MostFrequentClassClassifier(num_classes)
# model.train_model(train_data, train_labels)

# 3b. Train the MultiClassLinearModel
model = MultiClassLinearModel(num_classes=num_classes)
model.train_model(train_features, train_labels)

# 4. Evaluate the trained model on the test data
predictions = model.predict(test_features)

evaluator = ClassifierEvaluator(num_classes)
accuracy, confusion_matrix = evaluator.evaluate(test_labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix}")

