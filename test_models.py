import dotenv
import os
import torch
from data_load import load_messages_from_directory, split_data
from most_frequent_classifier import MostFrequentClassClassifier
from classifier_evaluator import ClassifierEvaluator
import torch.nn.functional as F

RAND_SEED = 42

# 1. Load the message data
dotenv.load_dotenv()
messages_folder = os.getenv("MESSAGES_FOLDER_PATH")
print(f"Loading messages from: {messages_folder}")
_, text_messages = load_messages_from_directory(messages_folder)

# 2. Split the data into training and test sets
train_data, _, test_data = split_data(text_messages, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, seed=RAND_SEED)
train_labels = F.one_hot(torch.tensor([msg.sender_idx for msg in train_data]))
test_labels = F.one_hot(torch.tensor([msg.sender_idx for msg in test_data]))

# 3. Train the MostFrequentClassClassifier
num_classes = train_labels.size(1)
model = MostFrequentClassClassifier(num_classes)
model.train_model(train_data, train_labels)

# 4. Evaluate the trained model on the test data
predictions = model.predict(test_data)

evaluator = ClassifierEvaluator(num_classes)
accuracy, confusion_matrix = evaluator.evaluate(test_labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix}")

