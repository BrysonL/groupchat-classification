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
_, _ = extract_features_and_labels(text_messages) # extract features and labels here to preserve order of messages. Once shuffled, previous message features will be incorrect

# 2. Split the data into training and test sets
train_data, _, test_data = split_data(text_messages, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, seed=RAND_SEED)
train_features, train_labels = extract_features_and_labels(train_data) # since we already compute features above, this will pull the stored values instead of recomputing (with out of order messages)
test_features, test_labels = extract_features_and_labels(test_data)

num_classes = train_labels.size(1)
num_features = train_features.size(1)

# # # 3a. Train the MostFrequentClassClassifier
# # model = MostFrequentClassClassifier(num_classes)
# # model.train_model(train_data, train_labels)

# 3b. Train the MultiClassLinearModel
# all features
model = MultiClassLinearModel(num_classes=num_classes)
model.train_model(train_features, train_labels)
train_predictions = model.predict(train_features)

# two most important features
# 8: ratio of lowercase letters
# 13-17: previous message sender
model2 = MultiClassLinearModel(num_classes=num_classes)
model2.train_model(train_features[:, [8, 13, 14, 15, 16, 17]], train_labels)
train_predictions2 = model2.predict(train_features[:, [8, 13, 14, 15, 16, 17]])

# all features with > 0.02 eta-squared or > 0.05 cramers-v
# 1: number of words
# 3: presence of URLs
# 4: presence of numbers
# 6: presence of question marks
# 7: number of lowercase letters
# 8: ratio of lowercase letters
# 13-17: previous message sender
# 25-36: month of year
model3 = MultiClassLinearModel(num_classes=num_classes)
model3.train_model(train_features[:, [1, 3, 4, 6, 7, 8, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]], train_labels)
train_predictions3 = model3.predict(train_features[:, [1, 3, 4, 6, 7, 8, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]])

# two most important features + interactions
model4 = MultiClassLinearModel(num_classes=num_classes, include_interaction=True)
model4.train_model(train_features[:, [8, 13, 14, 15, 16, 17]], train_labels)
train_predictions4 = model4.predict(train_features[:, [8, 13, 14, 15, 16, 17]])

# all features with > 0.02 eta-squared or > 0.05 cramers-v + their interactions
model5 = MultiClassLinearModel(num_classes=num_classes, include_interaction=True)
model5.train_model(train_features[:, [1, 3, 4, 6, 7, 8, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]], train_labels)
train_predictions5 = model5.predict(train_features[:, [1, 3, 4, 6, 7, 8, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]])

# 4. Evaluate the trained model on the test data
predictions = model.predict(test_features)
predictions2 = model2.predict(test_features[:, [8, 13, 14, 15, 16, 17]])
predictions3 = model3.predict(test_features[:, [1, 3, 4, 6, 7, 8, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]])
predictions4 = model4.predict(test_features[:, [8, 13, 14, 15, 16, 17]])
predictions5 = model5.predict(test_features[:, [1, 3, 4, 6, 7, 8, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]])

evaluator = ClassifierEvaluator(num_classes)
accuracy, confusion_matrix = evaluator.evaluate(test_labels, predictions)
train_accuracy, _ = evaluator.evaluate(train_labels, train_predictions)
print("Model 1: All features")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix}")

accuracy2, confusion_matrix2 = evaluator.evaluate(test_labels, predictions2)
train_accuracy2, _ = evaluator.evaluate(train_labels, train_predictions2)
print("Model 2: Two most important features")
print(f"Train Accuracy: {train_accuracy2:.4f}")
print(f"Accuracy: {accuracy2:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix2}")

accuracy3, confusion_matrix3 = evaluator.evaluate(test_labels, predictions3)
train_accuracy3, _ = evaluator.evaluate(train_labels, train_predictions3)
print("Model 3: All features with > 0.02 eta-squared or > 0.05 cramers-v")
print(f"Train Accuracy: {train_accuracy3:.4f}")
print(f"Accuracy: {accuracy3:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix3}")

accuracy4, confusion_matrix4 = evaluator.evaluate(test_labels, predictions4)
train_accuracy4, _ = evaluator.evaluate(train_labels, train_predictions4)
print("Model 4: Two most important features + interactions")
print(f"Train Accuracy: {train_accuracy4:.4f}")
print(f"Accuracy: {accuracy4:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix4}")

accuracy5, confusion_matrix5 = evaluator.evaluate(test_labels, predictions5)
train_accuracy5, _ = evaluator.evaluate(train_labels, train_predictions5)
print("Model 5: All features with > 0.02 eta-squared or > 0.05 cramers-v + their interactions")
print(f"Train Accuracy: {train_accuracy5:.4f}")
print(f"Accuracy: {accuracy5:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix5}")