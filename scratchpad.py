from data_load import *
from classifier_evaluator import ClassifierEvaluator
from most_frequent_classifier import MostFrequentClassClassifier
import dotenv
import os
import torch

dotenv.load_dotenv()

messages_folder = os.getenv("MESSAGES_FOLDER_PATH")
print(messages_folder)

messages, text_messages = load_messages_from_directory(messages_folder)
print(len(messages), len(text_messages))
print(messages[0])

cleaned_text_messages = clean_and_filter_messages(text_messages, min_words=1)
print(len(cleaned_text_messages))

features_tensor, sender_vector = extract_features_and_labels(cleaned_text_messages)

print(features_tensor.shape)
print(features_tensor[0])
# ce = ClassifierEvaluator(3)

# y_test = torch.zeros(100, 3)
# indices = torch.randint(0, 3, (100,))
# y_test[torch.arange(100), indices] = 1

# yhat_test = torch.rand(100, 3)
 
# # print(y_test)
# # print(yhat_test)

# print(ce.evaluate(y_test, yhat_test))

# num_classes = 3
# mf_classifier = MostFrequentClassClassifier(num_classes)
# train_labels = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
# mf_classifier.train_model(None, train_labels)
# yhat = mf_classifier.predict(["sample1", "sample2", "sample3"])
# y = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])

# print(ce.evaluate(y, yhat))
