# Review_Classifier
Builds a neural network using global average pooling and various activation / loss functions to classify a movie review as having either positive or negative sentiment
Written in Python

review_classifier_train.py
- Trains a model to classify a movie review as either negative (0) or positive (1)
- Model builds a neural network by taking 25,000 reviews from IMDB labeled with a positive/negative sentiment and linking them to 88,000 words in the English dictionary

review_classifier_test.py
- Uses the model trained from 'review_classifier_train.py' to classify a review it has never seen before in file 'JokerReview.txt'

System Requirements: TensorFlow, Numpy
