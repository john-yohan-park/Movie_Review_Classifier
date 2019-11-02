# Review_Classifier
Classifies a movie review as either positive or negative by building a neural network using:
- Global average pooling
- Rectified linear units activation function
- Sigmoid activation function
- Adam optimizer algorithm
- Binary cross-entropy loss function

Written in Python

review_classifier_train.py
- builds the model's neural network by taking 25,000 reviews from IMDB labeled with a positive/negative sentiment and using them to score 88,000 words in the English dictionary

review_classifier_test.py
- uses the model trained from 'review_classifier_train.py' to classify a review it has never seen before in 'JokerReview.txt'

System Requirements: TensorFlow, Numpy
