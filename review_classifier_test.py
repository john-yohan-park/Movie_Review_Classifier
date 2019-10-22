'''
John Park
Github: john-yohan-park

- Uses the model trained from 'review_classifier_train.py' to classify a movie review
  it has never seen before in file 'JokerReview.txt'

System Requirements: TensorFlow, Numpy
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words= 880000)

word_index = data.get_word_index()      # word_index -> word

word_index = {k:(v+3) for k, v in word_index.items()}       # key, value
word_index['<PAD>']    = 0   # added to pad review length
word_index['<START>']  = 1
word_index['<UNK>']    = 2   # unknown
word_index['<UNUSED>'] = 3

def review_encode(s):
    encoded = [1]       # starting tag
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)   # unknown tag
    return encoded

model = keras.models.load_model('model.h5')

with open('Joker_Review.txt', encoding = 'UTF-8') as inputFile:
    for line in inputFile.readlines():
        newLine = line.replace(',', '').replace('.', '').replace(';', '').replace(':', '').replace('\"', '').replace('\'', '').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('!', '').strip().split(' ')
        encode = review_encode(newLine)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value = word_index["<PAD>"], padding = "post", maxlen = 250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
