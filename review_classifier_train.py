'''
John Park
Github: john-yohan-park

- Trains a model to classify a review as either negative (0) or positive (1)
- Model builds a neural network by taking 25,000 reviews from IMDB labeled with a
  positive/negative sentiment and linking them to 88,000 words in the English dictionary

System Requirements: TensorFlow, Numpy
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb  # preprocessed data with each word in the review encoded as a word index

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words= 880000)

word_index = data.get_word_index()      # convert word_index to word

word_index = {k:(v+3) for k, v in word_index.items()}       # key, value
word_index['<PAD>']    = 0   # added to pad review length
word_index['<START>']  = 1
word_index['<UNK>']    = 2   # unknown
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)
test_data  = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])   # if we can't find value for i, return '?'
#print(decode_review(test_data[0]))

#===========================================START TRAINING===========================================

model = keras.Sequential()
model.add(keras.layers.Embedding(880000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = 'relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))    # between 0 and 1
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# check model performance
x_val = train_data[:10000]    # take 1st 10k
y_val = train_labels[:10000]

x_train = train_data[10000:]  # remaining 15k
y_train = train_labels[10000:]

fitModel = model.fit(   x_train, y_train, epochs = 40,
                        batch_size = 512,                   # how many reviews to load each time
                        validation_data = (x_val, y_val),
                        verbose = 1
                     )
results = model.evaluate(test_data, test_labels)
#print(results)

#===========================================END TRAINING===========================================

model.save('model.h5')

test_review = test_data[0]                  # print outcome
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)
