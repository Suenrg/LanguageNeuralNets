#Ben Shakow, with a ton of help from https://www.tensorflow.org/tutorials/keras/basic_text_classification
#sentiment analysis from files, worse version than sentiment_clean
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

import numpy as np

print(tf.__version__)

########################################################
goodFile = open('C:/Code/seq2seq/Keras/good.txt','r')
good = goodFile.read()
goodFile.close()
goodList = good.split('\n')

badFile = open('C:/Code/seq2seq/Keras/bad.txt','r')
bad = badFile.read()
badFile.close()
badList = bad.split('\n')


totList = goodList+badList
token = Tokenizer(filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
token.fit_on_texts(totList)
#print(token.word_counts)
#print(token.document_count)
#print(token.word_index)
#print(token.word_docs)
#print(goodList)
for k in range(len(goodList)):
    doc = keras.preprocessing.text.text_to_word_sequence(goodList[k], filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    for l in range(len(doc)):
        temp = doc[l]
        doc[l] = token.word_index.get(temp)
    while len(doc)<35:
        doc.append(0)
    goodList[k] = doc

for k in range(len(badList)):
    doc = keras.preprocessing.text.text_to_word_sequence(badList[k], filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    for l in range(len(doc)):
        temp = doc[l]
        doc[l] = token.word_index.get(temp)
    while len(doc)<35:
        doc.append(0)
    badList[k] = doc

#make the dictionary 1 = good, 0 = bad
data = goodList+badList
labels = []

for i in range(len(goodList)):
    labels.append(1)
for i in range(len(badList)):
    labels.append(0)

dataset = list(zip(data, labels))
random.shuffle(dataset)
data[:], labels[:] = zip(*dataset)

data = np.array(data)
labels = np.array(labels)

train_data = data[:450]
test_data = data[450:]

train_labels = labels[:450]
test_labels = labels[450:]

x_val = train_data[:100]
partial_x_train = train_data[100:]

y_val = train_labels[:100]
partial_y_train = train_labels[100:]

print()
print(len(y_val))
print(len(partial_y_train))

#######################################################

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 1000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(partial_x_train, partial_y_train, epochs=400, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)


print(results)

history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
#'''
