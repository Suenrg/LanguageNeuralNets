#Ben Shakow, with a ton of help from https://www.tensorflow.org/tutorials/keras/basic_text_classification
#This is just the sentiment model, but improved
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

print(tf.__version__)

########################################################

def load_text(filename):
    File = open(filename,'r')
    text = File.read()
    File.close()
    docs = text.split('\n')
    return docs

goodList = load_text('good.txt')
badList = load_text('bad.txt')


totList = goodList+badList
token = Tokenizer(filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
token.fit_on_texts(totList)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

testCut = round(.7 * len(data))
valCut = round(.2 * len(data))

train_data = data[:testCut]
test_data = data[testCut:]

train_labels = labels[:testCut]
test_labels = labels[testCut:]

x_val = train_data[:valCut]
partial_x_train = train_data[valCut:]

y_val = train_labels[:valCut]
partial_y_train = train_labels[valCut:]

print(len(data))
print(len(test_data))
print(len(y_val))
print(len(partial_y_train))


#####################################################################################################################################################################

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 1000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(partial_x_train, partial_y_train, epochs=600, batch_size=125, validation_data=(x_val, y_val), verbose=0)

results = model.evaluate(test_data, test_labels)

model.save('.modularSave.h5')


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
