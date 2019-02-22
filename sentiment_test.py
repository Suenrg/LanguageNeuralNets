#Ben Shakow
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import random
from keras.models import load_model
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle



model = load_model('C:/Code/seq2seq/Keras/modelSave2.h5')

with open('tokenizer.pickle', 'rb') as handle:
    token = pickle.load(handle)


def testPhrase(phrase):
    doc = keras.preprocessing.text.text_to_word_sequence(phrase, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    for l in range(len(doc)):
        temp = doc[l]
        doc[l] = token.word_index.get(temp)
        if (doc[l] == None):
            doc[l]=0
    while len(doc)<35:
        doc.append(0)
    data = []
    data.append(doc)
    data = np.array(data)
#    print(data)
    tested = model.predict(data)
    print('The phrase "'+ phrase + '" is ', end='')
    if(tested[0]<=.5):
        print('bad :(')
    else:
        print('good!')
    print('\n')

print('\n\n')
testPhrase('dobby takes it rough but gives it tender')
