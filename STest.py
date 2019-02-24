#Ben Shakow
#A class based version of the sentiment test
from __future__ import absolute_import, division, print_function
from keras.models import load_model
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle


class STest:
    def __init__(self):
        self.model = load_model('.modularSave.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            self.token = pickle.load(handle)

    def testPhrase(self, phrase):
        doc = keras.preprocessing.text.text_to_word_sequence(phrase, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
        for l in range(len(doc)):
            temp = doc[l]
            doc[l] = self.token.word_index.get(temp)
            if (doc[l] == None):
                doc[l] = 0
        while len(doc)<35:
            doc.append(0)
        data = []
        data.append(doc)
        data = np.array(data)
    #    print(data)
        tested = self.model.predict(data)
        return tested
