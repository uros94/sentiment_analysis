# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time
stemmer = LancasterStemmer()

training_data = []
validation_data = []

def read_data(filename):
    f=open(filename, "r")
    if f.mode == 'r':
        f1 = f.readlines()
        cnt = 0
        limit = round(len(f1)* 0.9) #90% of data forms training set and 10% of data  forms validation set
        for contents in f1:
            if cnt < limit:
                cnt += 1
            else:
                validation_data.append({"class":contents[-2], "sentence":contents[:-2]})
            
# loading data from files
read_data("C:\\Users\\Win 10\\Desktop\\New folder\\imdb_labelled.txt")
read_data("C:\\Users\\Win 10\\Desktop\\New folder\\amazon_cells_labelled.txt")
read_data("C:\\Users\\Win 10\\Desktop\\New folder\\yelp_labelled.txt")
print ("%s sentences in training data" % len(training_data))

### validation!!! ###
# probability threshold
ERROR_THRESHOLD = 0.4
synapse_file = "test-synapses.json"
with open(synapse_file) as data_file: 
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])
    words = np.asarray(synapse['words'])
    classes = np.asarray(synapse['classes'])
    
def classify(sentence, show_details=False):
    results = think(sentence, show_details)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results
  
def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2
  
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# doing validation
cnt = 0
for sentence in validation_data:
    if classify(sentence['sentence'])[0][0] == sentence['class']:
        cnt += 1
print ("validation: percent of correct predictions %s" % (cnt / len(validation_data)))

