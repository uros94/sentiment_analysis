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

def read_data(filename):
    f=open(filename, "r")
    if f.mode == 'r':
        f1 = f.readlines()
        for contents in f1:
            training_data.append({"class":contents[-2], "sentence":contents[:-2]})
            
# loading data from files
read_data("C:\\Users\\Win 10\\Desktop\\New folder\\imdb_labelled.txt")
read_data("C:\\Users\\Win 10\\Desktop\\New folder\\amazon_cells_labelled.txt")
read_data("C:\\Users\\Win 10\\Desktop\\New folder\\yelp_labelled.txt")

print ("%s sentences in training data" % len(training_data))

words = []
classes = []
documents = []
ignore_words = ['?', '.', 'the', 'a']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


# create our training data# creat 
training = [] # every row has len(words) elements: '1' if the sentence equvalent to that row cntains the word equvivalent to that column, otherweis '0'
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

print ("# words", len(words))
print ("# classes", len(classes))

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# ANN and Gradient Descent code from https://iamtrask.github.io//2015/07/27/python-network-part2/
def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    # weights go from -1 to 1
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1     # [num-of-words, num-of-neurons]
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1  # [num-of-neurons, num-of-classes]
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        # np.dot - the dot product of two arrays
        layer_0 = X
        #print("this is layer0: %sx%s" % (len(layer_0), len(layer_0[0])))
        #print("this is synapse0: %sx%s" % (len(synapse_0), len(synapse_0[0])))
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        #print("this is layer1: %sx%s" % (len(layer_1), len(layer_1[0])))
        #print("this is synapse1: %sx%s" % (len(synapse_1), len(synapse_1[0])))

        #print(layer_0[0])
        #print(synapse_0[0])
        #print(layer_1[0])
        #print(synapse_1[0])
        
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
        
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        #print("this is layer2: %sx%s" % (len(layer_2), len(layer_2[0])))
     
        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% (epochs / 20) == 0):
            print(str(int(j / epochs *100))+"%...", end="")
                
        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            # np.mean - compute the arithmetic mean along the specified axis
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
            
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "test-synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)

X = np.array(training)
y = np.array(output)

start_time = time.time()
# calling the training function
######train(X, y, hidden_neurons=20, alpha=0.1, epochs=50000, dropout=False, dropout_percent=0.2)
elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
