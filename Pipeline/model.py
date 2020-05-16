from bert_serving.client import BertClient
import numpy as np
import re
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle


def getEncodings(data):
    #Remove lenght requirements as some of the tweets are longer and it will throw an error
    #Gets encodings from BaaS and creates a npy file with encodings called encoded_file.npy
    tweet = [data]
    bc = BertClient(check_length = False)
    encoding = bc.encode(tweet)
    return encoding


def classifyTweets(model_file,tweet,encoding,output_file):
    #Uses model stored in svm_model.p to get predictions and we add these predictions to our data array
    clf = pickle.load(open(model_file,'rb'))
    predictions = clf.predict(encoding)
    print_arr = ['Irrelevant','Relevant']
    tweet['Relevance'] = str(print_arr[predictions[0]])
    return tweet
