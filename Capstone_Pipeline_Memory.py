#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Set Up Procedures
#1.Ensure All imports below have been installed
#2.Download the model: cased_L-12_H-768_A-12 found at: https://bert-as-service.readthedocs.io/en/latest/section/get-start.html
#3.Set up BaaS in background by running: bert-serving-start -model_dir <PathToModel>
#4.load all functions below main, then use main to run the pipeline


# In[1]:


#Imports 
import twint
import json
from bert_serving.client import BertClient
import nest_asyncio
import numpy as np
import re
import json
import sys
import pdb
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
import nltk
import spacy
from spacy.lang.en import English


# In[26]:


#Collecting all the tweets that will be passed through the pipeline
nlp = spacy.load('en_core_web_sm')
nest_asyncio.apply()
tags = getTags()
tags = processTags(tags)
scrapeTweets(tags)
data = processData()
data = cleanData(data)


# In[25]:


#Passing Every Tweet through pipeline
for tweet in data[:50]:
    #Get tweet encoding
    encoding = getEncodings(tweet['tweet'])
    #Classify the tweet as either relevant or irrelevant
    tweet = classifyTweets("svm_model.p",tweet,encoding,"classifiedTweets.txt")
    #Only Continue if tweet is relevant
    if(not removeIrrelevant(tweet)):
        continue
    tweet = applyNLTK(tweet)
    tweet = applySpacy(tweet,2)
    #Add Graph DB functionality


# In[2]:


#Get all security tags that we will be searching twitter with
def getTags():
    #Reads tags from file  and adds each tag to an array
    file = open("./security_tags.txt","r")
    arr = []
    line1 = file.readlines()
    for x in line1:
        arr.append(x)
    file.close()
    return arr


# In[3]:


#Process the tags so they are ready to be used with Twint
def processTags(arr):
    #Removes the hashtag from every tag
    for i in range(len(arr)):
        arr[i] = arr[i][1:]
    return arr


# In[4]:


#Use twint to collect tweets
def scrapeTweets(arr):
    
    #Iterates through all security terms and searches twitter, writes all the data to a file called test1.json
    for i in range(len(arr)):
        d = twint.Config()
        d.Search = arr[i]
        d.Limit = 1
        d.Store_object = True
        d.Store_json = True
        d.Output = "./test1.json"
        twint.run.Search(d)
    return


# In[5]:


#Process data collected from twint so it is ready to be encoded by bert
def processData():
    f= open("./test1.json","r",errors='ignore')
    line1 = f.readlines()
    arr = []
    for x in line1:
        arr.append(x)
    f.close() 
    arr.insert(0,"[")
    arr.append(']')
    for i in range(1,len(arr)-2):
        arr[i] = arr[i][:-1] + "," + arr[i][-1:]
    file = open("../test.json","w")
    for i in range(len(arr)):
        file.write(arr[i])
    file.close()
    f= open("../test.json","r",errors='ignore')
    line1 = f.readlines()
    data = []
    for x in line1:
        data.append(x)
    f.close()
    for i in range(len(data)):
        data[i] = (''.join([i if ord(i) < 128 else ' ' for i in str(data[i])]))
    file = open("./data.json","w")
    for i in range(len(data)):
        file.write(data[i])
    file.close()
    with open("./data.json", "r",errors='ignore') as read_file:
        data = json.load(read_file)
    return data


# In[6]:


def getEncodings(data):
    #Remove lenght requirements as some of the tweets are longer and it will throw an error
    #Gets encodings from BaaS and creates a npy file with encodings called encoded_file.npy
    tweet = [data]
    bc = BertClient(check_length = False)
    encoding = bc.encode(tweet)
    return encoding


# In[7]:


#Prepare Data for encoding
def cleanData(data):
    #parse json data into two arrays containing the tweet texts and their correponding relevancy label
    #track numbers of relevant and irrelevant tweets 
    relnum =0
    irrnum = 0
    text = []
    labels = []
    newData = []
    seen_urls = set()
    duplicate_counter = 0
    
    for x in data:
#          Dont need to worry about duplicates for streaming
#         if re.sub(r"http\S+", "", x['tweet']) in seen_urls:
#             duplicate_counter +=1
#             continue
#         else:
#             seen_urls.add(re.sub(r"http\S+", "", x['tweet']))

        text.append(re.sub(r"http\S+", "", x['tweet']))
    #Add regex for removing  urls

    for i in data:
        if(re.match('[a-zA-Z]',i['tweet'])):
            newData.append(i)
            continue
        else:
            continue
    newtext = [str for str in text if re.match('[a-zA-Z]', str)]
    data =newData
#     getEncodings(newtext)
    return data


# In[8]:


def classifyTweets(model_file,tweet,encoding,output_file):
    #Uses model stored in svm_model.p to get predictions and we add these predictions to our data array
    clf = pickle.load(open(model_file,'rb'))
    predictions = clf.predict(encoding)
    print_arr = ['Irrelevant','Relevant']
    tweet['Relevance'] = str(print_arr[predictions[0]])
    return tweet


# In[9]:


def removeIrrelevant(tweet):
    #Cleans up our data by only keeping relevant tweets
    if tweet['Relevance'] == "Relevant":
        return True
    else:
        return False


# In[10]:


#Outdated
# def applyNLTK(tweet):
#     #Runs all the tweets through nltk and stores all the nouns for each tweet in the data object

#     nouns = []
#     tokens = nltk.word_tokenize(tweet['tweet'])
#     tagged = nltk.pos_tag(tokens)
#     for i in range(len(tagged)):
#         if (tagged[i][1] == 'NNP'):
#             nouns.append(tagged[i][0])
#     tweet['nltk'] = nouns
#     return tweet


# In[11]:


def applyNLTK(tweet):
    nlp = spacy.load('en_core_web_sm')
    sen_noun = []
    sentence = tweet['tweet']
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    for j in range(len(tagged)):
        if (tagged[j][1] == 'NNP'):
            sen_noun.append(tagged[j][0])
    tweet['nltk'] = sen_noun
    return tweet


# In[12]:


#Outdated
# def applySpacy(tweet):
#     #Runs all the tweets through Spacy and stores the words extracted in the data object
#     doc = nlp(tweet['tweet'])
#     tweet['spacy'] = [ent for ent in doc.ents]
#     return tweet


# In[13]:



# Option = 1 => Get all Nouns, Oprion = 2 => Get position entities
def applySpacy( tweet, option ):
    nlp = spacy.load('en_core_web_sm')
    sentence = tweet['tweet'].replace("-", " ").replace("#", " ")
    doc = nlp(sentence)
    if option == 1:
        tagged = [chunk.text for chunk in doc.noun_chunks]
        tweet['spacy'] = tagged
        return tweet
    else:
        pos_ent = []
        for ent in doc.ents:
            pos_ent.append(ent.text)
        tweet['spacy'] = pos_ent
        return tweet


# In[14]:


#Alternative Spacy --Currently Not in use
########################################################################################################
# Uses Spacy for finding nouns, named entity recognition
# NEW results from spacy, after removing stop words
########################################################################################################

def get_nouns_wo_stop_words (data, option):
    nlp = English()

    my_doc = nlp(data['text'].replace("-", " ").replace("#", " "))
    # Create list of word tokens
    token_list = []
    for token in my_doc:
        token_list.append(token.text)
    from spacy.lang.en.stop_words import STOP_WORDS

    # Create list of word tokens after removing stopwords
    filtered_sentence =[] 

    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)    # Contains the sentence without stop words
    # print(token_list)
    # print(filtered_sentence) 

    new_text = filtered_sentence[0] + " "
    for i in range(1,len(filtered_sentence)):
        new_text += filtered_sentence[i] + " "

    nlp = spacy.load('en_core_web_sm')

    #  "nlp" Object is used to create documents with linguistic annotations.
    noun = []
    sen_noun = []
    sentence = new_text
    doc = nlp(sentence)
    if option == 1:
        tagged = [chunk.text for chunk in doc.noun_chunks]
        return tagged
    else:
        pos_ent = []
        for ent in doc.ents:
            pos_ent.append(ent.text)
#             print(ent.text, ent.label_)
        return pos_ent


# In[ ]:




