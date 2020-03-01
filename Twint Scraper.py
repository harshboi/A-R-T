
# coding: utf-8

# In[ ]:


import twint
file = open("../security_tags.txt","r")
arr = []
line1 = file.readlines()
for x in line1:
    arr.append(x)
f.close()    


# In[ ]:


for i in range(len(arr)):
    arr[i] = arr[i][1:]


# In[79]:


file = open("./test1.json","w")
file.close()


# In[80]:


# clear file
#Twint Search
for i in range(len(arr)):
    d = twint.Config()
    d.Search = arr[i]
    d.Limit = 50
    # d.Store_csv = True
    d.Store_object = True
    d.Store_json = True
    d.Output = "./test1.json"
    # d.Database = "tweets.db"


    x = twint.run.Search(d)


# In[14]:


# Search terms
# 0-day AND exploit
# “0-day”, “CVE-“, “CVE-2018-*”, “CVE-2019-*”
# usernames:
# kibbsy
# it_securitynews
# msftsecurity


# In[15]:


# x = twint.run.Search(d)
# twint -u noneprivacy --csv --output "/Users/psingh4/harsh/test4.json" --lang en --translate --translate-dest it --limit 100


# In[16]:


# d.__dict__.keys()


# In[ ]:


# twint.run.Search(d)
# twint.output


# In[81]:


f= open("./test1.json","r",errors='ignore')
line1 = f.readlines()
arr = []
for x in line1:
    arr.append(x)
f.close()    


# In[82]:


arr.insert(0,"[")
arr.append(']')
for i in range(1,len(arr)-2):
    arr[i] = arr[i][:-1] + "," + arr[i][-1:]


# In[83]:


file = open("../test.json","w")
for i in range(len(arr)):
    file.write(arr[i])
file.close()


# In[84]:


f= open("../test.json","r",errors='ignore')
line1 = f.readlines()
data = []
for x in line1:
    data.append(x)
f.close()


# In[85]:


# Removing non ASCII characters
for i in range(len(data)):
    data[i] = (''.join([i if ord(i) < 128 else ' ' for i in str(data[i])]))


# In[86]:


file = open("./data.json","w")
for i in range(len(data)):
    file.write(data[i])
file.close()


# In[87]:


import json
with open("./data.json", "r",errors='ignore') as read_file:
    data = json.load(read_file)


# In[88]:


import mysql.connector


# In[90]:


mydb = mysql.connector.connect(user='admin', password='Private2712!',
                              host='database-1.cok63qqiofsd.us-east-1.rds.amazonaws.com',
                              database='data')


# In[91]:


mycursor = mydb.cursor()


# In[92]:


sql = """INSERT INTO train_data (created_at, conversation_id, id, date, time, timezone, user_id, username, name, place, tweet, mentions, urls, replies_count, 
        retweets_count, likes_count, hashtags, cashtags, link, retweet, video, near, geo, source, user_rt_id, user_rt, retweet_id,
        retweet_date, translate, trans_src, trans_dest, photos, reply_to) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

val = []
for i in range(1, len(data)):
    val.append((data[i]['created_at'], data[i]['conversation_id'], str(data[i]['id']), data[i]['date'], data[i]['time'], data[i]['timezone'], data[i]['user_id'], data[i]['username'],
      data[i]['name'], data[i]['place'], data[i]['tweet'], str(data[i]['mentions']), str(data[i]['urls']), data[i]['replies_count'],
      data[i]['retweets_count'], data[i]['likes_count'], str(data[i]['hashtags']), str(data[i]['cashtags']), data[i]['link'], data[i]['retweet'],
      data[i]['video'], data[i]['near'], data[i]['geo'], data[i]['source'], data[i]['user_rt_id'], data[i]['user_rt'],
      data[i]['retweet_id'], data[i]['retweet_date'], data[i]['translate'], data[i]['trans_src'], data[i]['trans_dest'],str(data[i]['photos']), 
       str(data[i]['reply_to'])))
    
mycursor.executemany(sql, val)

mydb.commit()

print(mycursor.rowcount, "record inserted.")


# In[93]:


mydb.close()


# In[34]:


data[0]


# In[ ]:


val = []
for i in range(len(data)):
    val.append((data[i]['created_at'], data[i]['conversation_id'], data[i]['date'], data[i]['time'], data[i]['timezone'], data[i]['user_id'], data[i]['username'],
      data[i]['name'], data[i]['place'], data[i]['tweet'], str(data[i]['mentions']), str(data[i]['urls']), data[i]['replies_count'],
      data[i]['retweets_count'], data[i]['likes_count'], str(data[i]['hashtags']), str(data[i]['cashtags']), data[i]['link'], data[i]['retweet'],
      data[i]['video'], data[i]['near'], data[i]['geo'], data[i]['source'], data[i]['user_rt_id'], data[i]['user_rt'],
      data[i]['retweet_id'], data[i]['retweet_date'], data[i]['translate'], data[i]['trans_src'], data[i]['trans_dest'],0,str(data[i]['photos']), 
       str(data[i]['reply_to'])))


# In[113]:


# mydb.commit()
# !pip install tensorflow
# print(mycursor.rowcount, "record inserted.")


# In[114]:


# !pip install bert-serving-server
# !pip install bert-serving-client


# In[2]:


import nltk


# In[42]:


sentence = "#ICYMI: #Ryuk #ransomware caused a major disruption for some high-profile print media organizations in the United States. Here's what you need to know, from our weekly threat intelligence brief:  http://spr.ly/6014Ez7dr   pic.twitter.com/ctOFYC1t0U"
tokens = nltk.word_tokenize(sentence)


# In[43]:


tagged = nltk.pos_tag(tokens)


# In[44]:


tagged


# In[42]:


# !pip install -U bert-serving-server bert-serving-client


# In[41]:


# !bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 


# In[ ]:


# from bert_serving.client import BertClient
# bc = BertClient()
# bc.encode(['First do it', 'then do it right', 'then do it better'])


# In[100]:


# first neural network with keras tutorial
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
data = np.load('./tweet_encodings.npy')
label = np.load('./maybeincludedlabels.npy')
# define the keras model
model = Sequential()
model.add(Dense(400, input_dim=768, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='softmax'))
# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
# evaluate the keras model


# In[54]:


model.fit(data, label, epochs=10, batch_size=1, validation_split=0.25)


# In[103]:


import sklearn
from sklearn.linear_model import LogisticRegression
len(data), len(label)


# In[101]:


clf = LogisticRegression(random_state=1).fit(data[:-900], label[:-900])


# In[102]:


clf.score(data[-900:], label[-900:])


# In[70]:


y_pred = clf.predict(data[-800:])


# In[74]:


y_true = label[-800:]


# In[75]:


from sklearn import metrics

accuracy_score = metrics.accuracy_score(y_true,y_pred)
print("Accuracy score: {}".format(accuracy_score))
fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred,pos_label = 1)
auc_score = metrics.auc(fpr,tpr)
print("AUC: {}".format(auc_score))


# In[81]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
import numpy as np
import sys
import pdb
import matplotlib.pyplot as plt


def print_metrics(y_true,y_pred):
    accuracy_score = metrics.accuracy_score(y_true,y_pred)
    print("Accuracy score: {}".format(accuracy_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred,pos_label = 1)
    auc_score = metrics.auc(fpr,tpr)
    print("AUC: {}".format(auc_score))
    return accuracy_score,auc_score

def logistic_regression(x_train,x_test,y_train,y_test):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train,y_train)
    predictions = logisticRegr.predict(x_test)
    print_metrics(y_test,predictions)


def support_vector(x_train,x_test,y_train,y_test):
    model = svm.SVC()
    model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    print_metrics(y_test,predictions)

def torch_label_creator(label):
    if label.item() == 1:
        return torch.tensor([1,0])
    else:
        return torch.tensor([0,1])

    #plot test accuracies and auc over epochs    
    plt.plot(test_accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Test Accuracy vs Epoch')
    plt.show()
    plt.plot(test_auc)
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.title('Test AUC vs Epoch')
    plt.show()


# In[98]:


encodings = np.load('tweet_encodings.npy')
labels = np.load('maybeincludedlabels.npy')
x_train,x_test,y_train,y_test = train_test_split(encodings,labels, test_size=0.25, random_state=0)


# In[99]:


logistic_regression(x_train,x_test,y_train,y_test)


# In[89]:


clf.score(x_test, y_test)


# In[90]:


len(data)


# In[104]:


labels


# In[106]:


len(labels == 1)

