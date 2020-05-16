<div align="center">
  <img src="../../Images/mcafee_logo.png", height="200" width="350">
</div>

--------------------------------------------------------------------------------


# Tweet Relevancy Model using BERT-As-A-Service (BAAS)

This folder contains files used for creating a BERT model that tests whether a tweet is cybersecurity threat related or not. The base architecture is that a BAAS server is spun up that hosts a pretrained BERT model which returns vector encodings for sentences that are sent to it. These encodings are then passed onto a classification layer to get the final prediction of "Relevant" or "Irrelevant". The three classification layers that were implemented and tested were support vector machine, logistic regression, and single dense neural layer.

## Installation Requirements


### 1) Tensorflow (Version 1.10)
    
  Note: Version 1.10 is required for compatibility with BAAS

```
  $ pip install tensorflow==1.10      # if you do not want GPU support
  $ pip install tensorflow-gpu==1.10  # if you want GPU support
```
  Note on NumPy version: the command above will automatically install NumPy if it is not already installed. However, if there is a preexisting NumPy installation in your environment, the version might be too high. In that case setup a new environment with NumPy version 1.17.5

#### 2) Bert As A Service (BAAS)

  See the [Bert As A Service install guide](https://github.com/hanxiao/bert-as-service/#user-content-bert-as-service)

  1. BAAS has a server and client architecture. Install them using 

  ```
  $ pip install bert-serving-server  # server
  $ pip install bert-serving-client  # client, independent of `bert-serving-server`
  ```

  Download the pretrained BERT model - [Bert Large-Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)

  2. Start the BERT service
  After installing the server, you should be able to use bert-serving-start CLI as follows:

  bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 
  This will start a service with four workers, meaning that it can handle up to four concurrent requests. More concurrent requests will be queued in a load balancer. Details can be found in our FAQ and the benchmark on number of clients.

  3. Use Client to Get Sentence Encodes
  Now you can encode sentences simply as follows:
```
  from bert_serving.client import BertClient
  bc = BertClient()
  bc.encode(['First do it', 'then do it right', 'then do it better'])
```

### 3) Scikit-Learn

```
  $ pip install -U scikit-learn
```


### 4) Matplotlib

```
  $ pip install -U matplotlib
```

## Contents

### 1) baas_encode.py

This file takes in a json array object containing tweets, passes them to the BAAS server and produces three numpy arrays containing the tweet text, tweet encodings, and tweet labels. Along the way it also removes empty and duplciate tweets. It is critical that a BAAS server already be running in order for this script to work. It assumes that the json objects for individual tweets contain a field "tweet", if the actual tweet text is stored in a different field, adapt the script accordingly.

Usage:
```
  python runbert.py <json data input file path> <text output filename> <encoding output filename> <label output filename> [ignore maybes]
```
Type in 'ignore' for [ignore maybes] to ignore maybe relevant tweets, otherwise script uses them by default

### 2) trainmodel.py

This file trains a classification layer on top of the BERT encodings and provides performance metrics on the training and optionally the testing set. It takes in the type of classifier to be trained as well as NumPy arrays for the training encodings and labels. Optionally, NumPy arrays for testing encodings and labels can be provided as well

Usage:
```
  $ python trainmodel.py <classifier_type> <training_tweets> <training_labels> [test_tweets] [test_labels]
```
Options for ```<classifier_type>``` are 'logistic', 'svm', and 'neural'


### 3) testmodel.py

This file gives various options for testing a saved pickle file for either a scikit-learn svm or logistic regression classifier (it does not support saved pytorch models). It provides an interactive command line tool that allows you to enter sentences or lets you provide in NumPy arrays of text, encodings, and optionally labels, ultimately outputting a file that tabulates the model's predictions.

Usage Options:
```
  1. $ python testmodel.py "interactive" <model_pickle_file>                                                                          # for the command line interactive tool
  2. $ python testmodel.py "array" <model_pickle_file> <text_npy_file> <encoding_npy_file> <output_file_name>                         # to test unlabeled tweets
  3. $ python testmodel.py "array_label" <model_pickle_file> <text_npy_file> <encoding_npy_file> <label_npy_file> <output_file_name>  # to test labeled tweets
```
