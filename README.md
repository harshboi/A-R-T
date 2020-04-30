# A-R-T - Automated Research Tool
Repository for AI Threat Intelligence Capstone

AI Threat Intelligence is hunting for tweets on Twitter that can potentially indicate whether a tweet is talking about an emerging threat.

A-R-T uses a large number of technologies. Bert for Natural Language Processing, Model. Development in Tensorflow (Keras) & Pytorch, Spacy & NLTK for position entity tagging, Neo4j as the graph database and AWS MySql as the Relational database. 

## Install

1)
  See the [Bert As A Service (BAAS) install guide] (https://github.com/hanxiao/bert-as-service/#user-content-bert-as-service)

  Special Instructions: Setup a Conda/pip environment with the tf version less than 1.15 as graph generation from other versions do not work with BAAS

  1. BAAS has a server and client architecture. Install them using 

  ```
  $ pip install bert-serving-server  # server
  $ pip install bert-serving-client  # client, independent of `bert-serving-server`
  ```

  Download the pretrained BERT model - [Bert Large-Cased] (https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)

  2. Start the BERT service
  After installing the server, you should be able to use bert-serving-start CLI as follows:

  bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 
  This will start a service with four workers, meaning that it can handle up to four concurrent requests. More concurrent requests will be queued in a load balancer. Details can be found in our FAQ and the benchmark on number of clients.

  3. Use Client to Get Sentence Encodes
  Now you can encode sentences simply as follows:

  from bert_serving.client import BertClient
  bc = BertClient()
  bc.encode(['First do it', 'then do it right', 'then do it better'])

  Note: Functions for such are present in the final_pipeline.py. Check documentation for such for using them.

2. [Spacy] (https://spacy.io/usage/linguistic-features)

  [Install Spacy using:] (https://spacy.io/usage)

  ```
  $ python -m spacy download en
  $ python -m spacy download en_core_web_sm
  ```

  After this simply import and have fun

  ```
  >>> import spacy
  >>> nlp = spacy.load('en_core_web_sm')
  ```
  Note: Functions for such present in the final_pipeline.py. Check documentation on calling them
  
3. [NLTK] (https://www.nltk.org/)

  [Install NLTK Using:] (https://www.nltk.org/install.html)

  ```
  $ pip install --user -U nltk
  ```

  Download NLTK data using

  ```
  >>> import nltk
  >>> nltk.download()
  ```

  Note: Functions for such present in the final_pipeline.py. Check documentation on calling them
