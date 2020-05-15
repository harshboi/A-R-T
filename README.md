<div align="center">
  <img src="./Images/mcafee_logo.png", height="200" width="350">
</div>

--------------------------------------------------------------------------------

# A-R-T - Automated Research Tool: McAfee

AI Threat Intelligence is hunting for tweets on Twitter that can potentially indicate whether a tweet is talking about an emerging threat.

A-R-T uses a large number of technologies. Twint for dataset curation, Twitter API for streaming, Bert for Natural Language Processing, Model. Development in Tensorflow (Keras) & Pytorch, Spacy & NLTK for position entity tagging, Neo4j as the graph database and AWS MySql as the Relational database. 

## Install

#### 1) Bert As A Service (BAAS)
  See the [Bert As A Service install guide](https://github.com/hanxiao/bert-as-service/#user-content-bert-as-service)

  Special Instructions: Setup a Conda/pip environment with the tf version less than 1.15 as graph generation from other versions do not work with BAAS

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
  Note: Functions for such are present in final_pipeline.py. Check documentation for such for using them.

#### 2. [Hugginface Transformer - PyTorch](https://huggingface.co/)

  [Install Transformers using:](https://huggingface.co/transformers/installation.html)

  ```
  $ pip install transformers
  ```

  Load Pre-trained model:

  ```
  >>> # Load pre-trained model tokenizer (vocabulary)
  >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  ```

    Note: Functions for such will be present in final_pipeline.py in the future. Refer to finetune.py currently

#### 3. [Spacy](https://spacy.io/usage/linguistic-features)

  [Install Spacy using:](https://spacy.io/usage)

  ```
  $ python -m spacy download en
  $ python -m spacy download en_core_web_sm
  ```

  After this simply import and have fun

  ```
  >>> import spacy
  >>> nlp = spacy.load('en_core_web_sm')
  ```
  Note: Functions for such present in final_pipeline.py. Check in-file documentation on calling them
  
#### 4. [NLTK](https://www.nltk.org/)

  [Install NLTK Using:](https://www.nltk.org/install.html)

  ```
  $ pip install --user -U nltk
  ```

  Download NLTK data using

  ```
  >>> import nltk
  >>> nltk.download()
  ```

  Note: Functions for such present in final_pipeline.py. Check in-file documentation on calling them.

#### 5. [Neo4j](https://neo4j.com)

  [Install Python API using:](https://neo4j.com/developer/python/)

  ```
  $ pip install neo4j
  ```

  To establish connection to DB:

  ```
  >>> from neo4j import GraphDatabase
  >>> driver = GraphDatabase.driver("bolt://localhost",auth=("test_user","password"), encrypted=False)
  ```

  Note to be functionally compliant with final_pipeline.py, create user and password "test_user" and "Password" respectively

#### 6. [Twint](https://github.com/twintproject/twint/#user-content-twint---twitter-intelligence-tool)

  Install Using

  ```
  $ pip3 install twint
  ```

  Note: Functions for such present in the final_pipeline.py. Check in-file documentation on calling them.

#### 7. [Twitter API](https://developer.twitter.com/en/docs/basics/getting-started)

  Requires getting approved for a Twitter developer account. Streaming API requires a paid version (Enterprise) of the Twitter   API. We have a basic connection to the twitter API established.

  [Twitter documentation for authenticating/connecting to Twitter API](https://developer.twitter.com/en/docs/basics/authentication/oauth-1-0a/creating-a-signature) - Most time consuming step

  Here is a list of our keys that might help in looking for what keys should look like:

  ```
  >>> oauth_consumer_key = ZYOj2R1VYsIUhaRvsrcRuHxk6
  >>> oauth_nonce = kYjzVBB8Y0ZFabxSWbWovY3uYSQ2pTgmZeNu2VS4cg
  >>> Consumer Secret = API Secret Key = uTwMxW2CPXfv8wZCcqOSX8Yw46eqJVMNopqh5kh9O0VXiVQNRG
  >>> Access token secret = OAuth token secret: VB6fxb1akdtwRDf52sbQlPKE9tpHEPX1ZBSI3Y1t33REk
  >>> Signing key = uTwMxW2CPXfv8wZCcqOSX8Yw46eqJVMNopqh5kh9O0VXiVQNRG&VB6fxb1akdtwRDf52sbQlPKE9tpHEPX1ZBSI3Y1t33REk
  >>> oauth-token = 797644398670409728-I3aUqpmhD7uPscfFZFxMGfMWXPBmiaN
  >>> oauth_signature = Use HMAC
  ```

  Establishing a connection:

  ```
  >>> import twitter
  >>> api = twitter.Api(consumer_key="ZYOj2R1VYsIUhaRvsrcRuHxk6",
                        consumer_secret="uTwMxW2CPXfv8wZCcqOSX8Yw46eqJVMNopqh5kh9O0VXiVQNRG",
                        access_token_key="797644398670409728-Zwgcl9kcCFerhFNlFFGwR3emSbfpfpX",
                        access_token_secret="CzVCCqD8X9FC059X98deDiNYb24IWjhZYVeAhoU4F5v7l")
  ```
  Note: Functions for such present in the Twint Scraper.ipynb (Name for notebook will be changed in future for better           readability). Check in-file documentation on using them.

Instructions on Running:

Appendix:
Criticism and response to such
Lack of Unit tests - Added Unittests to modularized and 
Code not Modularized into files - Modularized code based on functionality into files
Cluttered home directory - Moved all data files to datasets, created folders for segregating ML models based on framework used
Instruction on running not provided in README - Added to README

### Running Pipeline
  Ensure the following dependencies are downloaded:
  
