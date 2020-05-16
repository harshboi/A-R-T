<div align="center">
  <img src="./Images/mcafee_logo.png", height="200" width="350">
</div>

--------------------------------------------------------------------------------

# A-R-T - Automated Research Tool: McAfee

AI Threat Intelligence is hunting for tweets on Twitter that can potentially indicate whether a tweet is talking about an emerging threat.

A-R-T uses a large number of technologies. Twint for dataset curation, Twitter API for streaming, Bert for Natural Language Processing, Model. Development in Tensorflow (Keras) & Pytorch, Spacy & NLTK for position entity tagging, Neo4j as the graph database and AWS MySql as the Relational database. 

## Install

#### 1. [PyTorch](https://pytorch.org/)

  Please use the selector on the PyTorch website to get the command appropriate for your system
  [PyTorch Installation Website](https://pytorch.org/get-started/locally/)

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

  [Install NLTK using:](https://www.nltk.org/install.html)

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

### Instructions on Running 
 1. Make sure that your security_tags.txt file and model.pt file are stored in your working directory.
 2. Run the pipeline using ```python pipeline.py```
 
Pipeline takes in 4 parameters
- A tweet
- The model
- The device for running the model
- The driver for connecting to the graph database

The model that is passed in is created using the code below:
```
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    model.load_state_dict(torch.load("model.pt",map_location=device))
    model.eval()
    model.to(device)
```

The device that is passed in is selected using the code below:
```
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
```

The driver for the graph database can be retrieved from the ```fetchDriver()``` function shown below:
```
driver = graphdb.fetchDriver()
```

To run the pipeline you use the pipeline function and pass all four parameters shown below:
```
pipeline(tweet,model,device,driver)
```

### Appendix:
Criticism and response to such <br>
Lack of Unit tests - Added Unittests to modularized <br>
Code not Modularized into files - Modularized code based on functionality into files <br>
Cluttered home directory - Moved all data files to datasets, created folders for segregating ML models based on framework used <br>
Instruction on running not provided in README - Added to README <br>

