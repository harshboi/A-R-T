<div align="center">
  <img src="../../Images/mcafee_logo.png", height="200" width="350">
</div>

--------------------------------------------------------------------------------


# Tweet Relevancy Model using Hugging

This folder contains files used for creating a finetuned BERT model that tests whether a tweet is cybersecurity threat related or not. The base architecture is that a BAAS server is spun up that hosts a pretrained BERT model which returns vector encodings for sentences that are sent to it. These encodings are then passed onto a classification layer to get the final prediction of "Relevant" or "Irrelevant". The three classification layers that were implemented and tested were support vector machine, logistic regression, and single dense neural layer.

## Installation Requirements

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

#### 3. Scikit-Learn

  [Install Scikit-Learn using:](https://scikit-learn.org/stable/install.html)
```
  $ pip install -U scikit-learn
```
#### 4) Matplotlib

  [Install matplotlib using:](https://matplotlib.org/3.2.1/users/installing.html)
```
  $ pip install -U matplotlib
```

#### 5) Pandas

  [Install Pandas using:](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
```
  $ pip install -U pandas
```

## Contents

#### 1) finetune.py

This script takes in NumPy arrays for training and testing sentences and labels, loads a pretrained BERT model, adds a single layer on top of it, and finetunes the model on the given data. This script was created by following the tutorial at https://mccormickml.com/2019/07/22/BERT-fine-tuning/ and adapting it to work with our data.

Usage:
```
  $ python finetune.py <train sentences> <train labels> <validation sentences> <validation labels>
```
```<train sentences>```, ```<train labels>```, ```<validation sentences>```, and ```<validation labels>``` should all be .npy files.

#### 2) interactive_tool.py

This script takes in a PyTorch model state and provides an interactive command line console to see which sentences are deemed relevant or irrelevant

Usage:
```
  $ python interactive_tool.py <pytorch model path>
```
```<pytorch model path>``` should be either a .pt or .pth file

#### 3) test_model.py

This script takes in a PyTorch model state as well as two NumPy arrays containing sentences and labels. It then runs all the given sentences through the given model and provides performance metrics as well as two text files. One text file contains the model's predictions on all sentences whereas the other contains the model's predictions on all sentences it classified incorrectly.

Usage:
```
  $ python testmodel.py <pytorch model path> <sentences> <labels> <predictions output file path> <incorrect predictions output file path>
```
```<pytorch model path>``` should be either a .pt or .pth file, ```<sentences>``` and ```<labels>``` should be .npy files, and ```<predictions output file path>``` and ```<incorrect predictions output file path>``` should be .txt files


