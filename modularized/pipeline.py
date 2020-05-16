import nest_asyncio
import MLmodel
import twitter
import dataProcessing
import graphdb
import nlp
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

def main():

    data =  grabTweets()
    #Setting up model
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
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
    #Fetching driver for graphDB
    driver = graphdb.fetchDriver()
    #Passing Every Tweet through Pipeline
    for tweet in data:
        pipeline(tweet,model,device,driver)


def grabTweets():
        #Prevents async errors when using twint functionality
        nest_asyncio.apply()

        #Collecting Tweets for passing through pipeline
        tags = twitter.getTags()
        tags = twitter.processTags(tags)
        twitter.scrapeTweets(tags)
        data = dataProcessing.processData()
        data = dataProcessing.cleanData(data)
        return data

def pipeline(tweet,model,device,driver):
        #Passed tweet to model
        token_ids,attention_masks = MLmodel.tokenize_sentences([tweet['tweet']])
        model_outputs = (model(token_ids.to(device), token_type_ids=None, attention_mask=attention_masks.to(device)))
        softmax_layer = torch.nn.Softmax()
        result = softmax_layer(model_outputs[0])
        prediction = torch.argmax(result).item()
        confidence = torch.max(result).item()
        if not prediction:
            return
        tweet['Relevance'] = "Relevant"



        #Natural Language Processing for removing important words
        tweet = nlp.applyNLTK(tweet)
        tweet = nlp.applySpacy(tweet,2)

        #Add Graph DB Functionality
        graphdb.addToGraph(driver,tweet)
        return
if __name__ == "__main__":
    main()
