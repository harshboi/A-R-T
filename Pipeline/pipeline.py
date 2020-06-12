import nest_asyncio
import MLmodel
import twitter
import dataProcessing
import graphdb
import nlp
import torch
import json
import sys
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

def main(option,file,modelFile,username,password):
    if(option == '1'):
        print("Reading Json")
        data = readJson(file)
    if(option == '2'):
        print("Pulling tweets from Twint")
        data =  grabTweets(file)
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
    model.load_state_dict(torch.load(modelFile,map_location=device))
    model.eval()
    model.to(device)
    #Fetching driver for graphDB
    driver = graphdb.fetchDriver(username,password)
    #Passing Every Tweet through Pipeline
    for tweet in data:
        pipeline(tweet,model,device,driver)

def grabTweets(file):
        #Prevents async errors when using twint functionality
        nest_asyncio.apply()

        #Collecting Tweets for passing through pipeline
        tags = twitter.getTags(file)
        tags = twitter.processTags(tags)
        twitter.scrapeTweets(tags)
        data = dataProcessing.processData()
        data = dataProcessing.cleanData(data)
        return data

def readJson(file):
    with open("./"+ file, "r",errors='ignore') as read_file:
        data = json.load(read_file)
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
        tweet['nltk'] = nlp.remove_noise_from_tweet( tweet['tweet'] )
        tweet['spacy'] = nlp.applySpacy(nlp.remove_noise_from_tweet( tweet['tweet'] ),2)

        #Add Graph DB Functionality
        #Combine extracted Nouns
        nouns = tweet['nltk'] + tweet['spacy']
        #Remove duplicates
        nouns = list(set(nouns))
        date = tweet['date']
        
#        graphdb.addToGraph(driver,nouns,date,tweet['id'])
        
        response = vectorizer.transform( [ remove_noise_from_tweet( tweet['tweet'] ) ] )
        vectorizer = nlp.load_tf_idf_model ( "../Development/vectorizer.pickle" )
        ans = nlp.get_top_tf_idf_words(response, len(sentence))
        remove = set(ans[int(-len(ans)*0.7):])   # 0.7 = Return the bottom 70% of words in importance
        f_nouns = []
        for n in nouns:
            if n.lower() not in remove and n.lower() != "fyi":
                f_nouns.append(n.lower())
        if f_nouns == []: continue
    #     print(i, f_nouns)
        graphdb.addToGraph(driver, f_nouns, tweet['date'], tweet['id'], tweet['link'], tweet['username'], tweet['name'], tweet['time'])
        (*map(graphdb.check_relevance, repeat(driver), f_nouns), )

        return
if __name__ == "__main__":
    #Option 1: python pipeline.py 1 data.json
    #Option 2: python pipeline.py 2 security_tags.txt
    if(len(sys.argv) != 6):
        print("Invalid Command Line arguments")
        print("Try with one of these valid commands:")
        print("Option 1 for reading tweets from a json file: python pipeline.py 1 data.json model.pt graphDBUserName graphDBPassword")
        print("Option 2 for using twint to pull tweets: python pipeline.py 2 security_tags.txt model.pt graphDBUserName graphDBPassword")
    else:
        for i , arg in enumerate(sys.argv):
            if(i==1):
                option = arg
            if(i==2):
                file = arg
            if(i==3):
                model = arg
            if(i==4):
                user = arg
            if(i==5):
                password = arg
        print("option is: " + option)
        print("File is: " + file)
        print("Model is: "+ model)
        print("Username is: "+ user)
        print("Password is: "+ password)
        if(option != '1' and option != '2'):
            print("Invalid Option selected try again with 1 or 2")
        else:

            main(option,file,model,user,password)
