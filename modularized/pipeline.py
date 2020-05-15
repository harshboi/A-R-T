import nest_asyncio
import model
import twitter
import dataProcessing
import nlp

def main():
    #Prevents async errors when using twint functionality
    nest_asyncio.apply()

    #Collecting Tweets for passing through pipeline
    tags = twitter.getTags()
    tags = twitter.processTags(tags)
    twitter.scrapeTweets(tags)
    data = dataProcessing.processData()
    data = dataProcessing.cleanData(data)

    #Passing Every Tweet through Pipeline
    for tweet in data[:50]:
        #Get tweet encoding
        encoding = model.getEncodings(tweet['tweet'])

        #Classify the tweet as either relevant or irrelevant
        tweet = model.classifyTweets("svm_model.p",tweet,encoding,"classifiedTweets.txt")

        #Break on irrelevant tweets
        if(not dataProcessing.removeIrrelevant(tweet)):
            continue

        #Natural Language Processing for removing important words
        tweet = nlp.applyNLTK(tweet)
        tweet = nlp.applySpacy(tweet,2)

        #Add Graph DB Functionality

    print(data[4])


if __name__ == "__main__":
    main()
