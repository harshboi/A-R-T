#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Unit Tests:
def unitTests():
    #Reading Tags
    nest_asyncio.apply()
    tags = getTags()
    tags = processTags(tags)
    #Checking if there was any hashtags left in there:
    for term in tags:
        if("#" in term):
            print("Found hashtag")
            break        
    print("Done checking for hashtags")
    #Collecting the tweets
    scrapeTweets(tags)
    data = processData()
    #Checking if clean data works:
    data = cleanData(data)
    print("Checking for any Empty tweets: ")
    for tweet in data:
        if(not tweet['tweet']):
            print("Found Empty Tweet")
    print("Done checking for Empty tweets")
    return

