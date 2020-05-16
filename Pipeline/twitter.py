import twint
import json



#Get all security tags that we will be searching twitter with
def getTags():
    #Reads tags from file  and adds each tag to an array
    file = open("./security_tags.txt","r")
    arr = []
    line1 = file.readlines()
    for x in line1:
        arr.append(x)
    file.close()
    return arr


#Process the tags so they are ready to be used with Twint
def processTags(arr):
    #Removes the hashtag from every tag
    for i in range(len(arr)):
        arr[i] = arr[i][1:]
    return arr



#Use twint to collect tweets
def scrapeTweets(arr):

    #Iterates through all security terms and searches twitter, writes all the data to a file called test1.json
    for i in range(len(arr)):
        d = twint.Config()
        d.Search = arr[i]
        d.Limit = 1
        d.Store_object = True
        d.Store_json = True
        d.Output = "./test1.json"
        twint.run.Search(d)
    return
