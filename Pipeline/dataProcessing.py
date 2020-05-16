import json
import re
#Process data collected from twint so it is ready to be encoded by bert
def processData():
    f= open("./test1.json","r",errors='ignore')
    line1 = f.readlines()
    arr = []
    for x in line1:
        arr.append(x)
    f.close()
    arr.insert(0,"[")
    arr.append(']')
    for i in range(1,len(arr)-2):
        arr[i] = arr[i][:-1] + "," + arr[i][-1:]
    file = open("../test.json","w")
    for i in range(len(arr)):
        file.write(arr[i])
    file.close()
    f= open("../test.json","r",errors='ignore')
    line1 = f.readlines()
    data = []
    for x in line1:
        data.append(x)
    f.close()
    for i in range(len(data)):
        data[i] = (''.join([i if ord(i) < 128 else ' ' for i in str(data[i])]))
    file = open("./data.json","w")
    for i in range(len(data)):
        file.write(data[i])
    file.close()
    with open("./data.json", "r",errors='ignore') as read_file:
        data = json.load(read_file)
    return data


#Prepare Data for encoding
def cleanData(data):
    #parse json data into two arrays containing the tweet texts and their correponding relevancy label
    #track numbers of relevant and irrelevant tweets
    relnum =0
    irrnum = 0
    text = []
    labels = []
    newData = []
    seen_urls = set()
    duplicate_counter = 0

    for x in data:
        text.append(re.sub(r"http\S+", "", x['tweet']))
    #Add regex for removing  urls

    for i in data:
        if(re.match('[a-zA-Z]',i['tweet'])):
            newData.append(i)
            continue
        else:
            continue
    newtext = [str for str in text if re.match('[a-zA-Z]', str)]
    data =newData
#     getEncodings(newtext)
    return data




def removeIrrelevant(tweet):
    #Cleans up our data by only keeping relevant tweets
    if tweet['Relevance'] == "Relevant":
        return True
    else:
        return False
