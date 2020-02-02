print("here")
from bert_serving.client import BertClient
bc = BertClient(check_length = False)
import numpy as np
import re
import json
data = {}
with open('data.json') as json_file:
    data = json.load(json_file)
i = 0
for x in data:
    if x['relevant'] == 2:
        x['relevant'] = 1
    i += 1
relevantnum =0
irrnum = 0
print("here")
text = []
labels = []
for x in data:
    text.append(x['tweet'])
    labels.append(x['relevant'])
    if x['relevant'] == 1:
        relevantnum +=1
    else:
        irrnum +=1

print(len(text))
print("printing first line:")
print(text[0])
print("printing second line:")
print(text[1])
newtext = [str for str in text if re.match('[a-zA-Z]', str)] 
newlabel = [labels[i] for i,str in enumerate(text) if re.match('[a-zA-Z]',str)]
print(len(newtext))
print(len(newlabel))
print(newtext[0])
print(newlabel[0])
np.save("maybeincludedlabels",newlabel)
encodings = bc.encode(newtext)
print(len(encodings))
print(type(encodings))
np.save("tweet_encodings",encodings)
