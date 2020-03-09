import sys
import os
import pdb
import numpy as np
import json
directory = os.getcwd()
output_file = sys.argv[1]
text = []
for filename in os.listdir(directory):
    data = {}
    print(filename)
    with open(filename,'rb') as json_file:
        data = json.load(json_file)
    for x in data['statuses']:
        if 'text' in x.keys():
            text.append(x['text'])
        elif 'full_text' in x.keys():
            text.append(x['full_text'])
    print(len(text))
np.save(output_file,text)
    