import sys
import os
import pdb
import numpy as np
import json
import re

from bert_serving.client import BertClient

#it is critical that a Bert as a Service (baas) server be running on the machine, add line for IP from baas api to connect to remote machine
bc = BertClient(check_length = False)


directory = os.getcwd()
text_output_filename = sys.argv[1]
encoding_output_filename = sys.argv[2]
label_output_filename = sys.argv[3]
relevant_label = sys.argv[4]
text = []
ignore_ids = set()
files_list = os.listdir(directory)
if "ignore.txt" in files_list:
    with open("ignore.txt",'rb') as ignore_file:
        for line in ignore_file:
            ignore_ids.add(line)

for filename in files_list:
    if filename == "ignore.txt":
        continue
    data = {}
    print(filename)
    with open(filename,'rb') as json_file:
        data = json.load(json_file)
    for x in data['statuses']:
        lang_metadata_verified = False
        if 'metadata' in x.keys():
            if x['metadata']['iso_language_code'] != "en":
                continue
            else:
                lang_metadata_verified = True
        if 'lang' in x.keys():        
            if x['lang'] != "en" and lang_metadata_verified == False:
                continue
        if x['id_str'] in ignore_ids:
            continue
        if 'text' in x.keys():
            text.append(re.sub(r"http\S+", "",x['text']))
        elif 'full_text' in x.keys():
            text.append(re.sub(r"http\S+", "",x['full_text']))

text = [str for str in text if re.match('[a-zA-Z]', str)]
np.save(text_output_filename,text)
print(len(text))


labels = np.zeros(len(text))
if relevant_label == "1":
    labels = np.ones(len(text))
np.save(label_output_filename,labels)

encodings = bc.encode(text)
np.save(encoding_output_filename,encodings)


    