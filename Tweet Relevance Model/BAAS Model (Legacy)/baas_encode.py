###################################################################################################################
# Author: Rohan Varma
# This file takes in a json object and generates 3 .npy files containing the tweet text, baas encodings, and labels.
# It filters out empty and duplicated tweets. It is critical to have a BERT-As-A-Server running in order for this
# script to work.
###################################################################################################################

from bert_serving.client import BertClient

#it is critical that a Bert as a Service (baas) server be running on the machine, add line for IP from baas api to connect to remote machine
bc = BertClient(check_length = False)

import numpy as np
import re
import json
import os.path
import sys
import pdb


#validate command line arguments
if len(sys.argv) < 5 or len(sys.argv) > 6 or (len(sys.argv) == 6 and sys.argv[5] != 'ignore'):
    print("Usage: runbert.py <json data input file path> <text output filename> <encoding output filename> <label output filename> [ignore maybes]\n Type in 'ignore' for [ignore maybes] to ignore maybe relevant tweets, otherwise script uses them by default")
else:    

    #toggle use maybe relevant tweets flag
    use_maybes = True
    if len(sys.argv) == 6:
        use_maybes = False

    #take in data,encoding, and label output filenames    
    data_file = os.path.normpath(sys.argv[1])
    encoding_filename = sys.argv[3]
    label_filename = sys.argv[4]

    #load json tweet data
    data = {}
    with open(data_file) as json_file:
        data = json.load(json_file)

    #parse json data into two arrays containing the tweet texts and their correponding relevancy label
    #track numbers of relevant and irrelevant tweets 
    relnum =0
    irrnum = 0
    text = []
    labels = []

    seen_urls = set()
    duplicate_counter = 0

    '''
    filter_text = "exploit"
    filter_rel = 0
    filter_irrel = 0
    '''
    for x in data:
        if re.sub(r"http\S+", "", x['tweet']) in seen_urls:
            duplicate_counter +=1
            continue
        else:
            seen_urls.add(re.sub(r"http\S+", "", x['tweet']))

        '''
        if filter_text in x['tweet']:
            if x['relevant'] == 0:
                filter_irrel += 1
            else:
                filter_rel += 1
        else:
            continue
        '''
        if x['relevant'] == 0:
            irrnum +=1
            text.append(re.sub(r"http\S+", "", x['tweet']))
            labels.append(x['relevant'])
        elif x['relevant'] == 1:
            relnum +=1
            text.append(re.sub(r"http\S+", "", x['tweet']))
            labels.append(x['relevant'])
        else:
            if use_maybes:
                relnum += 1
                text.append(re.sub(r"http\S+", "", x['tweet']))
                labels.append(1)

    #filter out tweets that are empty, and remove corresponding entries from label list
    newtext = [str for str in text if re.match('[a-zA-Z]', str)] 
    newlabel = [labels[i] for i,str in enumerate(text) if re.match('[a-zA-Z]',str)]

    #print statistics about extracted dataset
    print("Total Tweets: {}".format(len(newtext)))
    print("Number of Relevant Tweets: {}".format(sum(newlabel)))
    print("Number of Irrelevant Tweets: {}".format(len(newtext)-sum(newlabel)))
    print("Number of Duplicated Tweets Removed: {}".format(duplicate_counter))
    #print("Number of tweets containing {}: {}\n\tRelevant: {}\n\tIrrelevant: {}".format(filter_text,filter_irrel+filter_rel,filter_rel,filter_irrel))

    #encode tweet text and save numpy arrays for encodings and labels
    text_file_name = sys.argv[2]
    np.save(text_file_name,newtext)
    np.save(label_filename,newlabel)
    print("Finished saving labels, beginning encoding process, this might take a couple minutes!\nWe will notify on completion")
    encodings = bc.encode(newtext)
    np.save(encoding_filename,encodings)
    print("Completed encoding and saving!")
