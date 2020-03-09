import sys
from sklearn import svm
import pickle
import pdb
import numpy as np
from sklearn import metrics
import re
import matplotlib.pyplot as plt
from bert_serving.client import BertClient
bc = BertClient(check_length = False)
model_path = sys.argv[1]




original_text = np.load('mcafee_test_text.npy')
original_labels = np.load('mcafee_test_labels.npy')

first_text = []
first_labels = []
relnum =0
irrnum = 0


seen_urls = set()
duplicate_counter = 0
for i,x in enumerate(original_text):
        if re.sub(r"http\S+", "", x) in seen_urls:
            duplicate_counter +=1
            continue
        else:
            seen_urls.add(re.sub(r"http\S+", "", x))

        if original_labels[i] == 0:
            irrnum +=1
            first_text.append(re.sub(r"http\S+", "", x))
            first_labels.append(0)
        elif original_labels[i] == 1:
            relnum +=1
            first_text.append(re.sub(r"http\S+", "", x))
            first_labels.append(1)

#print statistics about extracted dataset
print("Total Tweets: {}".format(relnum+irrnum))
print("Number of Relevant Tweets: {}".format(relnum))
print("Number of Irrelevant Tweets: {}".format(irrnum))
print("Number of Duplicated Tweets Removed: {}".format(duplicate_counter))

#filter out tweets that are empty, and remove corresponding entries from label list
text = [str for str in first_text if re.match('[a-zA-Z]', str)] 
labels = [first_labels[i] for i,str in enumerate(first_text) if re.match('[a-zA-Z]',str)]

encodings = bc.encode(text)
np.save('mcafee_test_encodings',encodings)


#prints out accuracy and auc
def print_metrics(y_true,y_pred,y_probas,show_plot=True):
    accuracy_score = metrics.accuracy_score(y_true,y_pred)
    print("Accuracy score: {}".format(accuracy_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_probas,pos_label = 1)
    '''
    print("Thresholds: {}".format(thresholds))
    print("TPR: {}".format(tpr))
    print("FPR: {}".format(fpr))
    '''
    auc_score = metrics.auc(fpr,tpr)
    print("AUC: {}".format(auc_score))
    if show_plot:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.show()
    return accuracy_score,auc_score


def array_label(model_file,text_file,encoding_file,label_file,output_file):
    clf = pickle.load(open(model_file,'rb'))
    text = text_file
    x = encoding_file
    y = label_file
    predictions = clf.predict(x)
    probas = clf.predict_proba(x)
    print_arr = ['Irrelevant','Relevant']
    '''f = open(output_file, 'w')
    f.close()
    for i,t in enumerate(text):
        f = open(output_file, 'a')
        f.write("Tweet: "+t+'\n')
        f.write("Label: "+str(print_arr[y[i]])+'\n')
        f.write("Prediction: "+str(print_arr[predictions[i]])+'\n\n')
        f.close()
        print("Tweet: "+t)
        print("Label: "+str(print_arr[y[i]]))
        print("Prediction: "+str(print_arr[predictions[i]])+"\n")
    '''
    print_metrics(y,predictions,probas[:,1])

array_label(model_path,text,encodings,labels,"mcafee_test_predictions.txt")

