from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import pdb



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



def array(model_file,text_file,encoding_file,label_file,output_file):
    clf = pickle.load(open(model_file,'rb'))
    text = np.load(text_file)
    x = np.load(encoding_file)
    y = np.load(label_file)
    predictions = clf.predict(x)
    probas = clf.predict_proba(x)
    print_arr = ['Irrelevant','Relevant']
    f = open(output_file, 'w')
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
    print_metrics(y,predictions,probas[:,1])

def interactive(model_file):
    clf = pickle.load(open(model_file,'rb'))
    from bert_serving.client import BertClient
    import re

    #it is critical that a Bert as a Service (baas) server be running on the machine, add line for IP from baas api to connect to remote machine
    bc = BertClient(check_length = False)
    while(True):
        string = input("Enter text to classify: ")
        string = re.sub(r"http\S+", "", string)
        if not re.match('[a-zA-Z]',string):
            print("Please enter a non-empty string")
            continue
        else:
            encoding = bc.encode([string])
            prediction = clf.predict(encoding)
            if prediction[0]:
                print("Relevant")
            else:
                print("Irrelevant")

def main():
    usage_message = "Usage:\n1. python testmodel.py <model_pickle_file> array <text_npy_file> <encoding_npy_file> <label_npy_file> <output_file_name>\
                    \n2. python testmodel.py <model_pickle_file> interactive"
    if len(sys.argv) < 3:
        print(usage_message)
    elif len(sys.argv) == 3 and sys.argv[2] == "interactive":
        interactive(sys.argv[1])
    elif sys.argv[2] == "array":
        if len(sys.argv) == 7:
            array(sys.argv[1],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])
        else:
            usage_message = "Usage: python testmodel.py <model_pickle_file> array <text_npy_file> <encoding_npy_file> <label_npy_file> <output_file_name>"
            print(usage_message)
    else:
        print(usage_message)


main()