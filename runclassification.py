from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
import numpy as np
import sys

def print_metrics(y_true,y_pred):
    accuracy_score = metrics.accuracy_score(y_true,y_pred)
    print("Accuracy score: {}".format(accuracy_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred,pos_label = 1)
    auc_score = metrics.auc(fpr,tpr)
    print("AUC: {}".format(auc_score))

def logistic_regression(x_train,x_test,y_train,y_test):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train,y_train)
    predictions = logisticRegr.predict(x_test)
    print_metrics(y_test,predictions)


def support_vector(x_train,x_test,y_train,y_test):
    model = svm.SVC()
    model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    print_metrics(y_test,predictions)

def dense_layer(x_train,x_test,y_train,y_test):
    return

def main():
    encodings = np.load('tweet_encodings.npy')
    labels = np.load('maybeincludedlabels.npy')
    x_train,x_test,y_train,y_test = train_test_split(encodings,labels, test_size=0.25, random_state=0)
    if len(sys.argv) != 2:
        print("specify which classifier to use. Options are 'logistic', 'svm', or 'neural'")
    else:
        if sys.argv[1] == 'logistic':
            logistic_regression(x_train,x_test,y_train,y_test)
        elif sys.argv[1] == 'svm':
            support_vector(x_train,x_test,y_train,y_test)
        elif sys.argv[1] == 'neural':
            dense_layer(x_train,x_test,y_train,y_test)
        else:
            print("Options are 'logistic', 'svm', or 'neural'")
    
main()