from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import scikitplot as skplt
import matplotlib.pyplot as plt

#pytorch definition for 1 layer neural network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768,2)
    
    def forward(self,x):
        x = F.softmax(self.fc1(x))
        return x


#prints out accuracy and auc
def print_metrics(y_true,y_pred,y_probas,show_plot=True):
    accuracy_score = metrics.accuracy_score(y_true,y_pred)
    print("Accuracy score: {}".format(accuracy_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_probas,pos_label = 1)
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

#fits a logistic regression model and predicts on test set
def logistic_regression(x_train,x_test,y_train,y_test):
    logisticRegr = LogisticRegression(solver='lbfgs',max_iter=1000)
    logisticRegr.fit(x_train,y_train)
    train_predictions = logisticRegr.predict(x_train)
    train_probas = logisticRegr.predict_proba(x_train)
    test_predictions = logisticRegr.predict(x_test)
    test_probas = logisticRegr.predict_proba(x_test)
    print("Train Data Metrics:")
    print_metrics(y_train,train_predictions,train_probas[:,1])
    print("\nTest Data Metrics:")
    print_metrics(y_test,test_predictions,test_probas[:,1])

#fits a svm and predicts on test set
def support_vector(x_train,x_test,y_train,y_test):
    print("training model, takes a little time.")
    model = svm.SVC(probability=True,gamma='auto',verbose=1)
    model.fit(x_train,y_train)
    train_predictions = model.predict(x_train)
    train_probas = model.predict_proba(x_train)
    test_predictions = model.predict(x_test)
    test_probas = model.predict_proba(x_test)
    print("Train Data Metrics:")
    print_metrics(y_train,train_predictions,train_probas[:,1])
    print("\nTest Data Metrics:")
    print_metrics(y_test,test_predictions,test_probas[:,1])

#trains neural network and displays metrics for each epoch
def dense_layer(x_train,x_test,y_train,y_test):
    
    epoch_num = 100

    #initialize tensors to hold metrics
    train_probas = np.zeros((epoch_num,y_train.shape[0]))
    train_accuracies = np.zeros(epoch_num)
    train_auc = np.zeros(epoch_num)
    test_probas = np.zeros((epoch_num,y_test.shape[0]))
    test_accuracies = np.zeros(epoch_num)
    test_auc = np.zeros(epoch_num)
    
    #initialize model and datasets
    model = Net()
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

    #training iterations for network
    for epoch in range(epoch_num):
        model.train()
        for i,data in enumerate(x_train):
            optimizer.zero_grad()
            output = model(data)
            label = y_train[i].long().unsqueeze(0)
            loss = criterion(output.unsqueeze(0),label)
            loss.backward()
            optimizer.step()
        model.eval()

        #print and store metrics for each epoch, change 'False' to 'True' in print_metrics to generate ROC curves each epoch
        train_prediction = model(x_train)
        print("Epoch {}:\n".format(epoch+1))
        print("Training Data Metrics:")
        #pdb.set_trace()
        train_probas[epoch] = train_prediction[:,1].detach().numpy()
        train_accuracies[epoch], train_auc[epoch] = print_metrics(y_train.numpy(),train_prediction.max(1).indices.numpy(),train_prediction[:,1].detach().numpy(),False)
        test_prediction = model(x_test)
        print("\nTest Data Metrics:")
        test_probas[epoch] = test_prediction[:,1].detach().numpy()
        test_accuracies[epoch], test_auc[epoch] = print_metrics(y_test.numpy(),test_prediction.max(1).indices.numpy(),test_prediction[:,1].detach().numpy(),False)
        print("\n")

    
    #plot train accuracies and auc over epochs    
    plt.plot(train_accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Train Accuracy vs Epoch')
    plt.show()
    plt.plot(train_auc)
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.title('Train AUC vs Epoch')
    plt.show()    
    
    
    
    #plot test accuracies and auc over epochs    
    plt.plot(test_accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Test Accuracy vs Epoch')
    plt.show()
    plt.plot(test_auc)
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.title('Test AUC vs Epoch')
    plt.show()

    view_epoch = input("which epoch's ROC graph would you like to see: ")

    print("Training Metrics for Epoch {}:".format(view_epoch))
    print("Accuracy: {}".format(train_accuracies[view_epoch-1]))
    print("AUC: {}".format(train_auc[view_epoch-1]))
    fpr, tpr, thresholds = metrics.roc_curve(y_train,train_probas[view_epoch-1],pos_label = 1)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Training receiver operating characteristic, Epoch {}'.format(view_epoch))
    plt.show()



    print("Testing Metrics for Epoch {}:".format(view_epoch))
    print("Accuracy: {}".format(test_accuracies[view_epoch-1]))
    print("AUC: {}".format(test_auc[view_epoch-1]))
    fpr, tpr, thresholds = metrics.roc_curve(y_test,test_probas[view_epoch-1],pos_label = 1)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Testing receiver operating characteristic, Epoch {}'.format(view_epoch))
    plt.show()


#parses user input to determine which classifier to apply
def main():

    if len(sys.argv) != 4 and len(sys.argv) != 6: 
        print("usage: trainmodel.py <classifier> <training_tweets> <training_labels> [test_tweets] [test_labels]")
        print("       Options for <classifier> are 'logistic', 'svm', or 'neural'")
    else:
        x_train,x_test,y_train,y_test = None,None,None,None
        if len(sys.argv) == 4:
            train_encodings = np.load(sys.argv[2])
            train_labels = np.load(sys.argv[3])
            x_train,x_test,y_train,y_test = train_test_split(train_encodings,train_labels, test_size=0.25, random_state=0)
        else:
            x_train,y_train = shuffle(np.load(sys.argv[2]),np.load(sys.argv[3]),random_state=0)
            x_test = np.load(sys.argv[4])
            y_test = np.load(sys.argv[5])            
        if sys.argv[1] == 'logistic':
            logistic_regression(x_train,x_test,y_train,y_test)
        elif sys.argv[1] == 'svm':
            support_vector(x_train,x_test,y_train,y_test)
        elif sys.argv[1] == 'neural':
            dense_layer(x_train,x_test,y_train,y_test)
        else:
            print("Options for <classifier> are 'logistic', 'svm', or 'neural'")
    
main()
