from sklearn.model_selection import train_test_split
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
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768,2)
    
    def forward(self,x):
        x = F.softmax(self.fc1(x))
        return x



def print_metrics(y_true,y_pred):
    accuracy_score = metrics.accuracy_score(y_true,y_pred)
    print("Accuracy score: {}".format(accuracy_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred,pos_label = 1)
    auc_score = metrics.auc(fpr,tpr)
    print("AUC: {}".format(auc_score))
    return accuracy_score,auc_score

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

def torch_label_creator(label):
    if label.item() == 1:
        return torch.tensor([1,0])
    else:
        return torch.tensor([0,1])


def dense_layer(x_train,x_test,y_train,y_test):
    
    epoch_num = 50

    #initialize tensors to hold metrics
    train_accuracies = np.zeros(epoch_num)
    train_auc = np.zeros(epoch_num)
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

        #print and store metrics for each epoch
        train_predictions = model(x_train)
        print("Epoch {}:\n".format(epoch+1))
        print("Training Data Metrics:")
        train_accuracies[epoch], train_auc[epoch] = print_metrics(y_train,train_predictions.max(1).indices.numpy())
        test_predictions = model(x_test)
        print("\nTest Data Metrics:")
        test_accuracies[epoch], test_auc[epoch] = print_metrics(y_test,test_predictions.max(1).indices.numpy())
        print("\n")

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