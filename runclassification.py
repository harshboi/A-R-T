from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
encodings = np.load('tweet_encodings.npy')
labels = np.load('maybeincludedlabels.npy')
x_train,x_test,y_train,y_test = train_test_split(encodings,labels, test_size=0.25, random_state=0)
print(x_train.shape)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train,y_train)
predictions = logisticRegr.predict(x_test)
accuracy_score = logisticRegr.score(x_test,y_test)
print("Accuracy score: {}".format(accuracy_score))
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test,predictions,pos_label = 1)
auc_score = metrics.auc(fpr,tpr)
print("AUC: {}".format(auc_score))