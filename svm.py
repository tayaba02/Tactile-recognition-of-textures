from __future__ import absolute_import, division, print_function
 
from tensorflow.keras import Model, layers
import numpy as np
 
import pandas as pd
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU,Bidirectional
from numpy import genfromtxt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation,ActivityRegularization
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
 
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
scaler = StandardScaler()
import keras
from sklearn.metrics import precision_score, recall_score, accuracy_score 
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation,ActivityRegularization
from keras.utils.np_utils import to_categorical
from sklearn import metrics
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation,ActivityRegularization
from keras.layers import Conv2D, MaxPooling2D,Flatten

from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation,ActivityRegularization
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from numpy import load
#plotting f

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt
from pylab import rcParams
def textplotting(model,x_test,y_test):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_pred = y_pred.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    
def metric(name,best_model,x_train,y_train,x_test,y_test):
  plt.clf()
  y_pred = best_model.predict(x_test)
  sn.set(font_scale=1)
  rcParams['figure.figsize'] = 30, 30
  confusion_matrix = pd.crosstab(y_test.argmax(axis=1), y_pred.argmax(axis=1), rownames=['Actual'], colnames=['Predicted'])
  sn.heatmap(confusion_matrix, annot=True)

 

  plt.savefig(name+"Test.png")
  plt.clf()
  confusion_matrix = pd.crosstab(y_train.argmax(axis=1), best_model.predict(x_train).argmax(axis=1), rownames=['Actual'], colnames=['Predicted'])
  sn.heatmap(confusion_matrix, annot=True)

 

  plt.savefig(name+"Train.png")
  plt.clf()

def plotting(name,history):
  plt.clf()
  sn.set(font_scale=2)
  rcParams['figure.figsize'] = 10, 10
  
  fig = plt.figure()
  history_dict = history.history
  print(history_dict.keys())
  plt.subplot(2,1,1)
  plt.plot(history_dict['accuracy'])
  plt.plot(history_dict['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Training Set', 'Validation Set'], loc='lower right')
  

 

  plt.subplot(2,1,2)


  plt.plot( history_dict['loss'])
  plt.plot( history_dict['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Training Set', 'Validation Set'], loc='upper right')

 

  plt.tight_layout()
  plt.savefig(name +"Accuracy.png")
  plt.clf()
  plt.show
#Bio
ytrain = load('labels_trainAll.npy')
print(ytrain.shape)
xtrain = load('bio_trainAll.npy')
print(xtrain.shape)
xtest = load('bio_test.npy')
print(xtest.shape)
ytest = load('labels_test.npy')
print(ytest.shape)


'''
ytrain = load('labels_trainAll.npy')
print(ytrain.shape)
xtrain = load('icub_trainAll.npy')
print(xtrain.shape)


xtest = load('icub_test.npy')
print(xtest.shape)
ytest = load('labels_test.npy')
print(ytest.shape)
'''
#### 2D classifiers
def svm(xtrain,ytrain,xtest,ytest):
    xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1]*xtrain.shape[2])
    xtest = xtest.reshape(xtest.shape[0],xtest.shape[1]*xtest.shape[2])
    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=True)
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    from sklearn import metrics
    print("Test Accuracy:",metrics.accuracy_score(ytest, y_pred))
    print("Training Accuracy:",metrics.accuracy_score(ytrain, model.predict(xtrain)))
    
svm(xtrain,ytrain,xtest,ytest)