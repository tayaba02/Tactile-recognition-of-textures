from __future__ import absolute_import, division, print_function
 
from tensorflow.keras import Model, layers
import numpy as np
 
import pandas as pd
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU,Bidirectional
from numpy import genfromtxt

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

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation,ActivityRegularization
from keras.layers import Conv2D, MaxPooling2D,Flatten

from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import keras

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
    model = SVC(kernel='rbf', probability=True)
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    from sklearn import metrics
    print("Test Accuracy:",metrics.accuracy_score(ytest, y_pred))
    print("Training Accuracy:",metrics.accuracy_score(ytrain, model.predict(xtrain)))
    '''
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier, X_test, y_test, display_labels=class_names,
            cmap=plt.cm.Blues, normalize=normalize
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
    '''
    metric("SVM",model,xtrain,ytrain,xtest,ytest)




def ann(xtrain,ytrain,xtest,ytest):
    xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1]*xtrain.shape[2])
    xtest = xtest.reshape(xtest.shape[0],xtest.shape[1]*xtest.shape[2])    
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)
    model = Sequential()
    model.add(Dense(512, input_shape=(xtrain.shape[1],)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(20))
    model.add(Activation('softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # training the model and saving metrics in history
    history = model.fit(xtrain, ytrain,
          batch_size=128, epochs=50,
          validation_split=0.3)
    plotting(history)
    metric("ann",model,xtrain,ytrain,xtest,ytest)
    loss, acc = model.evaluate(xtest, ytest)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc))    
    
    
def cnn(xtrain,ytrain,xtest,ytest):
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
 
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    history = model.fit(xtrain, ytrain,
          batch_size=128, epochs=5,
          validation_split=0.3)

    loss, acc = model.evaluate(xtest, ytest)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc))
    

def sgd(xtrain,ytrain,xtest,ytest):
    xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1]*xtrain.shape[2])
    xtest = xtest.reshape(xtest.shape[0],xtest.shape[1]*xtest.shape[2])
    from sklearn.linear_model import SGDClassifier
    model =  SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    from sklearn import metrics
    print("Training Accuracy:",metrics.accuracy_score(ytest, y_pred))
    print("Test Accuracy:",metrics.accuracy_score(ytrain, model.predict(xtrain)))
    metric("SGD",model,xtrain,ytrain,xtest,ytest)   


def rf(xtrain,ytrain,xtest,ytest):
    xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1]*xtrain.shape[2])
    xtest = xtest.reshape(xtest.shape[0],xtest.shape[1]*xtest.shape[2])
    from skcdlearn.ensemble import RandomForestClassifier
    model =   RandcdomForestClassifier(n_estimators=10)
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    from sklearn import metrics
    print("Training Accuracy:",metrics.accuracy_score(ytest, y_pred))
    print("Test Accuracy:",metrics.accuracy_score(ytrain, model.predict(xtrain)))
    metric("RF",model,xtrain,ytrain,xtest,ytest)


#def rnn():
def rnn(xtrain,ytrain,xtest,ytest):
    xtrain = xtrain.reshape(xtrain.shape[0],1,xtrain.shape[1]*xtrain.shape[2])
    xtest = xtest.reshape(xtest.shape[0],1,xtest.shape[1]*xtest.shape[2])
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)
    print(xtrain.shape)
    print(xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)
    
    model = Sequential()
    model.add(SimpleRNN(1000, input_shape=(xtrain.shape[1:]), return_sequences=True))
    model.add(Dense(100))
    model.add(Activation("sigmoid")) 
    model.add(Dense(25))
    model.add(Activation("sigmoid")) 
    model.add(Dense(20))
    model.add(Activation("softmax")) 
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="SGD")
    history = model.fit(xtrain, ytrain,batch_size=30, epochs=30,validation_split=0.2,verbose=1)
    loss, acc = model.evaluate(xtest, ytest,verbose=0)
    Y_pred = model.predict(xtest)
    testAccuracy = metrics.accuracy_score(ytest.argmax(axis=1), model.predict(xtest).argmax(axis=1))
    trainAccuracy = metrics.accuracy_score(ytrain.argmax(axis=1), model.predict(xtrain).argmax(axis=1))
    print("testAccuracy", str(testAccuracy))
    print("trainAccuracy", str(trainAccuracy))   

    metric("RNN",model,xtrain,ytrain,xtest,ytest)
    
def gru(xtrain,ytrain,xtest,ytest):
    xtrain = xtrain.reshape(xtrain.shape[0],1,xtrain.shape[1]*xtrain.shape[2])
    xtest = xtest.reshape(xtest.shape[0],1,xtest.shape[1]*xtest.shape[2])
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)
    
    model = Sequential()
    model.add(GRU(1000, input_shape=(xtrain.shape[1:]), return_sequences=True))
    model.add(Activation("sigmoid"))     
    model.add(GRU(50))
    model.add(Activation("sigmoid")) 
    model.add(Dense(20))
    model.add(Activation("softmax")) 
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="SGD")
    history = model.fit(xtrain, ytrain,batch_size=30, epochs=30,validation_split=0.2,verbose=1)
    loss, acc = model.evaluate(xtest, ytest,verbose=0)
    Y_pred = model.predict(xtest)
    testAccuracy = metrics.accuracy_score(ytest.argmax(axis=1), model.predict(xtest).argmax(axis=1))
    trainAccuracy = metrics.accuracy_score(ytrain.argmax(axis=1), model.predict(xtrain).argmax(axis=1))
    print("testAccuracy", str(testAccuracy))
    print("trainAccuracy", str(trainAccuracy))
    metric("GRU",model,xtrain,ytrain,xtest,ytest)
    plotting("GRU",history)

def lstm(xtrain,ytrain,xtest,ytest):
    xtrain = xtrain.reshape(xtrain.shape[0],1,xtrain.shape[1]*xtrain.shape[2])
    xtest = xtest.reshape(xtest.shape[0],1,xtest.shape[1]*xtest.shape[2])
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)
    
    model = Sequential()
    model.add(LSTM(1000, input_shape=(xtrain.shape[1:]), return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(20))
    model.add(Activation("softmax")) 
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="SGD")
    history = model.fit(xtrain, ytrain,batch_size=30, epochs=30,validation_split=0.2,verbose=1)
    loss, acc = model.evaluate(xtest, ytest,verbose=0)
    Y_pred = model.predict(xtest)
    testAccuracy = metrics.accuracy_score(ytest.argmax(axis=1), model.predict(xtest).argmax(axis=1))
    trainAccuracy = metrics.accuracy_score(ytrain.argmax(axis=1), model.predict(xtrain).argmax(axis=1))
    print("testAccuracy", str(testAccuracy))
    print("trainAccuracy", str(trainAccuracy))
    metric("LSTM",model,xtrain,ytrain,xtest,ytest)
    plotting("LSTM",history)   
    
def bilstm(xtrain,ytrain,xtest,ytest):
    xtrain = xtrain.reshape(xtrain.shape[0],1,xtrain.shape[1]*xtrain.shape[2])
    xtest = xtest.reshape(xtest.shape[0],1,xtest.shape[1]*xtest.shape[2])
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)
    
    model = Sequential()
    model.add(Bidirectional(LSTM(1000, input_shape=(xtrain.shape[1:]), return_sequences=True)))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(20))
    model.add(Activation("softmax")) 
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="SGD")
    history = model.fit(xtrain, ytrain,batch_size=30, epochs=50,validation_split=0.2,verbose=1)
    loss, acc = model.evaluate(xtest, ytest,verbose=0)
    Y_pred = model.predict(xtest)
    testAccuracy = metrics.accuracy_score(ytest.argmax(axis=1), model.predict(xtest).argmax(axis=1))
    trainAccuracy = metrics.accuracy_score(ytrain.argmax(axis=1), model.predict(xtrain).argmax(axis=1))
    print("testAccuracy", str(testAccuracy))
    print("trainAccuracy", str(trainAccuracy))
    metric("BILSTM",model,xtrain,ytrain,xtest,ytest)
    plotting("biLSTM",history) 

#gru(xtrain,ytrain,xtest,ytest)
lstm(xtrain,ytrain,xtest,ytest)
#bilstm(xtrain,ytrain,xtest,ytest)
#rnn(xtrain,ytrain,xtest,ytest)
#svm(xtrain,ytrain,xtest,ytest)
#ann(xtrain,ytrain,xtest,ytest)
#sgd(xtrain,ytrain,xtest,ytest)
#rf(xtrain,ytrain,xtest,ytest)


