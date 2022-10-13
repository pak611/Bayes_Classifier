import os
import string
import numpy as np
import time
import pandas as pd
import warnings
from sklearn.feature_extraction import text
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import metrics
import shutil
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report


os.chdir('/Users/patrickkampmeyer/Dropbox/Ph.D/Machine_Learning/Project/Picture_Data')


data_set = np.array(pd.read_csv('mnist_train.csv'))

Y = data_set[:,0]

X = data_set[:,1:]

### K-fold Cross Validation
def kfold_validation(model, X, y, numFolds=10):

    kf = KFold(n_splits=numFolds,shuffle=True,random_state=0)
    print("\nIn this K-fold Cross Validation, K = ", kf.n_splits)
    totalAccuracy = 0

    for train_index, test_index in kf.split(X):
        X_train,X_test = X[train_index], X[test_index]
        y_train,y_test = y[train_index], y[test_index]

        model.fit(X = X_train, y = y_train)
        y_pred = model.predict(X_test)
        totalAccuracy += metrics.accuracy_score(y_true=y_test, y_pred=y_pred)

    return totalAccuracy / numFolds


    # we need test data to be in the data to get a score... so lets divide the train.csv dataset up into training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=0)


### Preprocessing
def preprocess(data_set):
    # load
    data_set = np.array(pd.read_csv('mnist_train.csv'))

    Y = data_set[:,0]

    X = data_set[:,1:]
    
    # we need test data to be in the data to get a score... so lets divide the train.csv dataset up into training and testing
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=0)

   # y_train_origin = train_split[:,1]
   # y_train_start = train_split[:,1]
   # X_test = test_split[:,1]

    # preprocessing

    # transform to numeric label
    
    le = preprocessing.LabelEncoder()
    le.fit([0,1,2,3,4,5,6,7,8,9])
    y_train = le.transform(y_train)
    # add in here
    y_test = le.transform(y_test)
    class_num = len(np.unique(y_train))

    # Transform from strings into counts of strings and then normalize
    
    vectorizer = text.TfidfVectorizer(max_features = 785, binary = True, stop_words = text.ENGLISH_STOP_WORDS)
    normalizer_train = preprocessing.Normalizer()
    
    # UNCOMMENT WHEN USING PIXEL DATA
 #  X_vectors_train = vectorizer.fit_transform(X_train)
#   X_vectors_test = vectorizer.transform(X_test)

    X_vectors_train = normalizer_train.transform(X_train)
    X_vectors_test = normalizer_train.transform(X_test)

    return X_vectors_train, y_train, X_vectors_test, y_test, class_num, le

vectors_train, y_train, vectors_test, y_test, class_num, le = preprocess('mnist_train.csv')


X = vectors_train
y = y_train


### Bernoulli Naïve Bayes from Scratch
class Bernoulli():
    
    def fit(self, X, y, k = 10):
        
        alpha = 1.0
        V_size = 785
        n,m = X.shape
        self.cond_prob = np.empty((m,k))
        self.prior = np.empty(10)

        #self.theta_k = np.zeros(k)
        #self.parameterMatrix = np.zeros(shape = (k,m))
        for i in range(k): 
            i_class_data = X[(y==i).flatten(), :] #for class i=1 for instance.. X.shape=(8686,2000) -> i_class_data.shape -> (1535,2000)
        
            self.prior[i] = i_class_data.shape[0]/n # equal to (number of instances in class i)/(total number of instances in X)
            self.cond_prob[:,i] = (i_class_data.mean(0)+alpha)/(self.prior[i]+(alpha*V_size)) #conditional probability for all of the words for the 
                 
                                 
                                 
    def predict(self, X_test):
        
        Bern_X_test = np.where(X_test > 0, 1, 0)
        
        probability = (Bern_X_test @ np.log(self.cond_prob)) + ((1-Bern_X_test) @ (np.log((1 - self.cond_prob) + 1e-5)))
                                 
        return np.argmax(probability, axis = 1) 
        # should be multiplied by the prior rather than by X



  ### Bernoulli Naïve Bayes from Scratch
class Multinomial():
    
    def fit(self, X, y, k = 10):
        
        alpha = 1.0
        V_size = 2000
        n,m = X.shape
        self.cond_prob = np.empty((m,k))
        self.prior = np.empty(10)

        #self.theta_k = np.zeros(k)
        #self.parameterMatrix = np.zeros(shape = (k,m))
        for i in range(k): 
            i_class_data = X[(y==i).flatten(), :] #for class i=1 for instance.. X.shape=(8686,2000) -> i_class_data.shape -> (1535,2000)
        
            self.prior[i] = i_class_data.shape[0]/n # equal to (number of instances in class i)/(total number of instances in X)
            self.cond_prob[:,i] = (i_class_data.mean(0)+alpha)/(self.prior[i]+(alpha * V_size)) #conditional probability for all of the 
    
    
    def predict(self, X_test):
        
        probability = (X_test @ np.log(self.cond_prob))
        
        return np.argmax(probability, axis = 1)



#BNB = Bernoulli()
BNB = Multinomial()


def main(model, train_file, test_data, numFolds = 5, class_num = 10):
    start = time.time()

    print(model)

    # data preprocessing
    X_train, y_train, X_test, y_test, class_num, le = preprocess(train_file)

    # K-fold Cross Validation
    
    Average_accuracy = kfold_validation(model, X_train, y_train, numFolds)
    print('\nThe average accuracy of {}-fold cross validation is: {:.5f}'.format(numFolds, Average_accuracy))

    # train with the whole training dataset
    model.fit(X_train, y_train, class_num)
    y_train_pred = model.predict(X_train)
    print('\nPerformance metrics:\n', metrics.classification_report(y_train, y_train_pred))

    # predict with test dataset
    y_test_pred = model.predict(X_test)

    # final output
    y_pred_inversed = le.inverse_transform(y_test_pred)
    print('\nPredicted labels of the test dataset:\n', y_pred_inversed)
    
    # Run time
    end = time.time()
    run_time = end - start
    print('\nRun time: {:.2f}s\n'.format(run_time))

    return y_pred_inversed


Test_pred = main(BNB, 'mnist_train.csv', 'mnist_test.csv', 5, 10)
