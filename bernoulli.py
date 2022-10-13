import time
import os
import numpy as np
import pandas as pd
import warnings
import sklearn
from sklearn.feature_extraction import text
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import metrics



warnings.filterwarnings('ignore')


## bernoulli naive bayes from scratch

class BernoulliNB():
        def fit(self, X, y, k=8):
            n,m = X.shape
        
            self.theta_k = np.zeros(k)
            self.parameterMatrix = np.zeros(shape = (k,m))
        
            for i in range(k):
                i_class_data = X[(y==i).flatten(), :]
                self.theta_k[i] = i_class_data.shape[0]/n
            
                # Laplace smoothing 
                ones = np.ones(shape = (1,m))
                zeros = np.zeros(shape = (1,m))
                i_class_data = np.vstack([i_class_data])
            
                self.parameterMatrix[i] = np.mean(i_class_data,axis = 0)
            
        def predict(self, X):
            probability = (X @ np.log(self.parameterMatrix.T + 1e-5)) + ((1-X) @ (np.log((1 - self.parameterMatrix.T) + 1e-5))) + np.log(self.theta_k + 1e-5)
            return np.argmax(probability, axis = 1) 
         
BNB = BernoulliNB()
            

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


    ### Data Preprocessing
def data_preprocess(train_file, test_data):
    # load
    data_train = np.array(pd.read_csv(train_file))
    data_test = np.array(pd.read_csv(test_data))
    X_train = data_train[:,0]
    y_train_origin = data_train[:,1]
    X_test = data_test[:,1]

    # preprocessing

    # label
    le = preprocessing.LabelEncoder()
    le.fit(['rpg','anime','datascience','hardware','cars','gamernews','gamedev','computers'])
    y_train = le.transform(y_train_origin)
    class_num = len(np.unique(y_train))

    # Vectorizer and Normalizer
    vectorizer = text.TfidfVectorizer(max_features = 2000, binary = True, stop_words = text.ENGLISH_STOP_WORDS)
    normalizer_train = preprocessing.Normalizer()
    
    vectors_train = vectorizer.fit_transform(X_train)
    vectors_test = vectorizer.transform(X_test)
    vectors_train = normalizer_train.transform(vectors_train).A
    vectors_test = normalizer_train.transform(vectors_test).A

    return vectors_train, y_train, vectors_test, class_num, le


def main(model, train_file, test_data, numFolds = 10, class_num = 8):
    start = time.time()

    print(model)

    # data preprocessing
    X_train, y_train, X_test, class_num, le = data_preprocess(train_file, test_data)

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


Test_pred = main(BNB, 'train.csv', 'test.csv', 10, 8)


os.getcwd()
os.chdir('/Users/patrickkampmeyer/Dropbox/Ph.D/Machine_Learning/Project')


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
from bs4 import BeautifulSoup
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

data_train['body_cleaned']=data_train['body']
data_train['body_cleaned'] = data_train['body_cleaned'].apply(clean_text)