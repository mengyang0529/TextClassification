import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, Convolution1D, MaxPooling1D, Input, LSTM
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.layers import merge

from keras.wrappers.scikit_learn import KerasClassifier
vocab_size=0
max_len=0
def preview(path):
    dataFile = pd.read_csv(path, sep='\t', names=["Label", "Text"])
    dataFile.head()
    train_df_text_len = dataFile.Text.str.split(",").apply(len)
    #show data statistic
    print(train_df_text_len.describe())
    #dataFile.Label.hist()
    #plt.title("Training Label Distribution")
    #plt.show()

def hist(data,minValue,maxValue,bin):
    bins = np.arange(minValue, maxValue, bin) # fixed bin size
    plt.xlim([minValue-bin, maxValue+bin])
    plt.hist(data, bins=bins)
    plt.title('Hist')
    plt.xlabel('variable X (bin size = 1)')
    plt.ylabel('count')
    plt.show()

def padding(trianPath,testPath):
    train_data = pd.read_csv(trianPath, sep='\t', names=["Label", "Text"])
    train_data_text_len = train_data.Text.str.split(",").apply(len)

    X_train = np.array(train_data.Text)
    Y_train = np.array(train_data.Label)

    test_data = pd.read_csv(testPath, sep=" ", names=["Text"])
    test_data_text_len = test_data.Text.str.split(",").apply(len)

    X_test = np.array(test_data.Text)
    X_total = np.concatenate((X_train, X_test))
    tknzr = Tokenizer(lower=False, split=',')
    tknzr.fit_on_texts(X_total)

    XS_train = tknzr.texts_to_sequences(X_train)

    XS_test = tknzr.texts_to_sequences(X_test)
    max_len = train_data_text_len.max()\
        if train_data_text_len.max() > test_data_text_len.max()\
        else test_data_text_len.max()
    print("max_len = ",max_len)
    XS_train = sequence.pad_sequences(XS_train, maxlen=max_len)
    XS_test = sequence.pad_sequences(XS_test, maxlen=max_len)

    vocab_size = len(tknzr.word_counts)
    print("word_counts = ",vocab_size)

    #hist(XS_train,100,127,1)
    pca = PCA(n_components = 64)
    XS_pca_train=pca.fit_transform(XS_train)
    XS_pca_test=pca.fit_transform(XS_test)
    return Y_train,XS_pca_train,XS_pca_test

def train(Y_train,X_train):
    model = GradientBoostingClassifier(n_estimators=1024, learning_rate=0.15, max_depth=3, random_state=0)\
    .fit(X_train, Y_train)
    score = model.score(X_train, Y_train)
    print("train score=",score)
    return model

def predict(model,X_test):
    result = model.predict(X_test);
    with open("output.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in result:
            writer.writerow([val])
    df = pd.DataFrame(result,columns = ['test result'])
    ax = df.plot.hist()
    plt.title('Hist')
    plt.xlabel('variable X (bin size = 1)')
    plt.ylabel('count')
    plt.show()

