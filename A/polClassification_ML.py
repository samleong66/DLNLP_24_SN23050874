# -*- coding: utf-8 -*-
import logging
import os
import pickle
import numpy as np
from . import contextProcessing
from . import word2vecProcessing
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def getFeaturesFromContext(aspectContext,W2V):
    w2v_feature=[]
    for word in aspectContext.context.split(' '):
        try:
            w2v_feature.append(W2V.wv[word.lower()])
        except:
            w2v_feature.append([0 for i in range(W2V.vector_size)])
            #print('not find :%s'%word.lower())
    w2v_feature=np.array(w2v_feature).mean(axis=0)
    
    dep_feature=[]
    for dep in aspectContext.dep_context:
        for word in dep:
            try:
                dep_feature.append(W2V.wv[word.lower()])
            except:
                dep_feature.append([0 for i in range(W2V.vector_size)])
                #print('not find :%s'%word.lower())
    dep_feature=np.array(dep_feature).mean(axis=0)
    
    return np.concatenate((w2v_feature,dep_feature)).tolist()
    
def getInfoFromList(aspectContextList,W2V):
    features=[getFeaturesFromContext(ac,W2V) for ac in aspectContextList]
    pols=[ac.pol for ac in aspectContextList]
    return features,pols
    
def getFeaturesAndPolsFromFile(filepath,d_type='re',per=0.8):
    if d_type=='re':
        d_name='Restaurants'
    else:
        d_name='LapTops'
    train_data,test_data=contextProcessing.splitContextFile(filepath,per)
    print('Loading dataset...')
    W2V=word2vecProcessing.loadForWord('B/model/%s.w2v'%d_name)

    trainX,trainY=getInfoFromList(train_data,W2V)
    testX,testY=getInfoFromList(test_data,W2V)
    
    trainX, trainY, testX, testY=np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)
    return trainX,trainY,testX,testY
    
def trainMLClassifier(trainX,trainY,classifier='SVM'):  
    if classifier=='SVM':
        print('Using SVM for classification')
        clf=LinearSVC()
        clf=clf.fit(trainX,trainY)
    elif classifier == 'LR':
        print('Using Logistic Regression for classification')
        clf=LogisticRegression()
        clf=clf.fit(trainX,trainY)
    elif classifier == 'MLP':
        print('Using MLP for classification')
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 256, 512), random_state=1)
        clf = clf.fit(trainX, trainY)
        
    return clf
    
def predictByML(testX,testY,clf):
    print('Start predicting')
    true_result=clf.predict(testX)
    pre_result=testY
    
    print('Classification Results: \n')
    logging.info(classification_report(true_result, pre_result,digits=4))
    print(classification_report(true_result, pre_result,digits=4))
    
    clf.score(testX,testY)
    
def examByML(d_type='re',classifier='SVM',per=0.8):
    model_path = 'B/model/%s_clf.pickle'%classifier
    if d_type=='re':
        filepath='B/contextFiles/re_train.cox'
    else:
        filepath='B/contextFiles/lp_train.cox'
        
    trainX,trainY,testX,testY=getFeaturesAndPolsFromFile(filepath,d_type,per)
    if not os.path.exists(model_path):
        print("Start training %s classifier"%classifier)
        clf=trainMLClassifier(trainX,trainY,classifier)
        predictByML(testX,testY,clf)
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
    else:
        print("Use pretrained model")
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        logging.info("Classification results by %s"%classifier)
        predictByML(testX,testY,clf)




    
    
        