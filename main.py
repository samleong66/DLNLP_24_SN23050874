import torch
import pandas as pd
import datetime
import logging
from A.preProcessing import *
from A.word2vecProcessing import trainingEmbedding, createCluster
from A.exam_CRF import evaluate
from A.contextProcessing import createAllForPol
from A.polClassification_ML import examByML
from A.nnModels import examByNN
from A.preparation import get_parser

import os, sys
JAVA_HOME = r"D:\Program Files\Java\jdk-22\bin\java.exe"
os.environ.setdefault('JAVA_HOME', JAVA_HOME)

current_time = datetime.datetime.now().strftime("%m%d%H%M")
log_name = os.path.join("B/logging", f"{current_time}.log")
logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    transformJSONFiles('re')
    transformJSONFiles('lp')

    get_parser()
    createAllDependence()
    # depend_list=loadDependenceInformation('B/dependences/re_train.dep')

    trainingEmbedding(300,'re',True)
    createCluster(100,'re')
    trainingEmbedding(300,'lp',True)
    createCluster(200,'lp')

    all_terms,all_offsets,origin_text_test,true_offsets=evaluate(False,'re')
    all_terms,all_offsets,origin_text_test,true_offsets=evaluate(False,'lp')

    createAllForPol(d_type='re',context=5)
    createAllForPol(d_type='lp',context=5)

    examByML('re','LR')
    examByML('lp','LR')
    examByML('re','SVM')
    examByML('lp','SVM')
    examByML('re','MLP')
    examByML('lp','MLP')

    examByNN('re', 'cnn')
    examByNN('lp', 'cnn')
    examByNN('re', 'lstm')
    examByNN('lp', 'lstm')