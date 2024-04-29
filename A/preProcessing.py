# -*- coding: utf-8 -*-
import os
import pandas
import json

from tqdm import tqdm
from . import entity
import xml.etree.ElementTree as ET
import sys
import nltk
from nltk.stem import SnowballStemmer
from nltk.parse.stanford import StanfordDependencyParser
import nltk.data  
import pickle

wn=nltk.WordNetLemmatizer()
sp=SnowballStemmer('english')

#Validates that the document is a conformant XML, returning all instances with all AspectCategories.
def validateXML(filename):
    #Parsing the XML to find all the sentences
    elements=ET.parse(filename).getroot().findall('sentence')
    aspects=[]
    for e in elements:
        #Get all the aspectTerms in each instance.
        for ats in e.findall('aspectTerms'):
            if ats is not None:
                for a in ats.findall('aspectTerm'):
                    aspects.append(entity.AspectTerm('','',[]).create(a).term)
    return elements,aspects
    
# Load corpus entities from files
def loadXML(filename):
    try:
        elements,aspects=validateXML(filename)
        print('XML with %d sentences, %d AspectTerms, %d different AspectTerms'
              %(len(elements),len(aspects),len(list(set(aspects)))))
    except:
        print("XML illegal", sys.exc_info()[0])
        raise
    corpus=entity.Corpus(elements)
    return corpus  
  
# Get full information about BIO
def createBIOClass(instances,d_type):
    bio_entity=entity.BIO_Entity(instances,d_type)
    bio_entity.createBIOTags()
    bio_entity.createPOSTags()
    bio_entity.createLemm()
    bio_entity.createW2VCluster()
    return bio_entity
    
def cutFileForBIO(filename,threshold=0.8,shuffle=False,d_type='re'):
    corpus=loadXML(filename)
    train_corpus,test_corpus=corpus.split(threshold, shuffle)
    print('------ Cutting the raw dataset is complete, start constructing features ------')
    
    bio_train=createBIOClass(train_corpus,d_type)
    print('----- Training BIO construction completed, start constructing test BIO -------')
    bio_test=createBIOClass(test_corpus,d_type)
    print('Test BIO construction completed')
    
    print('----- Getting features and markers for the training set -------')
    train_X,train_Y=bio_train.getFeaturesAndLabels()
    print('----- Get features and markers for the test set -------')
    test_X,test_Y=bio_test.getFeaturesAndLabels()
    
    true_offset=[]
    
    for i in range(len(bio_test.instances)):
        offset=[a.offset for a in bio_test.instances[i].aspect_terms]
        true_offset.append(offset)
        
    origin_text=bio_test.texts
    
    return train_X,train_Y,test_X,test_Y,true_offset,origin_text
    
# Read additional data and convert to CSV file
def transformJSONFiles(d_type='re',all_text=False):
    if d_type=='re':
        filepath=r'Dataset\extra\yelp\yelp_academic_dataset_review.json'
        outpath=r'Dataset\extra\yelp\Restaurants_Raw.csv'
        text_item='text'
    else:
        filepath=r'Dataset\extra\amzon\Electronics_5.json'
        outpath=r'Dataset\extra\amzon\LapTops_Raw.csv'
        text_item='reviewText'

    if os.path.exists(outpath):
        return
        
    print('Start loading JSON and get its text .....')

    review_list=[]
    if d_type=='re':
        with open(filepath,'r', encoding='utf-8') as f:
            items_list=f.readlines()

            if all_text==False:
                items_list=items_list[:150000]

            with tqdm(total=len(items_list), desc='Transforming extra data for Restaurant') as pbar:
                for item in items_list:
                    json_dict=json.loads(item)
                    review_list.append(' '.join(nltk.word_tokenize(json_dict[text_item])))
                    pbar.update(1)
    else:
        count = 0
        with open(filepath,'r') as f:
            items_list=f.readlines()

            with tqdm(total=90000, desc='Transforming extra data for Labtop') as pbar:
                for item in items_list:
                    json_dict=json.loads(item)
                    words=nltk.word_tokenize(json_dict[text_item])
                    words1=[word.lower() for word in words]
                    if 'notebook' in words1 or 'laptop' in words1:
                        review_list.append(' '.join(words))
                        count += 1
                        pbar.update(1)
                        if all_text==False and count > 90000:
                            break
                    
            
    # Transform to CSV
    output=pandas.DataFrame({'text':review_list})
    output.to_csv(outpath,index=False)
    print('Tansformation Completed')
    
def createDependenceInformation(inputfile,outputfile):
    corpus=loadXML(inputfile)
    texts=corpus.texts
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sent_num=[]
    all_sents=[]
    print('Clause begin')
    for text in texts:
        sents=tokenizer.tokenize(text)  
        sents=[nltk.word_tokenize(sent) for sent in sents]
        all_sents.extend(sents)
        sent_num.append(len(sents))
    print('Parse begin')
    eng_parser = StanfordDependencyParser(r"B\stanford parser\stanford-parser.jar",
                                          r"B\stanford parser\stanford-parser-3.6.0-models.jar")
    res_list = []
    res_list=list(eng_parser.parse_sents(all_sents))
    res_list=[list(i) for i in res_list]
    depends=[]
    for item in res_list:
        depend=[]
        for row in item[0].triples():
            depend.append(row)
        depends.append(depend)
    print('Spliting begin')
    index=0
    depend_list=[]
    for num in sent_num:
       depend_list.append(depends[index:index+num])
       index+=num
       
    print('Saving dependence list')
    with open(outputfile,'wb') as f:
        pickle.dump(depend_list,f)
    print('Done')
        
def loadDependenceInformation(filepath):
    print('Loading dependence list from：%s'%filepath)
    with open(filepath,'rb') as f:
        depend_list=pickle.load(f)
    return depend_list
    
def createAllDependence():
    re_train_dep_file = r'B\dependences\re_train.dep'
    re_test_dep_file = r'B\dependences\re_test.dep'
    lp_train_dep_file = r'B\dependences\lp_train.dep'
    lp_test_dep_file = r'B\dependences\lp_train.dep'

    print('Calculating dependence of Restaurants data')
    if not os.path.exists(re_train_dep_file):
        createDependenceInformation(r'Dataset\semeval14\Restaurants_Train_v2.xml',re_train_dep_file)
    if not os.path.exists(re_test_dep_file):
        createDependenceInformation(r'Dataset\semeval14\Restaurants_Test_Data_phaseB.xml',re_test_dep_file)
    
    print('Calculating dependence of Laptops data')
    if not os.path.exists(lp_train_dep_file):
        createDependenceInformation(r'Dataset\semeval14\LapTops_Train_v2.xml',lp_train_dep_file)
    if not os.path.exists(lp_test_dep_file):
        createDependenceInformation(r'Dataset\semeval14\LapTops_Test_Data_phaseB.xml',lp_test_dep_file)