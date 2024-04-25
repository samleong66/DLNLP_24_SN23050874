# -*- coding: utf-8 -*-
import nltk
from itertools import chain
from collections import Counter
from . import preProcessing
import pycrfsuite
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# Converts text into a format that CRFsuite can recognize
def word2features(sent, i,window_size=2):
    word = sent[i]['word']
    postag = sent[i]['pos']
    lemm=sent[i]['lemm']

    w2v_c=sent[i]['w2v_c']
    w2v_c_c=sent[i]['w2v_c_c']

    amod_l=sent[i]['amod_l']
    nsubj_r=sent[i]['nsubj_r']
    dobj_r=sent[i]['dobj_r']

    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-2:]=' + word[-2:],
        'word[-3:]=' + word[-3:],
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'lemm='+lemm,    

        'amod_l=%s'%amod_l,
        'nsubj_r=%s'%nsubj_r,
        'dobj_r=%s'%dobj_r,   
        
        'w2v_c=%d'%w2v_c,
        'w2v_c_c=%d'%w2v_c_c,

    ]
    
    for j in range(1,window_size+1):
        if i-j>=0:
            word1 = sent[i-j]['word']
            postag1 = sent[i-j]['pos']
            w2v_c1=sent[i-j]['w2v_c']

            features.extend([
                '-%d:word.lower=%s'%(j,word1.lower()),
                '-%d:postag=%s'%(j,postag1),
                '-%d:word.istitle=%s'%(j,word1.istitle()),
                '-%d:w2v_c=%s'%(j,w2v_c1),
            ])
            
        if i+j<=len(sent)-1:
            word1 = sent[i+j]['word']
            postag1 = sent[i+j]['pos']
            w2v_c1=sent[i+j]['w2v_c']

            features.extend([
                '+%d:word.lower=%s'%(j,word1.lower()),
                '+%d:postag=%s'%(j,postag1),
                '+%d:word.istitle=%s'%(j,word1.istitle()),
                '+%d:w2v_c=%s'%(j,w2v_c1),
            ])
    
    if i==0:
        features.append('BOS')
    if i==len(sent)-1:
        features.append('EOS')
                
    return features
    
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
            
def crfFormat_X(train_X,test_X):
    print('Transforming features to CRF')
    train_X = [sent2features(s) for s in train_X]
    test_X = [sent2features(s) for s in test_X]
    return train_X,test_X
    
def train_CRF(train_X,train_Y):
    trainer=pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(train_X, train_Y):
        trainer.append(xseq, yseq)  
    trainer.set_params({
        #'c1': 1.0,   # coefficient for L1 penalty
        #'c2': 1e-3,  # coefficient for L2 penalty
        #'max_iterations': 75,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    
    trainer.params()
    print('Start Training CRF')
    trainer.train('B/model/temp.crfsuite')
    trainer.logparser.last_iteration
    print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])
    
def tag_CRF(test_X):
    tagger = pycrfsuite.Tagger()
    tagger.open('B/model/temp.crfsuite')
    print('Tagging features')
    predict_Y=[tagger.tag(xseq) for xseq in test_X]
    return predict_Y,tagger
    
def report_CRF(y_true, y_pred):
    lb=LabelBinarizer()
    
    #used to convert a string to the list 'aaa' - > [' a ', 'a', 'a']
    y_true_combined=lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pre_combined=lb.transform(list(chain.from_iterable(y_pred)))
    # Exclude 0
    tagset = set(lb.classes_) - {'O'}
    tagset=list(tagset)
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}                
    
    return classification_report(
        y_true_combined,
        y_pre_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
        digits=4
    )
    
def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr)) 
        
# Get the Aspect Term through bio_tag
def getTermsFromYSeq(yseq,text):
    terms=[]
    flag=0
    term=''
    words=nltk.word_tokenize(text)
    for i in range(len(yseq)):
        if flag==0:
            if yseq[i]=='O':
                continue
            elif yseq[i]=='I':
                print('%s :Wrong O->I')
                continue
            elif yseq[i]=='B':
                term+=words[i]
                flag=1
        elif flag==1:
            if yseq[i]=='O':
                terms.append(term)
                term=''
                flag=0
            elif yseq[i]=='I':
                term+=' '
                term+=words[i]
                flag=2
                continue
            elif yseq[i]=='B':
                terms.append(term)
                term=''
        elif flag==2:
            if yseq[i]=='O':
                terms.append(term)
                term=''
                flag=0
            elif yseq[i]=='I':
                term+=' '
                term+=words[i]
                continue
            elif yseq[i]=='B':
                terms.append(term)
                term=''
                flag=1
    return terms   
    
def getOffestFromText(terms,text):
    offsets=[]
    for term in terms:
        try:
            t_from=text.index(term)
            t_to=t_from+len(term)
            offsets.append({'from':str(t_from),'to':str(t_to)})
        except:
            print(text)
            print("\r\nAn AspectTerm match failed: %s"%term) 
    return offsets
    
def semEvalValidate(pred_offsets,true_offsets, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(len(true_offsets)):
            # True
            cor = true_offsets[i]
            # Predictive
            pre = pred_offsets[i]
            common += len([a for a in pre if a in cor])
            retrieved += len(pre)
            relevant += len(cor)
        p = common / retrieved if retrieved > 0 else 0.
        r = common / relevant
        f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        print('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)' %
              (p,r,f1,common,retrieved,relevant))
    
def evaluate(detail=False,d_type='re'): 
    if d_type=='re':
        d_name='Restaurants'
    else:
        d_name='LapTops'
    
    print('Loading train dataset')
    train_corpus=preProcessing.loadXML('Dataset/semeval14/%s_Train_v2.xml'%d_name)
    train_bio=preProcessing.createBIOClass(train_corpus.corpus,d_type)
    dep_path='B/dependences/%s_train.dep'%d_type
    train_bio.createDependenceFeature(dep_path)
    train_X,train_Y=train_bio.getFeaturesAndLabels()
    
    print('Loading testing dataset')
    test_corpus=preProcessing.loadXML('Dataset/semeval14/%s_Test_Data_phaseB.xml'%d_name)
    test_bio=preProcessing.createBIOClass(test_corpus.corpus,d_type)
    dep_path='B/dependences/%s_test.dep'%d_type
    test_bio.createDependenceFeature(dep_path)
    test_X,test_Y=test_bio.getFeaturesAndLabels()
    
    true_offsets=[] 
    for i in range(len(test_bio.instances)):
        offset=[a.offset for a in test_bio.instances[i].aspect_terms]
        true_offsets.append(offset)      
    origin_text_test=test_bio.origin_texts
      
    train_X,test_X=crfFormat_X(train_X,test_X)
    train_CRF(train_X,train_Y)
    predict_Y,tagger=tag_CRF(test_X)
    report=report_CRF(test_Y,predict_Y)
    print('\n--------Results based on BIO---------')
    print(report)
    
    if detail==True:
        print('\n--------Other information based on BIO---------')
        info=tagger.info()  
        print("The most likely state transition:")
        print_transitions(Counter(info.transitions).most_common(10))
        print("\nThe least likely state transition:")
        print_transitions(Counter(info.transitions).most_common()[-10:])
        print("\nThe strongest feature correlation:")
        print_state_features(Counter(info.state_features).most_common(10))
        print("\nThe weakest feature correlation:")
        print_state_features(Counter(info.state_features).most_common()[-10:])
    
    all_terms=[]
    for i in range(len(origin_text_test)):
        all_terms.append(getTermsFromYSeq(predict_Y[i],origin_text_test[i]))
    all_offsets=[]
    for i in range(len(origin_text_test)):
        all_offsets.append(getOffestFromText(all_terms[i],origin_text_test[i]))
        
    print('\n--------SemEval Report---------')
    semEvalValidate(all_offsets,true_offsets, b=1)
    
    return all_terms,all_offsets,origin_text_test,true_offsets
