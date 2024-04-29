# -*- coding: utf-8 -*-
from . import preProcessing
from . import entity
import logging
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
import pickle
import pandas
from tqdm import tqdm


# Training for word2vec
def trainingForWords(texts,fnum,filepath,sg,hs):
    # split text
    sentences=[text.lower().split(' ') for text in texts]

    # start training
    num_features=fnum # Dimension of a word vector
    min_word_count=3 # Minimum word frequency threshold
    num_workers=8 # Number of threads, set to 1 if random seeds needed
    context=10 # Window size
    downsampling=1e-3 # downsampling scale
    num_iter=50 # number of iteration
    hs=hs
    sg=sg # If skip-gram model needed
    
    model_path=filepath
    sentences_with_pbar = tqdm(sentences)
    model=Word2Vec(sentences_with_pbar,workers=num_workers,hs=hs,
                   vector_size=num_features,min_count=min_word_count,seed=77,epochs=num_iter,
                   window=context,sample=downsampling,sg=sg)
    # Lock trained word2vec
    model.init_sims(replace=True)
    model.save(model_path)
    print('Train ended')
    return model
    
# Load word2vec model
def loadForWord(filepath):
    model=Word2Vec.load(filepath)
    print('Finish reading word2vec')
    return model


def trainingEmbedding(vector_len=150,d_type='re',add_extra=False):
    if d_type=='re':
        d_name='Restaurants'
        extraFile='Dataset/extra/yelp/Restaurants_Raw.csv'
    else:
        d_name='LapTops'
        extraFile='Dataset/extra/amzon/LapTops_Raw.csv'
    
    print('------Training the Word2Vec of %s Data------'%d_name)
    train_corpus=preProcessing.loadXML('Dataset/semeval14/%s_Train_v2.xml'%d_name)
    test_corpus=preProcessing.loadXML('Dataset/semeval14/%s_Test_Data_PhaseA.xml'%d_name)
    
    corpus=train_corpus.corpus
    corpus=train_corpus.corpus+test_corpus.corpus
    
    del train_corpus
    del test_corpus
    
    bio_entity=entity.BIO_Entity(corpus,d_type)    
    texts=bio_entity.texts
    
    if add_extra==True:
        print('Adding extra data:%s'%extraFile)
        extra_csv=pandas.read_csv(extraFile)
        extra_texts=list(extra_csv['text'])
        texts=texts+extra_texts
        del extra_csv
        del extra_texts
    
    print('Training WordEmbedding')
    trainingForWords(texts,vector_len,'B/model/%s.w2v'%d_name,1,0)
    print('Training WordEmbedding_CBOW')
    trainingForWords(texts,vector_len,'B/model/%s.w2v_cbow'%d_name,0,0)
    
   
# Create cluster for w2v
def kmeansClusterForW2V(filepath,outpath,cluster_num):
    W2Vmodel=loadForWord(filepath)
    vocab=list(W2Vmodel.wv.key_to_index.keys())
    vectors=[W2Vmodel.wv[vocab[i]] for i in (range(len(vocab)))]
    print('Start clustering')
    clf=KMeans(n_clusters=cluster_num,random_state=77)
    clf.fit(vectors)
    dict_re={vocab[i]:clf.labels_[i] for i in range(len(vocab))}
    with open(outpath,'wb') as f:
        pickle.dump(dict_re,f)
    return dict_re
    
def loadDict(dictpath):
    print('Loading dict from: %s'%dictpath)
    with open(dictpath,'rb') as f:
        dict_re=pickle.load(f)
    return dict_re
    
def createCluster(cluster_num=10,d_type='re'):
    if d_type=='re':
        d_name='Restaurants'
    else:
        d_name='LapTops'
            
    print('Create cluster for w2v')
    kmeansClusterForW2V('B/model/%s.w2v'%d_name,'B/cluster/%s_w2v.pkl'%d_type,cluster_num)
    print('Create cluster for w2v_cbow')
    kmeansClusterForW2V('B/model/%s.w2v_cbow'%d_name,'B/cluster/%s_w2v_c.pkl'%d_type,cluster_num)